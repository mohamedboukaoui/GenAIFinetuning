"""
╔══════════════════════════════════════════════════════════════════╗
║   Fine-Tuning McKinsey — UNSLOTH (LoRA fp16 accéléré)           ║
║   Gain : 2x plus rapide, -40% VRAM vs HuggingFace natif         ║
╚══════════════════════════════════════════════════════════════════╝

QU'EST-CE QU'UNSLOTH ?
────────────────────────
Unsloth réécrit les kernels d'attention et de LoRA en Triton/CUDA.
Résultat sur un Mistral-7B :
  • Vitesse d'entraînement : +200% (2x plus rapide)
  • VRAM économisée        : -40%
  • Perte de précision     : AUCUNE (même résultat que HuggingFace)

C'est aujourd'hui le standard de facto pour le fine-tuning local
de modèles 7B.

INSTALLATION
────────────
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
"""

# ══════════════════════════════════════════════════════════════════
# IMPORTS UNSLOTH
# ══════════════════════════════════════════════════════════════════

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import json, re
from pathlib import Path
import PyPDF2

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

BASE_MODEL  = "unsloth/mistral-7b-instruct-v0.2"  # version optimisée Unsloth
MAX_SEQ_LEN = 2048     # Unsloth gère RoPE scaling → peut monter plus haut
DTYPE       = torch.float16
LOAD_IN_4BIT = False   # LoRA fp16 pur (pas de quantification)

PDF_FOLDER  = "./mckinsey_pdfs"
ADAPTER_DIR = "./mckinsey_unsloth_lora"

# ══════════════════════════════════════════════════════════════════
# ÉTAPE 1 — Chargement du modèle avec Unsloth
# ══════════════════════════════════════════════════════════════════

def load_with_unsloth():
    """
    FastLanguageModel.from_pretrained() remplace AutoModelForCausalLM.
    Il applique automatiquement les optimisations Unsloth :
      - Flash Attention 2 (si GPU Ampere+)
      - Kernels LoRA Triton fusionnés
      - RoPE scaling automatique
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,    # False = LoRA fp16 pur
    )

    # Application LoRA via Unsloth (kernel fusionné)
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",  # version optimisée d'Unsloth
        random_state=42,
        use_rslora=False,    # True = Rank-Stabilized LoRA (expérimental)
        loftq_config=None,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"🔧 Paramètres entraînables : {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model, tokenizer


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 2 — Préparation du dataset (format ChatML pour Unsloth)
# ══════════════════════════════════════════════════════════════════
"""
Unsloth recommande le format ChatML plutôt qu'Alpaca.
Avantage : meilleure compatibilité avec Mistral-Instruct
qui a été entraîné avec des tokens spéciaux [INST]...[/INST].
"""

MCKINSEY_SYSTEM_PROMPT = (
    "You are a McKinsey & Company senior consultant. "
    "Your responses are precise, data-driven, and authoritative. "
    "You structure arguments clearly, use executive-level vocabulary, "
    "and always ground insights in measurable business impact."
)


def extract_and_prepare(pdf_folder: str) -> Dataset:
    """Extrait les PDFs et crée le dataset au format ChatML."""
    examples = []

    for pdf_path in Path(pdf_folder).glob("*.pdf"):
        text = ""
        with open(pdf_path, "rb") as f:
            for page in PyPDF2.PdfReader(f).pages:
                text += page.extract_text() or ""

        text = re.sub(r"McKinsey\s*&\s*Company\s*\d*", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text).strip()

        words = text.split()
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i+400])
            sentences = chunk.split(". ")
            if len(sentences) >= 4:
                mid = len(sentences) // 2
                user_msg = ". ".join(sentences[:mid]) + "."
                assistant_msg = ". ".join(sentences[mid:])

                # Format ChatML / messages
                examples.append({
                    "messages": [
                        {"role": "system",    "content": MCKINSEY_SYSTEM_PROMPT},
                        {"role": "user",      "content": f"Continue this analysis:\n\n{user_msg}"},
                        {"role": "assistant", "content": assistant_msg},
                    ]
                })
            i += 350  # overlap de 50 tokens

        print(f"  {pdf_path.name} → {len(examples)} exemples cumulés")

    ds = Dataset.from_list(examples).train_test_split(test_size=0.1, seed=42)
    print(f"\n✅ Dataset : {len(examples)} exemples total")
    print(f"📊 Train: {len(ds['train'])} | Eval: {len(ds['test'])}")
    return ds


def apply_chat_template(tokenizer, dataset):
    """Applique le template de chat Mistral au dataset."""
    tokenizer = get_chat_template(tokenizer, chat_template="mistral")

    def format_row(row):
        text = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    return dataset.map(format_row)


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 3 — Entraînement Unsloth
# ══════════════════════════════════════════════════════════════════

def train_unsloth(model, tokenizer, dataset):
    """
    DIFFÉRENCES vs HuggingFace SFTTrainer standard :
      - packing=True     : Unsloth pack efficacement les séquences
      - dataset_text_field : pointe vers le champ "text" formaté
      - Les kernels Triton s'activent automatiquement
    """
    args = SFTConfig(
        output_dir="./unsloth_checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=4,   # Unsloth permet des batchs plus grands
        gradient_accumulation_steps=2,
        optim="adamw_8bit",              # AdamW 8-bit d'Unsloth (VRAM réduite)
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,
        bf16=False,
        max_seq_length=MAX_SEQ_LEN,
        packing=True,
        dataset_text_field="text",
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=3,
        seed=42,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=args,
    )

    # Affiche les stats mémoire avant l'entraînement
    gpu_stats = torch.cuda.get_device_properties(0)
    reserved  = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
    print(f"💾 GPU : {gpu_stats.name} | VRAM totale : {gpu_stats.total_memory/1024**3:.1f}GB")
    print(f"💾 VRAM réservée avant train : {reserved}GB")

    print("\n🚀 Entraînement Unsloth LoRA fp16 ...")
    trainer_stats = trainer.train()

    used = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
    mins = round(trainer_stats.metrics["train_runtime"] / 60, 2)
    print(f"\n✅ Terminé en {mins} min | VRAM max utilisée : {used}GB")
    return trainer


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 4 — Sauvegarde Unsloth (GGUF ou LoRA)
# ══════════════════════════════════════════════════════════════════

def save_unsloth(model, tokenizer):
    """
    Unsloth propose plusieurs formats de sauvegarde :
    
    Option A : Adaptateurs LoRA seuls (HuggingFace compatible)
    Option B : GGUF (pour Ollama / llama.cpp en local)
    Option C : Modèle fusionné fp16
    """
    # Option A : LoRA adaptateurs
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"💾 Adaptateurs LoRA → {ADAPTER_DIR}")

    # Option B : GGUF pour Ollama (décommenter pour activer)
    # model.save_pretrained_gguf(
    #     "mckinsey_gguf",
    #     tokenizer,
    #     quantization_method="q4_k_m",  # quantification post-training pour l'inférence
    # )
    # print("💾 GGUF sauvegardé → mckinsey_gguf/")

    # Option C : Modèle fusionné fp16 (décommenter pour activer)
    # model.save_pretrained_merged(
    #     "mckinsey_merged_fp16",
    #     tokenizer,
    #     save_method="merged_16bit",
    # )


# ══════════════════════════════════════════════════════════════════
# INFÉRENCE avec Unsloth (mode rapide)
# ══════════════════════════════════════════════════════════════════

def inference_unsloth(model, tokenizer, user_input: str) -> str:
    """Active le mode inférence rapide d'Unsloth."""
    FastLanguageModel.for_inference(model)  # active les kernels d'inférence

    messages = [
        {"role": "system",    "content": MCKINSEY_SYSTEM_PROMPT},
        {"role": "user",      "content": user_input},
    ]
    tokenizer = get_chat_template(tokenizer, chat_template="mistral")
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            input_ids=inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )

    new_tokens = out[0][inputs.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("  UNSLOTH — LoRA fp16 McKinsey Fine-Tuning")
    print("=" * 55)

    model, tokenizer = load_with_unsloth()

    print("\n📦 Préparation du dataset ...")
    raw_ds = extract_and_prepare(PDF_FOLDER)
    ds     = apply_chat_template(tokenizer, raw_ds)

    trainer = train_unsloth(model, tokenizer, ds)
    save_unsloth(model, tokenizer)

    # Test
    print("\n" + "═" * 55)
    print("  TEST D'INFÉRENCE")
    print("═" * 55)
    tests = [
        "AI adoption is growing fast in enterprise.",
        "Most digital transformation projects fail to deliver value.",
    ]
    for t in tests:
        print(f"\n📝 Input  : {t}")
        print(f"🤖 Output : {inference_unsloth(model, tokenizer, t)}")

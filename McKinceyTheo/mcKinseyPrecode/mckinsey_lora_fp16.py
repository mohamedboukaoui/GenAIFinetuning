"""
╔══════════════════════════════════════════════════════════════════╗
║   Fine-Tuning McKinsey Tone — LoRA Classique fp16               ║
║   Modèle  : Mistral-7B-Instruct-v0.2                            ║
║   Méthode : LoRA (fp16, SANS quantification)                    ║
║   Runtime : Azure ML Compute                                     ║
╚══════════════════════════════════════════════════════════════════╝

DIFFÉRENCE CLEF vs QLoRA
─────────────────────────
• QLoRA  : modèle de base en 4-bit → ~5GB VRAM  (mais perte légère de précision)
• LoRA   : modèle de base en fp16  → ~14GB VRAM (précision maximale)
• Ici on choisit LoRA pur → nécessite au minimum une A100 40GB ou 2x A10 24GB

PRÉREQUIS
─────────
pip install transformers peft trl accelerate datasets PyPDF2 sentencepiece
"""

# ══════════════════════════════════════════════════════════════════
# CONFIG GLOBALE
# ══════════════════════════════════════════════════════════════════

BASE_MODEL   = "mistralai/Mistral-7B-Instruct-v0.2"
PDF_FOLDER   = "./McKinseyReport"
OUTPUT_JSONL = "./mckinsey_dataset.jsonl"
ADAPTER_DIR  = "./mckinsey_lora_fp16"
OUTPUT_DIR   = "./checkpoints_lora_fp16"

# Hyperparamètres LoRA
LORA_R       = 16      # rang : expressivité des adaptateurs
LORA_ALPHA   = 32      # scaling = alpha/r = 2x (standard)
LORA_DROPOUT = 0.05

# Hyperparamètres entraînement
EPOCHS       = 3       # plus d'epochs possible car fp16 est plus stable
LR           = 1e-4    # LR plus faible que QLoRA (fp16 = gradient plus précis)
BATCH_SIZE   = 4       # peut monter grâce à la précision fp16
GRAD_ACCUM   = 2       # effective batch = 4*2 = 8
MAX_SEQ_LEN  = 1024


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 1 — Préparation des données McKinsey
# ══════════════════════════════════════════════════════════════════

import os, json, re
from pathlib import Path
import PyPDF2
from datasets import Dataset

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)


def extract_pdf(path: str) -> str:
    text = ""
    with open(path, "rb") as f:
        for page in PyPDF2.PdfReader(f).pages:
            text += page.extract_text() or ""
    return text


def clean_text(text: str) -> str:
    text = re.sub(r"McKinsey\s*&\s*Company\s*\d*", "", text)
    text = re.sub(r"McKinsey\s+Quarterly", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return re.sub(r" {2,}", " ", text).strip()


def chunk_text(text: str, size: int = 400, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        c = " ".join(words[i : i + size])
        if len(c) > 80:
            chunks.append(c)
        i += size - overlap
    return chunks


def build_examples(chunks: list[str]) -> list[dict]:
    examples = []
    for chunk in chunks:
        sentences = chunk.split(". ")
        if len(sentences) < 4:
            continue
        mid = len(sentences) // 2
        prompt  = ". ".join(sentences[:mid]) + "."
        completion = ". ".join(sentences[mid:])

        # Format 1 : continuation de l'analyse
        examples.append({"text": ALPACA_TEMPLATE.format(
            instruction="Continue the following McKinsey consulting analysis:",
            input=prompt,
            output=completion,
        )})

        # Format 2 : reformulation dans le ton McKinsey
        examples.append({"text": ALPACA_TEMPLATE.format(
            instruction=(
                "Rewrite the following business insight in the McKinsey style: "
                "authoritative tone, data-driven, structured, executive-level vocabulary."
            ),
            input=chunk,
            output=chunk,
        )})
    return examples


def prepare_dataset() -> Dataset:
    pdfs = list(Path(PDF_FOLDER).glob("*.pdf"))
    print(f"📄 {len(pdfs)} PDFs trouvés")
    all_examples = []
    for pdf in pdfs:
        text = clean_text(extract_pdf(str(pdf)))
        chunks = chunk_text(text)
        examples = build_examples(chunks)
        all_examples.extend(examples)
        print(f"  {pdf.name} → {len(chunks)} chunks, {len(examples)} exemples")

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n✅ {len(all_examples)} exemples → {OUTPUT_JSONL}")
    ds = Dataset.from_list(all_examples).train_test_split(test_size=0.1, seed=42)
    print(f"📊 Train: {len(ds['train'])} | Eval: {len(ds['test'])}")
    return ds


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 2 — Chargement du modèle en LoRA FP16 (PAS de quantification)
# ══════════════════════════════════════════════════════════════════
"""
POURQUOI fp16 et pas bf16 ici ?
─────────────────────────────────
• fp16 (float16) : standard sur les GPU Azure (V100, A10, A100)
• bf16 (bfloat16) : meilleur pour les très longs entraînements,
  mais nécessite Ampere (A100, H100) — préférer bf16 si disponible
• Sur Azure NC-series (V100) → fp16 obligatoire
• Sur Azure ND-series (A100) → switcher bf16=True dans SFTConfig
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


def load_model_lora_fp16():
    print(f"⬇️  Chargement de {BASE_MODEL} en fp16 ...")

    # Chargement en fp16 SANS BitsAndBytes (LoRA pur)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,   # ← fp16 natif, pas de quantification
        device_map="auto",            # répartit sur les GPUs disponibles
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Active le gradient checkpointing pour économiser la VRAM
    # (trade-off : +20% de temps, -30% de mémoire)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # Configuration LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        # Toutes les couches d'attention + FFN de Mistral
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        # En LoRA fp16, on peut aussi cibler les embeddings
        # modules_to_save=["embed_tokens", "lm_head"],  # décommenter si overfitting
    )

    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"🔧 Paramètres entraînables : {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    print(f"💾 VRAM estimée nécessaire : ~14-16 GB (fp16 Mistral-7B)")

    return model, tokenizer


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 3 — Entraînement SFT avec LoRA fp16
# ══════════════════════════════════════════════════════════════════

from trl import SFTTrainer, SFTConfig


def train(model, tokenizer, dataset):
    args = SFTConfig(
        output_dir=OUTPUT_DIR,

        # ── Durée ──────────────────────────────────────────────
        num_train_epochs=EPOCHS,

        # ── Batch & mémoire ────────────────────────────────────
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True,

        # ── Optimisation ───────────────────────────────────────
        optim="adamw_torch",         # AdamW standard (pas besoin de paged en fp16)
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,           # gradient clipping (important en fp16)

        # ── Précision ──────────────────────────────────────────
        fp16=True,                   # ← LoRA fp16
        bf16=False,

        # ── Séquences ──────────────────────────────────────────
        max_seq_length=MAX_SEQ_LEN,
        packing=True,
        dataset_text_field="text",

        # ── Logging & sauvegarde ───────────────────────────────
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        # ── Reproductibilité ───────────────────────────────────
        seed=42,
        report_to="none",            # mettre "azure_ml" si tu veux les métriques dans AzureML
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=args,
    )

    print("🚀 Démarrage de l'entraînement LoRA fp16 ...")
    trainer.train()
    print(f"✅ Terminé → {OUTPUT_DIR}")
    return trainer


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 4 — Sauvegarde et inférence
# ══════════════════════════════════════════════════════════════════

def save_and_test(trainer, tokenizer):
    # Sauvegarde des adaptateurs LoRA (~100MB en fp16 vs ~50MB en 4bit)
    trainer.model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"💾 Adaptateurs LoRA fp16 → {ADAPTER_DIR}")

    # Test rapide
    model = trainer.model
    model.eval()

    test_prompts = [
        "AI adoption in companies is growing fast.",
        "Supply chains are broken and need to be fixed.",
        "Most digital transformation projects fail.",
    ]

    print("\n" + "═" * 55)
    print("  ÉVALUATION QUALITATIVE — TON MCKINSEY")
    print("═" * 55)

    for text in test_prompts:
        prompt = ALPACA_TEMPLATE.format(
            instruction="Rewrite in McKinsey consulting style.",
            input=text,
            output="",
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(f"\n📝 Input  : {text}")
        print(f"🤖 Output : {response.strip()}")


# ══════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("  LoRA fp16 — McKinsey Tone Fine-Tuning")
    print("=" * 55)

    ds      = prepare_dataset()
    model, tokenizer = load_model_lora_fp16()
    trainer = train(model, tokenizer, ds)
    save_and_test(trainer, tokenizer)

"""
╔══════════════════════════════════════════════════════════════════╗
║   Fine-Tuning McKinsey — NVIDIA DGX Spark (Local)               ║
║   Config : DGX Spark = 1x Grace Blackwell GB200                  ║
║            128GB mémoire unifiée CPU+GPU                         ║
║   Méthode : LoRA fp16 multi-GPU via torchrun                     ║
╚══════════════════════════════════════════════════════════════════╝

LE DGX SPARK EN 2 MOTS
────────────────────────
Le DGX Spark (annoncé 2025) est la version compacte de DGX.
Specs clés pour le fine-tuning :
  • Grace Blackwell GB200 SoC
  • 128GB mémoire unifiée (CPU + GPU partagent le même espace)
  • NVLink-C2C entre CPU et GPU (bande passante massive)
  • FP16 natif + TF32

CONSÉQUENCE POUR LE FINE-TUNING :
  • Un Mistral-7B en fp16 prend ~14GB → rentrera facilement
  • Pas besoin de quantification
  • Pas besoin de gradient checkpointing (assez de mémoire)
  • batch_size peut monter à 8-16 sans problème

LANCEMENT (depuis le terminal DGX Spark) :
  torchrun --nproc_per_node=1 mckinsey_dgx_spark.py

  Ou avec Accelerate :
  accelerate launch --config_file accelerate_config.yaml mckinsey_dgx_spark.py
"""

import os
import torch
import json
import re
from pathlib import Path
from datasets import Dataset
import PyPDF2
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# ══════════════════════════════════════════════════════════════════
# CONFIG DGX SPARK
# ══════════════════════════════════════════════════════════════════

BASE_MODEL   = "mistralai/Mistral-7B-Instruct-v0.2"
PDF_FOLDER   = "./mckinsey_pdfs"
ADAPTER_DIR  = "./mckinsey_dgx_lora"
OUTPUT_DIR   = "./dgx_checkpoints"

# Sur DGX Spark : on peut se permettre des paramètres plus agressifs
LORA_R       = 32      # rang plus élevé (mémoire suffisante)
LORA_ALPHA   = 64
BATCH_SIZE   = 8       # batch plus grand grâce aux 128GB
GRAD_ACCUM   = 1       # plus besoin d'accumulation avec un grand batch
MAX_SEQ_LEN  = 2048    # séquences plus longues possibles
EPOCHS       = 3
LR           = 1e-4


# ══════════════════════════════════════════════════════════════════
# UTILITAIRE : vérification de l'environnement DGX
# ══════════════════════════════════════════════════════════════════

def check_dgx_environment():
    """Vérifie que l'environnement DGX Spark est bien configuré."""
    print("🔍 Vérification environnement DGX Spark ...")
    print(f"  PyTorch  : {torch.__version__}")
    print(f"  CUDA     : {torch.version.cuda}")

    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA non disponible — vérifier les drivers NVIDIA")

    n_gpu = torch.cuda.device_count()
    print(f"  GPUs     : {n_gpu}")
    for i in range(n_gpu):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / 1024**3
        print(f"  GPU {i}    : {props.name} — {total_gb:.1f}GB")

    # DGX Spark a une mémoire unifiée — vérification
    total_ram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if total_ram >= 64:
        print(f"  ✅ Mémoire suffisante ({total_ram:.0f}GB) — LoRA fp16 full possible")
    else:
        print(f"  ⚠️  Mémoire limitée ({total_ram:.0f}GB) — envisager gradient_checkpointing=True")

    print()


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 1 — Dataset (même logique, paramètres DGX-optimisés)
# ══════════════════════════════════════════════════════════════════

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

SYSTEM_PROMPT = (
    "You are a McKinsey & Company senior consultant. "
    "Your responses are precise, data-driven, authoritative, "
    "and grounded in measurable business impact."
)


def prepare_dataset_dgx() -> Dataset:
    """
    Sur DGX Spark on peut utiliser des chunks plus longs
    grâce à la mémoire unifiée et à MAX_SEQ_LEN=2048.
    """
    pdfs = list(Path(PDF_FOLDER).glob("*.pdf"))
    print(f"📄 {len(pdfs)} PDFs trouvés")
    examples = []

    for pdf_path in pdfs:
        text = ""
        with open(pdf_path, "rb") as f:
            for page in PyPDF2.PdfReader(f).pages:
                text += page.extract_text() or ""

        text = re.sub(r"McKinsey\s*&\s*Company\s*\d*", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text).strip()

        words = text.split()
        i = 0
        # Chunks plus longs : 600 mots (vs 400 sur Colab)
        while i < len(words):
            chunk = " ".join(words[i:i+600])
            sentences = chunk.split(". ")
            if len(sentences) >= 4:
                mid = len(sentences) // 2
                examples.append({"text": ALPACA_TEMPLATE.format(
                    instruction="Continue this McKinsey consulting analysis:",
                    input=". ".join(sentences[:mid]) + ".",
                    output=". ".join(sentences[mid:]),
                )})
            i += 550
        print(f"  {pdf_path.name} → {len(examples)} exemples cumulés")

    ds = Dataset.from_list(examples).train_test_split(test_size=0.1, seed=42)
    print(f"✅ {len(examples)} exemples | Train: {len(ds['train'])} | Eval: {len(ds['test'])}")
    return ds


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 2 — Chargement modèle optimisé DGX Spark
# ══════════════════════════════════════════════════════════════════

def load_model_dgx():
    """
    Sur DGX Spark, on charge en fp16 SANS quantification et SANS
    gradient checkpointing → vitesse maximale grâce aux 128GB.
    
    Le device_map="auto" distribue automatiquement sur la mémoire
    unifiée du DGX Spark (CPU+GPU partagent le même bus NVLink).
    """
    print(f"⬇️  Chargement {BASE_MODEL} en fp16 (DGX Spark) ...")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        # DGX Spark supporte Flash Attention 2 (Blackwell)
        attn_implementation="flash_attention_2",
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model.enable_input_require_grads()
    # Pas de gradient_checkpointing sur DGX : mémoire suffisante,
    # et sans checkpointing le forward pass est plus rapide

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,          # rang 32 (possible grâce à la mémoire)
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"🔧 Paramètres entraînables : {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model, tokenizer


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 3 — Entraînement optimisé DGX Spark
# ══════════════════════════════════════════════════════════════════

def train_dgx(model, tokenizer, dataset):
    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,

        # ── Batch DGX (128GB = batch large possible) ────────────
        per_device_train_batch_size=BATCH_SIZE,   # 8 (vs 2-4 sur Colab)
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,   # 1 (pas besoin d'accumuler)
        gradient_checkpointing=False,              # désactivé : mémoire ok

        # ── Optimisation ────────────────────────────────────────
        optim="adamw_torch_fused",    # version fusionnée (plus rapide sur Blackwell)
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,

        # ── Précision ───────────────────────────────────────────
        fp16=True,
        bf16=False,
        # Note : sur Blackwell (DGX Spark), bf16 est aussi supporté
        # Passer bf16=True, fp16=False pour de meilleures performances

        # ── Séquences ───────────────────────────────────────────
        max_seq_length=MAX_SEQ_LEN,   # 2048 (vs 1024 sur Colab)
        packing=True,
        dataset_text_field="text",

        # ── Logging & sauvegarde ────────────────────────────────
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=30,
        save_steps=50,
        save_total_limit=5,
        load_best_model_at_end=True,

        # ── Parallélisme DGX ────────────────────────────────────
        dataloader_num_workers=4,      # parallélisme CPU pour le chargement
        dataloader_pin_memory=True,    # épingle les tenseurs en RAM pour NVLink

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

    # Stats mémoire
    for i in range(torch.cuda.device_count()):
        reserved = torch.cuda.max_memory_reserved(i) / 1024**3
        print(f"💾 GPU {i} VRAM réservée : {reserved:.2f}GB")

    print("🚀 Entraînement sur DGX Spark ...")
    stats = trainer.train()

    mins = round(stats.metrics["train_runtime"] / 60, 2)
    print(f"✅ Terminé en {mins} minutes")

    for i in range(torch.cuda.device_count()):
        used = torch.cuda.max_memory_reserved(i) / 1024**3
        print(f"💾 GPU {i} VRAM max utilisée : {used:.2f}GB")

    return trainer


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 4 — Sauvegarde + Export GGUF (pour usage local Ollama)
# ══════════════════════════════════════════════════════════════════

def save_and_export(model, tokenizer):
    # Sauvegarde LoRA
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"💾 Adaptateurs LoRA → {ADAPTER_DIR}")

    # Fusion et export (pour déploiement sur Ollama local)
    print("\n🔀 Fusion LoRA + modèle de base ...")
    merged = model.merge_and_unload()

    merged_dir = "./mckinsey_merged_fp16"
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    print(f"💾 Modèle fusionné → {merged_dir}")
    print("\n💡 Pour utiliser dans Ollama :")
    print(f"   ollama create mckinsey-7b -f ./Modelfile")
    print(f"   # Modelfile : FROM {merged_dir}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("  DGX SPARK — LoRA fp16 McKinsey Fine-Tuning")
    print("=" * 55 + "\n")

    check_dgx_environment()
    ds = prepare_dataset_dgx()
    model, tokenizer = load_model_dgx()
    trainer = train_dgx(model, tokenizer, ds)
    save_and_export(trainer.model, tokenizer)

    print("\n✅ Pipeline DGX Spark terminé !")

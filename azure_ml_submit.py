# ╔══════════════════════════════════════════════════════════════════╗
# ║   Azure ML — Config + Script de lancement                       ║
# ║   Pour soumettre le job de fine-tuning depuis ta machine locale  ║
# ╚══════════════════════════════════════════════════════════════════╝
#
# PRÉREQUIS
# ─────────
# pip install azure-ai-ml azure-identity
# az login
#
# UTILISATION
# ───────────
# 1. Remplis les variables dans CONFIG ci-dessous
# 2. python azure_ml_submit.py
#
# CHOIX DE COMPUTE AZURE pour LoRA fp16 Mistral-7B
# ─────────────────────────────────────────────────
# Standard_NC24ads_A100_v4  → 1x A100 80GB  ← recommandé (1 GPU, ~3€/h)
# Standard_NC48ads_A100_v4  → 2x A100 80GB  ← si tu veux aller vite
# Standard_ND96asr_v4       → 8x A100 40GB  ← overkill pour 7B

from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.entities import (
    AmlCompute,
    Environment,
    BuildContext,
)
from azure.identity import DefaultAzureCredential
from azure.ai.ml.constants import AssetTypes
import os

# ══════════════════════════════════════════════════════════════════
# CONFIG — À remplir avec tes infos Azure
# ══════════════════════════════════════════════════════════════════

CONFIG = {
    "subscription_id" : "TON_SUBSCRIPTION_ID",
    "resource_group"  : "TON_RESOURCE_GROUP",
    "workspace_name"  : "TON_WORKSPACE_NAME",
    "compute_name"    : "gpu-cluster-a100",
    "compute_size"    : "Standard_NC24ads_A100_v4",  # 1x A100 80GB
    "experiment_name" : "mckinsey-lora-fp16",
    "hf_token"        : os.environ.get("HF_TOKEN", ""),  # export HF_TOKEN=hf_xxx
}

# ══════════════════════════════════════════════════════════════════
# CONNEXION AU WORKSPACE AZURE ML
# ══════════════════════════════════════════════════════════════════

def get_ml_client():
    credential = DefaultAzureCredential()
    client = MLClient(
        credential=credential,
        subscription_id=CONFIG["subscription_id"],
        resource_group_name=CONFIG["resource_group"],
        workspace_name=CONFIG["workspace_name"],
    )
    print(f"✅ Connecté à {CONFIG['workspace_name']}")
    return client


# ══════════════════════════════════════════════════════════════════
# CRÉATION DU COMPUTE (si pas déjà existant)
# ══════════════════════════════════════════════════════════════════

def ensure_compute(client: MLClient):
    try:
        compute = client.compute.get(CONFIG["compute_name"])
        print(f"✅ Compute existant : {compute.name} ({compute.size})")
    except Exception:
        print(f"⚙️  Création du compute {CONFIG['compute_name']} ...")
        compute = AmlCompute(
            name=CONFIG["compute_name"],
            size=CONFIG["compute_size"],
            min_instances=0,          # scale to 0 quand inactif → économie
            max_instances=1,
            idle_time_before_scale_down=300,  # 5 min avant scale down
            tier="Dedicated",
        )
        client.compute.begin_create_or_update(compute).result()
        print(f"✅ Compute créé : {CONFIG['compute_name']}")


# ══════════════════════════════════════════════════════════════════
# ENVIRONNEMENT DOCKER (dépendances pip)
# ══════════════════════════════════════════════════════════════════

DOCKERFILE = """
FROM mcr.microsoft.com/azureml/curated/acft-hf-nlp-gpu:latest

RUN pip install --upgrade pip && \\
    pip install \\
        transformers==4.40.0 \\
        peft==0.10.0 \\
        trl==0.8.6 \\
        accelerate==0.29.3 \\
        datasets==2.19.0 \\
        PyPDF2==3.0.1 \\
        sentencepiece==0.2.0 \\
        bitsandbytes==0.43.1
"""

def get_or_create_environment(client: MLClient):
    env_name = "mckinsey-lora-env"
    try:
        env = client.environments.get(env_name, version="1")
        print(f"✅ Environnement existant : {env_name}")
        return env
    except Exception:
        print(f"⚙️  Création de l'environnement {env_name} ...")
        # Sauvegarde le Dockerfile
        os.makedirs("./docker", exist_ok=True)
        with open("./docker/Dockerfile", "w") as f:
            f.write(DOCKERFILE)

        env = Environment(
            name=env_name,
            version="1",
            build=BuildContext(path="./docker"),
            description="Environnement pour fine-tuning LoRA McKinsey",
        )
        env = client.environments.create_or_update(env)
        print(f"✅ Environnement créé")
        return env


# ══════════════════════════════════════════════════════════════════
# SOUMISSION DU JOB
# ══════════════════════════════════════════════════════════════════

def submit_training_job(client: MLClient, env):
    """
    Soumet le script mckinsey_lora_fp16.py comme job Azure ML.
    Le dossier ./mckinsey_pdfs sera uploadé automatiquement.
    """

    job = command(
        experiment_name=CONFIG["experiment_name"],
        display_name="mckinsey-lora-fp16-mistral7b",
        description="Fine-tuning LoRA fp16 Mistral-7B sur corpus McKinsey",

        # Script à exécuter
        code="./",   # upload tout le dossier courant
        command=(
            "python mckinsey_lora_fp16.py "
            "--pdf_folder ${{inputs.pdf_folder}}"
        ),

        # Inputs
        inputs={
            "pdf_folder": Input(
                type=AssetTypes.URI_FOLDER,
                path="./mckinsey_pdfs",  # uploadé vers Azure Blob Storage
            )
        },

        # Environnement et compute
        environment=f"{env.name}:{env.version}",
        compute=CONFIG["compute_name"],

        # Variables d'environnement
        environment_variables={
            "HF_TOKEN": CONFIG["hf_token"],
            "TRANSFORMERS_CACHE": "/tmp/hf_cache",
            "HF_HOME": "/tmp/hf_home",
        },

        # Distribution (1 GPU ici, mettre distribution=MpiDistribution(2) pour 2 GPU)
        instance_count=1,
    )

    submitted = client.jobs.create_or_update(job)
    print(f"\n🚀 Job soumis !")
    print(f"   Nom        : {submitted.name}")
    print(f"   Statut     : {submitted.status}")
    print(f"   Studio URL : {submitted.studio_url}")
    print(f"\n💡 Suivi en temps réel :")
    print(f"   az ml job show -n {submitted.name} -w {CONFIG['workspace_name']}")
    print(f"   az ml job stream -n {submitted.name} -w {CONFIG['workspace_name']}")
    return submitted


# ══════════════════════════════════════════════════════════════════
# RÉCUPÉRATION DES ARTEFACTS APRÈS LE JOB
# ══════════════════════════════════════════════════════════════════

def download_artifacts(client: MLClient, job_name: str):
    """Télécharge les adaptateurs LoRA une fois le job terminé."""
    print(f"\n⬇️  Téléchargement des artefacts du job {job_name} ...")
    client.jobs.download(
        name=job_name,
        download_path="./azure_outputs",
        output_name="default",
    )
    print("✅ Artefacts téléchargés → ./azure_outputs/")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("  AZURE ML — Soumission Job LoRA fp16 McKinsey")
    print("=" * 55 + "\n")

    client = get_ml_client()
    ensure_compute(client)
    env = get_or_create_environment(client)
    job = submit_training_job(client, env)

    # Pour télécharger les résultats après la fin du job :
    # download_artifacts(client, job.name)

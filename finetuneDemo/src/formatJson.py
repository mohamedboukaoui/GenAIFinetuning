import json
import os

def format_data(input_file, output_file,limit=1000):
    # On vérifie si le fichier existe pour éviter une erreur bête
    if not os.path.exists(input_file):
        print(f"Erreur : Le fichier {input_file} est introuvable.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in data:
            # On utilise les tokens spéciaux définis précédemment
            # On concatène TOUT dans une seule clé "text" pour Unsloth
            full_text = (
                f"<|begin_prompt|>{entry['prompt']}<|end_prompt|>\n"
                f"<|begin_thought|>{entry['complex_cot']}<|end_thought|>\n"
                f"<|begin_response|>{entry['response']}<|end_response|>"
            )
            
            # Format attendu par la majorité des trainers (SFTTrainer)
            json_line = {"text": full_text}
            f.write(json.dumps(json_line) + '\n')

# Récupère le dossier où se trouve le script actuel (src)
current_dir = os.path.dirname(os.path.abspath(__file__))

# On remonte d'un cran pour atteindre la racine du projet, puis on descend dans dataset
input_path = os.path.join(current_dir, "..", "dataset", "Alpie-core_medical_psychology_dataset.json")
output_path = os.path.join(current_dir, "..", "dataset", "train_ready.jsonl")

format_data(input_path, output_path)
print(f"Fichier prêt pour le combat ! Disponible ici : {output_path}")
import fitz  # PyMuPDF
import os
import json
import re

def clean_mckinsey_text(text):
    """
    Nettoie le texte spécifique aux rapports de consulting.
    """
    # 1. Supprimer les en-têtes et pieds de page types
    text = re.sub(r"McKinsey\s*&\s*Company", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Copyright\s*©.*", "", text, flags=re.IGNORECASE)
    
    # 2. Supprimer les numéros de page isolés
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    
    # 3. Supprimer les URLs
    text = re.sub(r"https?://\S+", "", text)
    
    # 4. Normaliser les espaces et sauts de ligne
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def pdf_to_consulting_dataset(input_folder, output_file):
    dataset = []
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".pdf"):
            path = os.path.join(input_folder, filename)
            doc = fitz.open(path)
            full_text = ""
            
            for page in doc:
                full_text += page.get_text("text") + " "
            
            cleaned_text = clean_mckinsey_text(full_text)
            
            # On découpe par sections (ici on simule un bloc d'apprentissage)
            # Dans l'idéal, tu devrais découper par "Insight" ou par chapitre
            dataset.append({
                "source": filename,
                "content": cleaned_text
            })
            
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

# Utilisation
# pdf_to_consulting_dataset("./mes_rapports_pdf", "raw_data.json")
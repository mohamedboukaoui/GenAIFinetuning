from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# 1. Définition de la structure de sortie souhaitée
class ConsultingPair(BaseModel):
    instruction: str = Field(description="La consigne de rédaction")
    input_notes: str = Field(description="Les notes brutes et désordonnées")
    output_report: str = Field(description="Le rapport final style McKinsey (Pyramide de Minto)")

# 2. Configuration du modèle et du prompt
model = ChatOpenAI(model="gpt-4o", temperature=0.7)
parser = JsonOutputParser(pydantic_object=ConsultingPair)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Tu es un expert en communication chez McKinsey & Company. "
               "Ton rôle est de transformer des rapports finis en paires d'entraînement pour une IA. "
               "Tu dois générer un JSON contenant : \n"
               "1. Une instruction spécifique.\n"
               "2. Une version 'notes de terrain' brouillonne (l'INPUT).\n"
               "3. Le texte original structuré et élégant (l'OUTPUT).\n"
               "{format_instructions}"),
    ("user", "Transforme ce texte en paire d'entraînement : \n\n{original_text}")
])

# 3. Création de la chaîne (Chain)
chain = prompt | model | parser

def generate_dataset_entry(raw_text):
    try:
        response = chain.invoke({
            "original_text": raw_text,
            "format_instructions": parser.get_format_instructions()
        })
        return response
    except Exception as e:
        print(f"Erreur lors de la génération : {e}")
        return None

# Exemple d'utilisation
text_sample = "Le marché de l'électrique croît de 20%. Nos coûts sont trop hauts. Il faut investir dans les batteries LFP."
result = generate_dataset_entry(text_sample)

print(result)
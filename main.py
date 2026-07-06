import json
from models import FunctionDefinition, TestPrompt

def load_data():
    # Remplacez les noms de fichiers par le chemin exact si besoin (ex: 'data/input/...')
    with open('data/functions_definition.json', 'r') as f:
        raw_functions = json.load(f)
        
    with open('data/function_calling_tests.json', 'r') as f:
        raw_prompts = json.load(f)

    # Validation magique de Pydantic !
    # Si le JSON est mal formé, Pydantic plantera proprement avec un beau message d'erreur
    functions = [FunctionDefinition(**f) for f in raw_functions]
    prompts = [TestPrompt(**p) for p in raw_prompts]

    print(f"✅ {len(functions)} fonctions chargées et validées !")
    print(f"✅ {len(prompts)} prompts chargés et validés !")
    
    # Petit test d'affichage
    print(f"La première fonction s'appelle : {functions[0].fn_name}")

if __name__ == "__main__":
    load_data()
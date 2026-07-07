import json
from models import FunctionDefinition, TestPrompt
from llm_sdk import Small_LLM_Model

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
    # 1. Load the model
    model = Small_LLM_Model(model_name="Qwen/Qwen1.5-0.5B")

    # 2. Use a better prompt structure!
    prompt = "What is the sum of 2 and 3?"

    print("Thinking...")
    input_tensor = model._encode(prompt)
    print(input_tensor)

#    # 3. Give it a bit more room to speak (15 tokens instead of 5)
    output_tensor = model._model.generate(input_tensor, max_new_tokens=15)
#
#    # 4. Decode the AI's numerical response back into English text
    print(output_tensor)
    final_text = model._decode(output_tensor[0])
    print(final_text)
#    new_text = final_text.replace(prompt, "")
#    clean_answer = new_text.split("\n")[0].strip()
#
#    print("\n--- AI RESPONSE ---")
#    print(clean_answer)
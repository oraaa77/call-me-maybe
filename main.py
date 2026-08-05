import json
import os
import sys
from llm_sdk import Small_LLM_Model
from src.state_machine import DynamicJSONFSM

def construire_prompt_systeme(fonctions: list) -> str:
    """
    Génère dynamiquement le prompt. 
    100% UNIVERSEL : Aucune connaissance métier (maths, regex) n'est codée en dur.
    Utilise des exemples d'un domaine neutre (Smart Home) pour enseigner la structure.
    """
    prompt = "You are an expert AI translating natural language to strict JSON function calls.\n\n"
    
    # --- 1. RÈGLES GÉNÉRALES (Agnostiques) ---
    prompt += "RULES:\n"
    prompt += "1. Analyze the user's core intent.\n"
    prompt += "2. Match the intent to the closest available function.\n"
    prompt += "3. Extract arguments strictly matching the required types (e.g., convert words to numbers if a float is expected).\n\n"
    
    prompt += "Available functions:\n"
    
    # --- 2. EXTRACTION DYNAMIQUE DES FONCTIONS ---
    for fn in fonctions:
        # On crée un indice naturel (ex: fn_add_numbers -> "add numbers")
        nom_naturel = fn["fn_name"].replace("fn_", "").replace("_", " ")
        
        args_list = []
        for arg_name in fn["args_names"]:
            arg_type = fn["args_types"][arg_name]
            args_list.append(f"{arg_name}: {arg_type}")
        
        args_str = ", ".join(args_list)
        prompt += f"- {fn['fn_name']}({args_str}) -> {fn['return_type']} (Hint: Use this to {nom_naturel})\n"
        
    prompt += "\nOutput a strict JSON object with 'function_name' and 'arguments'.\n"
    
    # --- 3. EXEMPLES NEUTRES (Domaine : Smart Home / Domotique) ---
    # Ces exemples apprennent la syntaxe et la logique d'extraction au modèle,
    # sans jamais polluer son contexte avec des maths ou du Regex.
    prompt += "\nExamples of structural mapping:\n"
    
    prompt += '\nUser input: "Turn on the living room lights"'
    prompt += '\nOutput:\n{"function_name": "fn_toggle_device", "arguments": {"device": "lights", "room": "living room", "state": "on"}}\n'
    
    prompt += '\nUser input: "Set the oven temperature to four hundred degrees"'
    prompt += '\nOutput:\n{"function_name": "fn_set_temperature", "arguments": {"target": "oven", "temp": 400.0}}\n'
    
    prompt += '\nUser input: "Replace the word apple with banana in the shopping list"'
    prompt += '\nOutput:\n{"function_name": "fn_modify_list", "arguments": {"old_item": "apple", "new_item": "banana"}}\n'
    
    return prompt

def generer_appel_fonction(model, fsm, id_to_text, vocab_size, system_prompt, question, char_to_tokens):
    """
    Exécute la boucle de Constrained Decoding pour UNE seule question.
    Optimisé pour la vitesse maximale avec Cache FSM et Indexation inversée.
    """
    prompt_complet = f"{system_prompt}\n\nUser input: \"{question}\"\nOutput:\n"
    input_ids = model._encode(prompt_complet).tolist()[0]
    prompt_length = len(input_ids)
    
    max_tokens = 150 
    fsm_cache = {}
    
    def get_allowed_cached(text):
        if text not in fsm_cache:
            fsm_cache[text] = fsm.get_allowed_characters(text)
        return fsm_cache[text]
    
    for etape in range(max_tokens):
        generated_ids = input_ids[prompt_length:]
        current_generated_text = model._decode(generated_ids)
        
        allowed_chars = get_allowed_cached(current_generated_text)
        
        if not allowed_chars:
            break
            
        logits = model.get_logits_from_input_ids(input_ids)
        
        if hasattr(logits, "tolist"):
            logits = logits.tolist()
            
        # Sécurité : Extraire le dernier jeton si logits est un tableau 2D/3D
        while len(logits) > 0 and isinstance(logits[0], list):
            logits = logits[-1]
            
        # OPTIMISATION MAJEURE : On ne récupère QUE les jetons qui commencent par une lettre autorisée
        candidate_tokens = []
        for char in allowed_chars:
            if char in char_to_tokens:
                for tid in char_to_tokens[char]:
                    candidate_tokens.append((tid, logits[tid]))
                    
        # On ne trie qu'une infime fraction du dictionnaire, c'est instantané !
        sorted_candidates = sorted(candidate_tokens, key=lambda x: x[1], reverse=True)
        
        best_token_id = None
        
        for token_id, logit_value in sorted_candidates:
            token_str = id_to_text[token_id]
            is_valid = True
            temp_text = current_generated_text + token_str[0]
            
            for i in range(1, len(token_str)):
                next_allowed = get_allowed_cached(temp_text)
                if token_str[i] not in next_allowed:
                    is_valid = False
                    break
                temp_text += token_str[i]
                
            if is_valid:
                best_token_id = token_id
                break 
                
        if best_token_id is None:
            break
            
        input_ids.append(best_token_id)
        print(id_to_text[best_token_id], end="", flush=True)
        
    return model._decode(input_ids[prompt_length:])

def main():
    print("="*50)
    print("INITIALISATION DU MOTEUR LLM")
    print("="*50)
    
    # --- GESTION ROBUSTE DES ERREURS (Critère 4) ---
    schema_path = "data/functions_definition.json"
    tests_path = "data/function_calling_tests.json"
    
    if not os.path.exists(schema_path):
        print(f"❌ Erreur critique : Le fichier {schema_path} est introuvable.")
        sys.exit(1)
        
    if not os.path.exists(tests_path):
        print(f"❌ Erreur critique : Le fichier {tests_path} est introuvable.")
        sys.exit(1)

    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        with open(tests_path, "r", encoding="utf-8") as f:
            questions_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Erreur critique : Fichier JSON malformé - {str(e)}")
        sys.exit(1)

    print("✅ Fichiers chargés avec succès.")
    
    try:
        print("Chargement du modèle d'IA (Qwen1.5-0.5B)...")
        model = Small_LLM_Model(model_name="Qwen/Qwen1.5-0.5B")
        fsm = DynamicJSONFSM(schema)
        system_prompt = construire_prompt_systeme(schema)

        print("Préparation du dictionnaire de tokens...")
        vocab_size = len(model._tokenizer)
        
        id_to_text = [model._tokenizer.decode([i], skip_special_tokens=True) for i in range(vocab_size)]
        
        # --- OPTIMISATION : Indexation rapide au démarrage ---
        print("Création de l'index rapide des jetons...")
        char_to_tokens = {}
        for token_id, token_str in enumerate(id_to_text):
            if token_str:
                first_char = token_str[0]
                if first_char not in char_to_tokens:
                    char_to_tokens[first_char] = []
                char_to_tokens[first_char].append(token_id)
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle ou de la FSM : {str(e)}")
        sys.exit(1)

    print("\n" + "="*50)
    print("DÉBUT DU TRAITEMENT PAR LOTS")
    print("="*50)
    
    resultats = []
    
    # --- BOUCLE PRINCIPALE AVEC GESTION DES TIMEOUTS / ERREURS ---
    for item in questions_data:
        # Gérer le cas où le prompt est manquant ou vide dans le JSON de test
        question = item.get("prompt", "")
        if not question:
            continue

        print(f"\n[Question] : {question}")
        print("[Génération] : ", end="")
        
        try:
            # Ajout de l'argument char_to_tokens
            json_resultat = generer_appel_fonction(model, fsm, id_to_text, vocab_size, system_prompt, question, char_to_tokens)
            parsed_result = json.loads(json_resultat)
            print("\n[Statut] : ✅ Terminé et Valide")
        except json.JSONDecodeError:
            print("\n[Statut] : ❌ Erreur de Parsing JSON")
            parsed_result = {"error": "Invalid JSON generated", "raw_output": json_resultat}
        except Exception as e:
            print(f"\n[Statut] : ❌ Erreur inattendue : {str(e)}")
            parsed_result = {"error": f"Exception: {str(e)}"}
        
        resultats.append({
            "prompt": question,
            "function_call": parsed_result
        })

    print("\n" + "="*50)
    print("SAUVEGARDE DES RÉSULTATS")
    print("="*50)
    
    try:
        with open("output.json", "w", encoding="utf-8") as f:
            json.dump(resultats, f, indent=2, ensure_ascii=False)
        print("✅ Tout est terminé avec succès ! Fichier 'output.json' généré. 🚀")
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde : {str(e)}")

if __name__ == "__main__":
    main()
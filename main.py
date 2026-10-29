import json
import os
import re
import sys

from llm_sdk.llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition
from srcc.state_machine import DynamicJSONFSM

MODEL_NAME = "Qwen/Qwen3-0.6B"


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
    prompt += "4. Copy string arguments exactly from the user input.\n"
    prompt += "5. Do not paraphrase, normalize, expand, or rewrite string arguments.\n"
    prompt += "6. Preserve apostrophes, capitalization, spaces, and punctuation exactly.\n\n"
    prompt += """7. For every argument whose type is "number", always output a decimal notation.
            Even whole numbers must include ".0".
            Examples:
            - 3 must be written as 3.0
            - 5 must be written as 5.0
            - 12.5 remains 12.5
            Do not output integer notation for values of type "number"."""

    prompt += "Available functions:\n"

    # --- 2. EXTRACTION DYNAMIQUE DES FONCTIONS ---
    for fn in fonctions:
        # On crée un indice naturel (ex: fn_add_numbers -> "add numbers")
        nom_naturel = fn["name"].replace("fn_", "").replace("_", " ")

        args_list = []
        for arg_name, arg_definition in fn["parameters"].items():
            arg_type = arg_definition["type"]
            args_list.append(f"{arg_name}: {arg_type}")

        args_str = ", ".join(args_list)
        return_type = fn["returns"]["type"]
        prompt += f"- {fn['name']}({args_str}) -> {return_type} (Hint: Use this to {nom_naturel})\n"

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


def normaliser_parametres(function_name: str, arguments: dict, fonctions: list) -> dict:
    """Convert numeric arguments to the float representation used by corrections."""
    function_definition = next(
        (fn for fn in fonctions if fn["name"] == function_name),
        None,
    )
    if function_definition is None:
        return arguments

    parameters = function_definition["parameters"]
    normalised = dict(arguments)
    for arg_name, arg_definition in parameters.items():
        if (
            arg_definition["type"] == "number"
            and arg_name in normalised
            and not isinstance(normalised[arg_name], bool)
        ):
            normalised[arg_name] = float(normalised[arg_name])
    return normalised


def preserver_arguments_litteraux(
    function_name: str,
    arguments: dict,
    question: str,
    fonctions: list,
) -> dict:
    """Preserve string values that can be extracted unambiguously."""
    function_definition = next(
        (fn for fn in fonctions if fn["name"] == function_name),
        None,
    )
    if function_definition is None:
        return arguments

    string_parameters = [
        name
        for name, definition in function_definition["parameters"].items()
        if definition["type"] == "string"
    ]
    if not string_parameters:
        return arguments

    preserved = dict(arguments)
    path_match = re.search(r"""(?<!\w)(?:[A-Za-z]:\\[^\s]+|/[^\s]+)""", question)
    if path_match:
        path_parameter = next(
            (
                name
                for name in string_parameters
                if any(term in name.lower() for term in ("path", "file", "location"))
            ),
            None,
        )
        if path_parameter is not None:
            preserved[path_parameter] = path_match.group(0).rstrip(".,!?")

    if len(string_parameters) == 1 and ":" in question:
        _, value = question.split(":", 1)
        if value.strip():
            preserved[string_parameters[0]] = value.lstrip()
            return preserved

    quoted_strings = [
        match.group(2)
        for match in re.finditer(r"""(?<!\w)(["'])(.*?)\1""", question)
    ]
    if not quoted_strings:
        return preserved

    assigned_parameters = set()
    source_parameter = next(
        (
            name
            for name in string_parameters
            if any(term in name.lower() for term in ("source", "text", "content", "input"))
        ),
        None,
    )
    if source_parameter is not None and len(quoted_strings) > 1:
        source_value = max(quoted_strings, key=len)
        preserved[source_parameter] = source_value
        assigned_parameters.add(source_parameter)
        quoted_strings = [
            value for value in quoted_strings if value != source_value
        ]

    remaining_parameters = [
        name for name in string_parameters if name not in assigned_parameters
    ]
    for parameter_name, value in zip(remaining_parameters, quoted_strings):
        preserved[parameter_name] = value
    return preserved


def generer_appel_fonction(model, fsm, id_to_text, vocab_size, system_prompt, question, char_to_tokens):
    """
    Exécute la boucle de Constrained Decoding pour UNE seule question.
    Optimisé pour la vitesse maximale avec Cache FSM et Indexation inversée.
    """
    prompt_complet = f"{system_prompt}\n\nUser input: \"{question}\"\nOutput:\n"
    input_ids = model.encode(prompt_complet).tolist()[0]
    prompt_length = len(input_ids)

    max_tokens = 150
    fsm_cache = {}

    def get_allowed_cached(text):
        if text not in fsm_cache:
            fsm_cache[text] = fsm.get_allowed_characters(text, question)
        return fsm_cache[text]

    for _ in range(max_tokens):
        generated_ids = input_ids[prompt_length:]
        current_generated_text = model.decode(generated_ids)

        allowed_chars = get_allowed_cached(current_generated_text)

        if not allowed_chars:
            break

        logits = model.get_logits_from_input_ids(input_ids)

        if hasattr(logits, "tolist"):
            logits = logits.tolist()

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

    return model.decode(input_ids[prompt_length:])

def main():
    print("="*50)
    print("INITIALISATION DU MOTEUR LLM")
    print("="*50)

    # --- GESTION ROBUSTE DES ERREURS (Critère 4) ---
    schema_path = "data/input/functions_definition.json"
    tests_path = "data/input/function_calling_tests.json"

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
        print(f"Chargement du modèle d'IA ({MODEL_NAME})...")
        model = Small_LLM_Model(model_name=MODEL_NAME)
        function_definitions = [
            FunctionDefinition(
                fn_name=function["name"],
                args_names=list(function["parameters"]),
                args_types={
                    name: definition["type"]
                    for name, definition in function["parameters"].items()
                },
                return_type=function["returns"]["type"],
            )
            for function in schema
        ]
        fsm = DynamicJSONFSM(function_definitions)
        system_prompt = construire_prompt_systeme(schema)

        print("Préparation du dictionnaire de tokens...")
        sample_input_ids = model.encode(system_prompt).tolist()[0]
        vocab_size = len(model.get_logits_from_input_ids(sample_input_ids))

        id_to_text = [
            model.decode([i])
            for i in range(vocab_size)
        ]

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
        parsed_result["prompt"] = question
        resultats.append(parsed_result)
        #if (
        #    isinstance(parsed_result, dict)
        #    and isinstance(parsed_result.get("function_name"), str)
        #    and isinstance(parsed_result.get("arguments"), dict)
        #):
        #    resultats.append({
        #        "prompt": question,
        #        "name": parsed_result["function_name"],
        #        "parameters": preserver_arguments_litteraux(
        #            parsed_result["function_name"],
        #            normaliser_parametres(
        #                parsed_result["function_name"],
        #                parsed_result["arguments"],
        #                schema,
        #            ),
        #            question,
        #            schema,
        #        ),
        #    })
        #else:
        #    # Keep generation errors explicit while using the same top-level
        #    # structure as successful results.
        #    resultats.append({
        #        "prompt": question,
        #        "error": parsed_result,
        #    })

    print("\n" + "="*50)
    print("SAUVEGARDE DES RÉSULTATS")
    print("="*50)

    try:
        with open("data/output/function_calling_results.json", "w", encoding="utf-8") as f:
            json.dump(resultats, f, indent=2, ensure_ascii=False)
        print("✅ Tout est terminé avec succès ! Fichier 'data/output/function_calling_results.json' généré. 🚀")
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde : {str(e)}")


if __name__ == "__main__":
    main()

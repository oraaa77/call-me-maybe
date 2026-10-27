from llm_sdk.llm_sdk import Small_LLM_Model
from srcc.parser import Parser
from srcc.state_machine import DynamicJSONFSM
from typing import Dict, List


class Call_Me_Maybe(Small_LLM_Model):
    def __init__(self, data: Parser, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        self.functions = data.functions
        self.tests = data.tests
        self.fsm = DynamicJSONFSM()
        #        # Optimization caches
        self.char_to_tokens: Dict[str, List[int]] = {}
        self.id_to_text: List[str] = []
        self._init_vocabulary()

    @property
    def system_prompt(self) -> str:
        """
        Génèration dynamique du prompt.
        """
        prompt = ("You are an expert AI translating natural "
                  + "language to strict JSON function calls.\n\n")

        # --- 1. RÈGLES GÉNÉRALES (Agnostiques) ---
        prompt += "RULES:\n"
        prompt += "1. Analyze the user's core intent.\n"
        prompt += "2. Match the intent to the closest available function.\n"
        prompt += ("3. Extract arguments strictly matching the required "
                   + "types (e.g., convert words to numbers "
                   + "if a float is expected).\n\n")
        prompt += ("4. Copy string arguments exactly from the user input.\n")
        prompt += ("5. Do not paraphrase, normalize, "
                   + "expand, or rewrite string arguments.\n")
        prompt += ("6. Preserve apostrophes, capitalization, "
                   + "spaces, and punctuation exactly.\n\n")

        prompt += "Available functions:\n"

        # --- 2. EXTRACTION DYNAMIQUE DES FONCTIONS ---
        for fn in self.functions:
            nom_naturel = fn["name"].replace("fn_", "").replace("_", " ")

            args_list = []
            for arg_name, arg_definition in fn["parameters"].items():
                arg_type = arg_definition["type"]
                args_list.append(f"{arg_name}: {arg_type}")

            args_str = ", ".join(args_list)
            return_type = fn["returns"]["type"]
            prompt += (f"- {fn['name']}({args_str}) -> {return_type} "
                       f"(Hint: Use this to {nom_naturel})\n")

        prompt += ("\nOutput a strict JSON object with"
                   + " 'function_name' and 'arguments'.\n")

        # --- 3. EXEMPLES ---

        prompt += "\nExamples of structural mapping:\n"

        prompt += '\nUser input: "Turn on the living room lights"'
        prompt += ('\nOutput:\n{"function_name": "fn_toggle_device", '
                   + '"arguments": {"device": "lights", "room": '
                   + '"living room", "state": "on"}}\n')

        prompt += ('\nUser input: "Set the oven temperature '
                   + 'to four hundred degrees"')
        prompt += ('\nOutput:\n{"function_name": "fn_set_temperature", '
                   + '"arguments": {"target": "oven", "temp": 400.0}}\n')

        prompt += ('\nUser input: "Replace the word apple '
                   + 'with banana in the shopping list"')
        prompt += ('\nOutput:\n{"function_name": "fn_modify_list", '
                   + '"arguments": {"old_item": "apple", '
                   + '"new_item": "banana"}}\n')

        return prompt

#class Call_Me_Maybe(Small_LLM_Model):
#    def __init__(self, data: Parser, model_name: str = "Qwen/Qwen3-0.6B") -> None:
#        super().__init__(model_name=model_name)
#        self.data_parser = data
#        self.functions = data.functions
#        self.tests = data.tests
#        self.raw_schema = data.raw_schema
#        self.fsm = DynamicJSONFSM(self.raw_schema)

#        # Optimization caches
#        self.char_to_tokens: Dict[str, List[int]] = {}
#        self.id_to_text: List[str] = []
#        self._init_vocabulary()

    def _init_vocabulary(self):
        """Initialise le vocabulaire et l'indexation rapide."""
        print("Préparation du dictionnaire de tokens...")
        sample_input_ids = self.encode(self.system_prompt).tolist()[0]
        vocab_size = len(self.get_logits_from_input_ids(sample_input_ids))

        self.id_to_text = [self.decode([i]) for i in range(vocab_size)]

        print("Création de l'index rapide des jetons...")
        for token_id, token_str in enumerate(self.id_to_text):
            if token_str:
                first_char = token_str[0]
                if first_char not in self.char_to_tokens:
                    self.char_to_tokens[first_char] = []
                self.char_to_tokens[first_char].append(token_id)

    def normalize_parameters(self, function_name: str, arguments: dict) -> dict:
        function_definition = next((fn for fn in self.functions if fn.fn_name == function_name), None)

        if not function_definition:
            return arguments

        normalised = dict(arguments)

        for arg_name, arg_type in function_definition.args_types.items():
            if (arg_type == "number" # ou "float" selon comment vous l'avez défini
                and arg_name in normalised
                and not isinstance(normalised[arg_name], bool)):
                normalised[arg_name] = float(normalised[arg_name])

        return normalised

    def preserve_literal_arguments(self, function_name: str, arguments: dict, question: str) -> dict:
        # 1. On cherche dans self.functions avec l'attribut .fn_name
        function_definition = next((fn for fn in self.functions if fn.fn_name == function_name), None)
        if not function_definition:
            return arguments

        # 2. On utilise l'attribut .args_types de l'objet Pydantic
        string_parameters = [
            arg_name for arg_name, arg_type in function_definition.args_types.items()
            if arg_type == "string"
        ]

        if not string_parameters:
            return arguments

        preserved = dict(arguments)

        # --- Le reste de la logique Regex ne change pas ---

        # Path Extraction
        path_match = re.search(r"""(?<!\w)(?:[A-Za-z]:\\[^\s]+|/[^\s]+)""", question)
        if path_match:
            path_parameter = next((name for name in string_parameters if any(term in name.lower() for term in ("path", "file", "location"))), None)
            if path_parameter:
                preserved[path_parameter] = path_match.group(0).rstrip(".,!?")

        # Single string value colon separation
        if len(string_parameters) == 1 and ":" in question:
            _, value = question.split(":", 1)
            if value.strip():
                preserved[string_parameters[0]] = value.lstrip()
                return preserved

        # Quoted strings extraction
        quoted_strings = [match.group(2) for match in re.finditer(r"""(?<!\w)(["'])(.*?)\1""", question)]
        if not quoted_strings:
            return preserved

        assigned_parameters = set()
        source_parameter = next((name for name in string_parameters if any(term in name.lower() for term in ("source", "text", "content", "input"))), None)

        if source_parameter and len(quoted_strings) > 1:
            source_value = max(quoted_strings, key=len)
            preserved[source_parameter] = source_value
            assigned_parameters.add(source_parameter)
            quoted_strings = [val for val in quoted_strings if val != source_value]

        remaining_parameters = [name for name in string_parameters if name not in assigned_parameters]
        for parameter_name, value in zip(remaining_parameters, quoted_strings):
            preserved[parameter_name] = value

        return preserved


    def generate_function_call(self, question: str) -> str:
        prompt_complet = f"{self.system_prompt}\n\nUser input: \"{question}\"\nOutput:\n"
        input_ids = self.encode(prompt_complet).tolist()[0]
        prompt_length = len(input_ids)
        max_tokens = 150
        fsm_cache = {}

        def get_allowed_cached(text):
            if text not in fsm_cache:
                fsm_cache[text] = self.fsm.get_allowed_characters(text)
            return fsm_cache[text]

        for _ in range(max_tokens):
            generated_ids = input_ids[prompt_length:]
            current_generated_text = self.decode(generated_ids)
            allowed_chars = get_allowed_cached(current_generated_text)

            if not allowed_chars:
                break

            logits = self.get_logits_from_input_ids(input_ids)
            if hasattr(logits, "tolist"):
                logits = logits.tolist()
            while len(logits) > 0 and isinstance(logits[0], list):
                logits = logits[-1]

            candidate_tokens = []
            for char in allowed_chars:
                if char in self.char_to_tokens:
                    for tid in self.char_to_tokens[char]:
                        candidate_tokens.append((tid, logits[tid]))

            sorted_candidates = sorted(candidate_tokens, key=lambda x: x[1], reverse=True)
            best_token_id = None

            for token_id, _ in sorted_candidates:
                token_str = self.id_to_text[token_id]
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
            print(self.id_to_text[best_token_id], end="", flush=True)

        return self.decode(input_ids[prompt_length:])

    def process_batch(self, output_file: str = "data/output/function_calling_results.json"):
        print("\n" + "="*50)
        print("DÉBUT DU TRAITEMENT PAR LOTS")
        print("="*50)

        resultats = []

        for test in self.tests:
            question = test.prompt
            if not question:
                continue

            print(f"\n[Question] : {question}")
            print("[Génération] : ", end="")

            try:
                json_resultat = self.generate_function_call(question)
                parsed_result = json.loads(json_resultat)
                print("\n[Statut] : ✅ Terminé et Valide")

                if (isinstance(parsed_result, dict)
                    and isinstance(parsed_result.get("function_name"), str)
                    and isinstance(parsed_result.get("arguments"), dict)):

                    normalized_args = self.normalize_parameters(parsed_result["function_name"], parsed_result["arguments"])
                    final_args = self.preserve_literal_arguments(parsed_result["function_name"], normalized_args, question)

                    resultats.append({
                        "prompt": question,
                        "name": parsed_result["function_name"],
                        "parameters": final_args
                    })
                else:
                    resultats.append({"prompt": question, "error": parsed_result})

            except json.JSONDecodeError:
                print("\n[Statut] : ❌ Erreur de Parsing JSON")
                resultats.append({"prompt": question, "error": {"error": "Invalid JSON generated", "raw_output": json_resultat}})
            except Exception as e:
                print(f"\n[Statut] : ❌ Erreur inattendue : {str(e)}")
                resultats.append({"prompt": question, "error": {"error": f"Exception: {str(e)}"}})

        print("\n" + "="*50)
        print("SAUVEGARDE DES RÉSULTATS")
        print("="*50)

        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(resultats, f, indent=2, ensure_ascii=False)
            print(f"✅ Tout est terminé avec succès ! Fichier '{output_file}' généré. 🚀")
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde : {str(e)}")


if __name__ == "__main__":
    parse = Parser()
    call_me_maybe = Call_Me_Maybe(parse)
    print(call_me_maybe.prompt)

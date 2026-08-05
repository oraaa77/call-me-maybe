from llm_sdk_copy import Small_LLM_Model
from srcc.parser import Parser
from typing import List, Dict, Any

class Call_Me_Maybe(Small_LLM_Model):
    def __init__(self, data: Parser) -> None:
        self.functions = data.functions
        self.tests = data.tests

    @property
    def prompt(self) -> str:
        prompt = "You are an expert AI translating natural language to strict JSON function calls.\n\n"

        # --- 1. RÈGLES GÉNÉRALES (Agnostiques) ---
        prompt += "RULES:\n"
        prompt += "1. Analyze the user's core intent.\n"
        prompt += "2. Match the intent to the closest available function.\n"
        prompt += "3. Extract arguments strictly matching the required types (e.g., convert words to numbers if a float is expected).\n\n"

        prompt += "Available functions:\n"

        # --- 2. EXTRACTION DYNAMIQUE DES FONCTIONS ---
        for fn in self.functions:
            # On crée un indice naturel (ex: fn_add_numbers -> "add numbers")

            args_list = []
            for arg in fn.parameters:
                args_list.append(f"{arg.name}: {arg.type}")

            args_str = ", ".join(args_list)
            prompt += f"- {fn.name}({args_str}) -> {fn.returns} (Hint: Use this to {fn.description})\n"

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

if __name__ == "__main__":
    parse = Parser()
    call_me_maybe = Call_Me_Maybe(parse)
    print(call_me_maybe.prompt)
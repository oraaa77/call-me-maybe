"""Reusable constrained-decoding LLM engine.

The model is loaded only when :class:`LLMModel` is instantiated.  This makes
the module safe to import from another application or from a test.
"""

import json
from pathlib import Path
from typing import Any

MODEL_NAME = "Qwen/Qwen3-0.6B"


class LLMModel:
    """Engine that converts natural-language questions into JSON function calls."""

    def __init__(
        self,
        schema_path: str | Path = "data/input/functions_definition.json",
        *,
        model_name: str = MODEL_NAME,
    ) -> None:
        from llm_sdk.llm_sdk import Small_LLM_Model
        from srcc.models import Argument, FunctionDefinition
        from srcc.state_machine import DynamicJSONFSM
        from main import construire_prompt_systeme

        self.schema_path = Path(schema_path)
        self.schema = self._load_schema()
        self.model = Small_LLM_Model(model_name=model_name)
        self.function_definitions = [
            FunctionDefinition(
                name=function["name"],
                description=function.get("description", ""),
                parameters=[
                    Argument(name=name, type=definition["type"])
                    for name, definition in function["parameters"].items()
                ],
                returns=function["returns"]["type"],
            )
            for function in self.schema
        ]
        self.fsm = DynamicJSONFSM(self.function_definitions)
        self.system_prompt = construire_prompt_systeme(self.schema)
        self.id_to_text, self.char_to_tokens = self._build_token_index()

    def _load_schema(self) -> list[dict[str, Any]]:
        """Load and validate the function schema used by the prompt and FSM."""
        try:
            with self.schema_path.open(encoding="utf-8") as schema_file:
                schema = json.load(schema_file)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Function schema not found: {self.schema_path}"
            ) from error
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON in function schema: {self.schema_path}"
            ) from error

        if not isinstance(schema, list):
            raise ValueError("Function schema must contain a JSON list.")
        return schema

    def _build_token_index(self) -> tuple[list[str], dict[str, list[int]]]:
        """Index vocabulary tokens by their first decoded character."""
        sample_input_ids = self.model.encode(self.system_prompt).tolist()[0]
        vocab_size = len(self.model.get_logits_from_input_ids(sample_input_ids))
        id_to_text = [self.model.decode([token_id]) for token_id in range(vocab_size)]

        char_to_tokens: dict[str, list[int]] = {}
        for token_id, token_text in enumerate(id_to_text):
            if token_text:
                char_to_tokens.setdefault(token_text[0], []).append(token_id)
        return id_to_text, char_to_tokens

    def generate(self, question: str) -> dict[str, Any]:
        """Generate and decode one constrained JSON function call."""
        if not question:
            raise ValueError("Question must not be empty.")

        from main import generer_appel_fonction

        generated = generer_appel_fonction(
            self.model,
            self.fsm,
            self.id_to_text,
            len(self.id_to_text),
            self.system_prompt,
            question,
            self.char_to_tokens,
        )
        try:
            result = json.loads(generated)
        except json.JSONDecodeError as error:
            raise ValueError("The model generated invalid JSON.") from error

        if not isinstance(result, dict):
            raise ValueError("The model output must be a JSON object.")
        return result

    __call__ = generate


llm_model = LLMModel


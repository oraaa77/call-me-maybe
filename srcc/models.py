from pydantic import BaseModel
from typing import Dict, List

# 1. Modèle pour "functions_definition.json"

class Argument(BaseModel):
    name: str
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: List[Argument]
    returns: str


# 2. Modèle pour "function_calling_tests.json"
class TestPrompt(BaseModel):
    prompt: str


# 3. Modèle pour la sortie attendue (Section 5.4 des consignes)
# Le dictionnaire parameters contiendra les arguments (ex: {"a": 2, "b": 3})
class FunctionCallOutput(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, float | str | int | bool]

from pydantic import BaseModel
from typing import List, Dict

# 1. Modèle pour "functions_definition.json"
class FunctionDefinition(BaseModel):
    fn_name: str
    args_names: List[str]
    args_types: Dict[str, str]
    return_type: str

# 2. Modèle pour "function_calling_tests.json"
class TestPrompt(BaseModel):
    prompt: str

# 3. Modèle pour la sortie attendue (Section 5.4 des consignes)
# Le dictionnaire parameters contiendra les arguments (ex: {"a": 2, "b": 3})
class FunctionCallOutput(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, float | str | int | bool]
from pydantic import BaseModel
from typing import List, Dict


class FunctionDefinition(BaseModel):
    fn_name: str
    args_names: List[str]
    args_types: Dict[str, str]
    return_type: str


class TestPrompt(BaseModel):
    prompt: str


class FunctionCallOutput(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, float | str | int | bool]

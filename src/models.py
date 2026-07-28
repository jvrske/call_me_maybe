from pydantic import BaseModel, ConfigDict
from typing import Literal

ParamType = Literal["number", "string", "boolean"]


class ParameterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: ParamType


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str
    description: str
    parameters: dict[str, ParameterSpec]
    returns: ParameterSpec


class PromptInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    prompt: str


class FunctionCall(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str
    parameters: dict[str, int | float | str | bool]

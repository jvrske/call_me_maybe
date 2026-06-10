from pydantic import BaseModel, ConfigDict
from typing import Literal

ParamType = Literal["number", "string", "boolean"]


class ParameterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: ParamType

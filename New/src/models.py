"""Pydantic models for function-calling definitions."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ParameterType = Literal["string", "number", "boolean", "integer"]


class ParameterDefinition(BaseModel):
    """Define the type of a function parameter."""

    model_config = ConfigDict(extra="forbid")

    type: ParameterType


class ReturnDefinition(BaseModel):
    """Define the type of a function return value."""

    model_config = ConfigDict(extra="forbid")

    type: ParameterType


class FunctionDefinition(BaseModel):
    """Define a function's name, description, parameters, and return type."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ReturnDefinition


class PromptInput(BaseModel):
    """Define a natural-language prompt provided as input."""

    model_config = ConfigDict(extra="forbid")

    prompt: str

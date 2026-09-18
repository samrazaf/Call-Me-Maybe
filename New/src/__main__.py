"""Command-line interface for the function-calling project."""

import argparse
import json
import pathlib
import sys
from typing import Protocol, Sequence

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from pydantic import ValidationError

from src.models import FunctionDefinition, PromptInput
from src.vocab_parser import (
    DecoderCache,
    LLMModel,
    TokenTensor,
    generate_constrained_json,
    validate_function_call,
)

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_FUNCTIONS = "data/input/functions_definition.json"
DEFAULT_INPUT = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT = "data/output/function_calling_results.json"


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="CallMeMaybe",
        description="Translate natural-language prompts into function calls.",
    )
    parser.add_argument(
        "--functions_definition",
        default=DEFAULT_FUNCTIONS,
        help="Path to the function definitions JSON file.",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Path to the prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Path to the output JSON file.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model identifier used by the provided llm_sdk.",
    )
    return parser.parse_args()


def load_json_file(path: pathlib.Path) -> object:
    """Load JSON data from a file."""
    if not path.is_file():
        raise ValueError(f"File not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error.msg}") from error
    except OSError as error:
        raise ValueError(f"Cannot read {path}: {error}") from error


def load_functions(path: pathlib.Path) -> list[FunctionDefinition]:
    """Load and validate function definitions."""
    data = load_json_file(path)
    if not isinstance(data, list):
        raise ValueError("Function definitions must be a JSON array.")
    try:
        functions = [FunctionDefinition.model_validate(item) for item in data]
    except ValidationError as error:
        raise ValueError(f"Invalid function definition: {error}") from error
    if not functions:
        raise ValueError("At least one function definition is required.")
    names = [function.name for function in functions]
    if len(names) != len(set(names)):
        raise ValueError("Function names must be unique.")
    return functions


def load_prompts(path: pathlib.Path) -> list[PromptInput]:
    """Load and validate natural-language prompts."""
    data = load_json_file(path)
    if not isinstance(data, list):
        raise ValueError("Input prompts must be a JSON array.")
    try:
        return [PromptInput.model_validate(item) for item in data]
    except ValidationError as error:
        raise ValueError(f"Invalid prompt entry: {error}") from error


class ModelWithVocabulary(LLMModel, Protocol):
    """Describe the additional public SDK method needed by the CLI."""

    def get_path_to_vocab_file(self) -> str:
        """Return the path to the model vocabulary file."""


def _token_ids(encoded: TokenTensor) -> list[int]:
    """Extract one-dimensional token ids from an encoded tensor."""
    values = encoded.tolist()
    if len(values) != 1:
        raise ValueError("The SDK returned an unexpected token tensor shape.")
    return values[0]


def load_vocabulary(model: ModelWithVocabulary) -> dict[int, str]:
    """Load the public vocabulary exposed by the SDK model."""
    vocabulary_path = pathlib.Path(model.get_path_to_vocab_file())
    raw_vocabulary = load_json_file(vocabulary_path)
    if not isinstance(raw_vocabulary, dict):
        raise ValueError("The model vocabulary must be a JSON object.")

    vocabulary: dict[int, str] = {}
    for token, token_id in raw_vocabulary.items():
        if not isinstance(token, str) or not isinstance(token_id, int):
            raise ValueError("The model vocabulary contains an invalid entry.")
        vocabulary[token_id] = token.replace("Ġ", " ")
    return vocabulary


def build_vocabulary(
    model: ModelWithVocabulary,
) -> tuple[dict[int, str], int]:
    """Build the vocabulary and return the model's exact logits size."""
    vocabulary = load_vocabulary(model)
    dummy_ids = _token_ids(model.encode("dummy"))
    vocabulary_size = len(model.get_logits_from_input_ids(dummy_ids))
    filtered = {
        token_id: token
        for token_id, token in vocabulary.items()
        if 0 <= token_id < vocabulary_size and token
    }
    return filtered, vocabulary_size


def run(
    functions: Sequence[FunctionDefinition],
    prompts: Sequence[PromptInput],
    model_name: str,
) -> list[dict[str, object]]:
    """Generate and validate one function call for every prompt."""
    model = Small_LLM_Model(model_name=model_name)
    vocabulary, vocabulary_size = build_vocabulary(model)
    cache = DecoderCache(
        model=model,
        vocabulary=vocabulary,
        functions=functions,
        vocabulary_size=vocabulary_size,
    )

    results: list[dict[str, object]] = []
    for item in prompts:
        print(f"\nPrompt: {item.prompt}")
        try:
            raw_output = generate_constrained_json(item.prompt, cache)
            decoded = json.loads(raw_output)
            result = validate_function_call(decoded, item.prompt, functions)
        except (RuntimeError, json.JSONDecodeError, ValueError) as error:
            print(f"Error: {error}", file=sys.stderr)
            result = {
                "prompt": item.prompt,
                "name": "",
                "parameters": {},
            }
        results.append(result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return results


def write_results(
    path: pathlib.Path, results: list[dict[str, object]]
) -> None:
    """Write validated function calls to the output JSON file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(results, file, indent=4, ensure_ascii=False)
            file.write("\n")
    except OSError as error:
        raise ValueError(f"Cannot write {path}: {error}") from error


def main() -> int:
    """Run the command-line application."""
    args = parse_arguments()
    try:
        functions = load_functions(pathlib.Path(args.functions_definition))
        prompts = load_prompts(pathlib.Path(args.input))
        results = run(functions, prompts, args.model)
        write_results(pathlib.Path(args.output), results)
    except (ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Error: SDK or model failure: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

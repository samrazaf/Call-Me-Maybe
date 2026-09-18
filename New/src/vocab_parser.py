"""Fast schema-aware constrained decoding utilities."""

import json
import math
import re
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from src.models import FunctionDefinition


class TokenTensor(Protocol):
    """Describe the tensor operation used by the decoder."""

    def tolist(self) -> list[list[int]]:
        """Convert token ids to Python lists."""


class LLMModel(Protocol):
    """Describe the public SDK methods used by the decoder."""

    def encode(self, text: str) -> TokenTensor:
        """Encode text into token ids."""

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Return next-token logits for input ids."""


BoolArray = NDArray[np.bool_]
_NUMBER_RE = re.compile(
    r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]*)?(?:[eE][+-]?[0-9]*)?"
)
_JSON_PREFIX = '{"name":"'
_BRIDGE = ',"parameters":{'


@dataclass(frozen=True)
class VocabularyIndex:
    """Store precomputed vocabulary groups used by the decoder."""

    tokens: dict[int, str]
    size: int
    by_first_char: dict[str, tuple[int, ...]]
    string_safe: tuple[int, ...]
    string_special: tuple[int, ...]
    number_tokens: tuple[int, ...]
    literal_tokens: tuple[int, ...]


class DecoderCache:
    """Store model, vocabulary and precomputed decoder data."""

    def __init__(
        self,
        model: LLMModel,
        vocabulary: dict[int, str],
        functions: Sequence[FunctionDefinition],
        vocabulary_size: int | None = None,
    ) -> None:
        """Initialize the decoder cache with model and vocabulary data."""
        self.model = model
        self.vocabulary = vocabulary
        self.functions = tuple(functions)
        self.vocabulary_index = _build_vocabulary_index(
            vocabulary, vocabulary_size
        )


def _build_vocabulary_index(
    vocabulary: dict[int, str], vocabulary_size: int | None = None
) -> VocabularyIndex:
    """Precompute inexpensive token groups once per model vocabulary."""
    by_first: dict[str, list[int]] = {}
    string_safe: list[int] = []
    string_special: list[int] = []
    number_tokens: list[int] = []
    literal_tokens: list[int] = []

    for token_id, token in vocabulary.items():
        if not token:
            continue
        by_first.setdefault(token[0], []).append(token_id)
        if all(ord(char) >= 0x20 and char not in '"\\' for char in token):
            string_safe.append(token_id)
        else:
            string_special.append(token_id)
        if all(char in "0123456789+-.eE,}" for char in token):
            number_tokens.append(token_id)
        if all(char in "truefals" for char in token):
            literal_tokens.append(token_id)

    inferred_size = max(vocabulary, default=-1) + 1
    size = inferred_size if vocabulary_size is None else vocabulary_size
    if size < inferred_size:
        raise ValueError(
            "Vocabulary contains a token id outside the model vocabulary size."
        )

    return VocabularyIndex(
        tokens=vocabulary,
        size=size,
        by_first_char={key: tuple(value) for key, value in by_first.items()},
        string_safe=tuple(string_safe),
        string_special=tuple(string_special),
        number_tokens=tuple(number_tokens),
        literal_tokens=tuple(literal_tokens),
    )


def _decode_prefix_is_valid(text: str, function: FunctionDefinition) -> bool:
    """Check whether text is a valid prefix for one function call."""
    if not text.startswith(_JSON_PREFIX):
        return _JSON_PREFIX.startswith(text)

    rest = text[len(_JSON_PREFIX):]
    name_end = _find_string_end(rest, -1)
    if name_end is None:
        return function.name.startswith(rest)
    if name_end < 0:
        return False
    name = rest[:name_end]
    if name != function.name:
        return False

    suffix = rest[name_end + 1:]
    if len(suffix) < len(_BRIDGE):
        return _BRIDGE.startswith(suffix)
    if not suffix.startswith(_BRIDGE):
        return False
    return _parameters_prefix_is_valid(suffix[len(_BRIDGE):], function)


def _parameters_prefix_is_valid(
    text: str, function: FunctionDefinition
) -> bool:
    """Check whether a parameters object is a valid JSON prefix."""
    if text == "":
        return True

    index = 0
    used: set[str] = set()
    parameters = function.parameters

    while index < len(text):
        if text[index] == "}":
            return used == set(parameters) and text[index:] in ("}", "}}")
        if text[index] != '"':
            return False

        key_end = _find_string_end(text, index)
        if key_end == -1:
            return False
        if key_end is None:
            return any(
                key not in used
                and json.dumps(key, ensure_ascii=False).startswith(
                    text[index:]
                )
                for key in parameters
            )

        try:
            key = json.loads(text[index:key_end + 1])
        except json.JSONDecodeError:
            return False
        if not isinstance(key, str) or key not in parameters or key in used:
            return False
        used.add(key)
        index = key_end + 1

        if index == len(text):
            return True
        if text[index] != ":":
            return False
        index += 1
        if index == len(text):
            return True

        consumed = _value_prefix_length(text[index:], parameters[key].type)
        if consumed is None:
            return True
        if consumed < 0:
            return False
        index += consumed
        if index == len(text):
            return True
        if text[index] == ",":
            if used == set(parameters):
                return False
            index += 1
            if index == len(text):
                return True
            continue
        if text[index] == "}" and used == set(parameters):
            return text[index:] in ("}", "}}")
        return False
    return True


def _find_string_end(text: str, start: int) -> int | None:
    """Return a JSON string closing quote, if one is present."""
    escaped = False
    unicode_digits = 0
    begin = start + 1
    for index in range(begin, len(text)):
        char = text[index]
        if unicode_digits:
            if char not in "0123456789abcdefABCDEF":
                return -1
            unicode_digits -= 1
            continue
        if escaped:
            if char == "u":
                unicode_digits = 4
            elif char not in '"\\/bfnrt':
                return -1
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            return index
        elif ord(char) < 0x20:
            return -1
    return None


def _value_prefix_length(text: str, value_type: str) -> int | None:
    """Validate a value prefix and return its consumed length."""
    if value_type == "string":
        return _string_prefix_length(text)
    if value_type == "boolean":
        return _literal_prefix_length(text, ("true", "false"))
    if value_type == "number":
        return _number_prefix_length(text)
    if value_type == "integer":
        return _integer_prefix_length(text)
    return -1


def _string_prefix_length(text: str) -> int | None:
    """Validate a JSON string prefix."""
    if not text.startswith('"'):
        return -1
    end = _find_string_end(text, 0)
    if end == -1:
        return -1
    if end is None:
        return None
    return end + 1


def _literal_prefix_length(text: str, literals: tuple[str, ...]) -> int | None:
    """Validate a JSON literal prefix."""
    for literal in literals:
        if literal.startswith(text):
            return None if len(text) < len(literal) else len(literal)
        if text.startswith(literal):
            return len(literal)
    return -1


def _number_prefix_length(text: str) -> int | None:
    """Return the consumed length of a valid JSON-number prefix."""
    if text == "":
        return None

    number_pattern = r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
    complete = re.fullmatch(number_pattern, text)
    if complete is not None:
        return len(text)

    prefix_match = re.match(number_pattern, text)
    if prefix_match is not None:
        consumed = len(prefix_match.group(0))
        if consumed < len(text) and text[consumed] in ",}":
            return consumed

    incomplete_patterns = (
        r"-",
        r"-?(?:0|[1-9][0-9]*)\.",
        r"-?(?:0|[1-9][0-9]*)[eE]",
        r"-?(?:0|[1-9][0-9]*)[eE][+-]",
        r"-?(?:0|[1-9][0-9]*)\.[0-9]+[eE]",
        r"-?(?:0|[1-9][0-9]*)\.[0-9]+[eE][+-]",
    )
    if any(
        re.fullmatch(pattern, text) is not None
        for pattern in incomplete_patterns
    ):
        return None
    return -1


def _integer_prefix_length(text: str) -> int | None:
    if text == "":
        return None

    integer_pattern = r"-?(?:0|[1-9][0-9]*)"
    complete = re.fullmatch(integer_pattern, text)
    if complete is not None:
        return len(text)

    prefix_match = re.match(integer_pattern, text)
    if prefix_match is not None:
        consumed = len(prefix_match.group(0))
        if consumed < len(text) and text[consumed] in ",}":
            return consumed

    if text == "-":
        return None
    return -1


def _token_can_extend_prefix(
    current_text: str,
    token_text: str,
    functions: Sequence[FunctionDefinition],
) -> bool:
    """Return whether a token keeps at least one schema-valid prefix."""
    candidate = current_text + token_text
    return any(
        _decode_prefix_is_valid(candidate, function) for function in functions
    )


def _candidate_ids_by_first_char(
    index: VocabularyIndex,
    characters: str,
) -> tuple[int, ...]:
    """Return token ids whose text starts with one of ``characters``."""
    result: list[int] = []
    seen: set[int] = set()
    for character in characters:
        for token_id in index.by_first_char.get(character, ()):
            if token_id not in seen:
                result.append(token_id)
                seen.add(token_id)
    return tuple(result)


def _parameter_state(
    text: str,
    function: FunctionDefinition,
) -> tuple[str, str | None, set[str]]:
    """Return the parameter-object state for a valid or incomplete prefix."""
    if text == "":
        return "key", None, set()

    index = 0
    used: set[str] = set()
    parameters = function.parameters
    expected_keys = set(parameters)

    while index < len(text):
        if text[index] == "}":
            if used == expected_keys and text[index:] == "}}":
                return "complete", None, used
            if used == expected_keys and text[index:] == "}":
                return "outer_close", None, used
            return "invalid", None, used

        if text[index] != '"':
            return "invalid", None, used
        key_end = _find_string_end(text, index)
        if key_end == -1:
            return "invalid", None, used
        if key_end is None:
            return "key", text[index + 1:], used

        try:
            key = json.loads(text[index:key_end + 1])
        except json.JSONDecodeError:
            return "invalid", None, used
        if not isinstance(key, str) or key not in parameters or key in used:
            return "invalid", None, used
        used.add(key)
        index = key_end + 1

        if index == len(text):
            return "colon", key, used
        if text[index] != ":":
            return "invalid", None, used
        index += 1
        if index == len(text):
            return "value_start", key, used

        value_text = text[index:]
        value_type = parameters[key].type
        consumed = _value_prefix_length(value_text, value_type)
        if consumed is None:
            if value_type == "string" and value_text.startswith('"'):
                return "string", key, used
            if value_type in ("number", "integer"):
                return "number", key, used
            if value_type == "boolean":
                return "boolean", key, used
            return "value_start", key, used
        if consumed < 0:
            return "invalid", None, used

        index += consumed
        if index == len(text):
            if value_type in ("number", "integer"):
                return "number", key, used
            return "after_value", key, used
        if text[index] == ",":
            if used == expected_keys:
                return "invalid", None, used
            index += 1
            if index == len(text):
                return "key", None, used
            continue
        if text[index] == "}" and used == expected_keys:
            if text[index:] == "}":
                return "outer_close", None, used
            if text[index:] == "}}":
                return "complete", None, used
        return "invalid", None, used

    return "key", None, used


def _selected_function_from_prefix(
    text: str,
    functions: Sequence[FunctionDefinition],
) -> FunctionDefinition | None:
    """Return the selected function when its name is complete."""
    if not text.startswith(_JSON_PREFIX):
        return None
    rest = text[len(_JSON_PREFIX):]
    name_end = _find_string_end(rest, -1)
    if name_end is None or name_end < 0:
        return None
    return _find_function(rest[:name_end], functions)


def _name_prefix(text: str) -> str | None:
    """Return the unfinished function-name prefix, if the name is open."""
    if not text.startswith(_JSON_PREFIX):
        return "" if _JSON_PREFIX.startswith(text) else None
    rest = text[len(_JSON_PREFIX):]
    name_end = _find_string_end(rest, -1)
    return rest if name_end is None else None


def _parameter_text(text: str) -> str | None:
    """Return the text inside the parameters object, when it exists."""
    if not text.startswith(_JSON_PREFIX):
        return None
    rest = text[len(_JSON_PREFIX):]
    name_end = _find_string_end(rest, -1)
    if name_end is None or name_end < 0:
        return None
    suffix = rest[name_end + 1:]
    if not _BRIDGE.startswith(suffix) and not suffix.startswith(_BRIDGE):
        return None
    if len(suffix) < len(_BRIDGE):
        return None
    return suffix[len(_BRIDGE):]


def _key_prefix(text: str, function: FunctionDefinition) -> str | None:
    """Return the current unfinished parameter-key text."""
    parameter_text = _parameter_text(text)
    if parameter_text is None:
        return None
    state, _, _ = _parameter_state(parameter_text, function)
    if state != "key":
        return None
    if parameter_text == "" or parameter_text.endswith(","):
        return ""
    opening_quote = parameter_text.rfind('"')
    if opening_quote < 0:
        return None
    return parameter_text[opening_quote + 1:]


def _string_context(text: str) -> tuple[str, str | None] | None:
    """Return whether the current prefix is inside a JSON string."""
    in_string = False
    escaped = False
    opening = -1
    for index, character in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
            opening = index

    if not in_string:
        return None
    if opening == len(_JSON_PREFIX):
        return "name", None
    bridge_start = text.find(_BRIDGE)
    if bridge_start >= 0:
        parameter_start = bridge_start + len(_BRIDGE)
        before_opening = text[parameter_start:opening]
        if before_opening.endswith(("{", ",")) or before_opening == "":
            return "key", None
        if before_opening.endswith(":"):
            return "value", None
    return "value", None


def _allowed_token_ids(
    current_text: str,
    cache: DecoderCache,
    function: FunctionDefinition | None,
) -> tuple[Sequence[int], str]:
    """Return candidate token ids and the current schema state."""
    index = cache.vocabulary_index
    name_prefix = _name_prefix(current_text)
    if name_prefix is not None:
        possible_names = [
            item.name
            for item in cache.functions
            if item.name.startswith(name_prefix)
        ]
        next_chars = {
            name[len(name_prefix)]
            for name in possible_names
            if len(name) > len(name_prefix)
        }
        candidates = _candidate_ids_by_first_char(
            index, "".join(next_chars) + '"'
        )
        return candidates, "name"

    selected = function or _selected_function_from_prefix(
        current_text, cache.functions
    )
    if selected is None:
        return (), "invalid"

    rest = current_text[len(_JSON_PREFIX):]
    name_end = _find_string_end(rest, -1)
    if name_end is None or name_end < 0:
        return (), "invalid"
    suffix = rest[name_end + 1:]
    if len(suffix) < len(_BRIDGE):
        return _candidate_ids_by_first_char(
            index, _BRIDGE[len(suffix)]
        ), "bridge"
    if not suffix.startswith(_BRIDGE):
        return (), "invalid"

    parameter_text = suffix[len(_BRIDGE):]
    state, key, used = _parameter_state(parameter_text, selected)

    if state == "key":
        prefix = _key_prefix(current_text, selected)
        if prefix is None:
            return (), "invalid"
        possible_keys = [
            name
            for name in selected.parameters
            if name not in used and name.startswith(prefix)
        ]
        next_chars = {
            name[len(prefix)]
            for name in possible_keys
            if len(name) > len(prefix)
        }
        return _candidate_ids_by_first_char(
            index, "".join(next_chars) + '"'
        ), "key"

    if state == "colon":
        return _candidate_ids_by_first_char(index, ":"), "colon"

    if state == "value_start":
        value_type = selected.parameters[key].type if key else None
        if value_type == "string":
            return _candidate_ids_by_first_char(index, '"'), "string"
        if value_type in ("number", "integer"):
            return _candidate_ids_by_first_char(index, "-0123456789"), "number"
        if value_type == "boolean":
            return _candidate_ids_by_first_char(index, "tf"), "boolean"
        return (), "invalid"

    if state == "string":
        return index.string_safe + index.string_special, "string"

    if state == "number":
        return index.number_tokens, "number"

    if state == "boolean":
        return index.literal_tokens, "boolean"

    if state == "after_value":
        if used == set(selected.parameters):
            candidates = _candidate_ids_by_first_char(index, "}")
        else:
            candidates = _candidate_ids_by_first_char(index, ",")
        return candidates, "after_value"

    if state == "outer_close":
        return _candidate_ids_by_first_char(index, "}"), "outer_close"
    if state == "complete":
        return (), "complete"
    return (), "invalid"


def _allowed_token_mask(
    current_text: str,
    index: VocabularyIndex,
    functions: Sequence[FunctionDefinition],
) -> BoolArray:
    """Build the schema-constrained mask used by compatibility tests."""
    mask = np.zeros(index.size, dtype=np.bool_)
    cache = DecoderCache.__new__(DecoderCache)
    cache.vocabulary_index = index
    cache.functions = tuple(functions)
    candidates, _ = _allowed_token_ids(current_text, cache, None)
    for token_id in candidates:
        token_text = index.tokens.get(token_id)
        if token_text and _token_can_extend_prefix(
            current_text, token_text, functions
        ):
            mask[token_id] = True
    return mask


def _find_function(
    name: str | None,
    functions: Sequence[FunctionDefinition],
) -> FunctionDefinition | None:
    """Find a function definition by name."""
    return next(
        (function for function in functions if function.name == name),
        None,
    )


def _prefix_is_complete(
    text: str, functions: Sequence[FunctionDefinition]
) -> bool:
    """Return whether the generated text is a complete valid call."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return False
    if not isinstance(data, dict) or set(data) != {"name", "parameters"}:
        return False
    name = data.get("name")
    function = _find_function(
        name if isinstance(name, str) else None, functions
    )
    if function is None:
        return False
    parameters = data.get("parameters")
    if (
        not isinstance(parameters, dict)
        or set(parameters) != set(function.parameters)
    ):
        return False
    for key, definition in function.parameters.items():
        value = parameters[key]
        if definition.type == "string" and not isinstance(value, str):
            return False
        if definition.type == "number" and (
            not isinstance(value, (int, float)) or isinstance(value, bool)
        ):
            return False
        if definition.type == "integer" and (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or (isinstance(value, float) and not value.is_integer())
        ):
            return False
        if definition.type == "boolean" and not isinstance(value, bool):
            return False
    return True


def validate_function_call(
    data: object,
    prompt: str,
    functions: Sequence[FunctionDefinition],
) -> dict[str, object]:
    """Validate and normalize one generated function call."""
    if not isinstance(data, dict):
        raise ValueError("Generated output must be a JSON object.")
    if set(data) != {"name", "parameters"}:
        raise ValueError("Generated output contains invalid keys.")
    name = data.get("name")
    if not isinstance(name, str):
        raise ValueError("Function name must be a string.")
    function = _find_function(name, functions)
    if function is None:
        raise ValueError(f"Unknown function: {name}")
    parameters = data.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("Function parameters must be an object.")
    expected = set(function.parameters)
    actual = set(parameters)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing parameters: {', '.join(missing)}")
        if extra:
            details.append(f"unknown parameters: {', '.join(extra)}")
        raise ValueError("Invalid parameters (" + "; ".join(details) + ").")

    normalized: dict[str, object] = {}
    for key, definition in function.parameters.items():
        value = parameters[key]
        if definition.type == "string":
            if not isinstance(value, str):
                raise ValueError(f"Parameter '{key}' must be a string.")
            normalized[key] = value
        elif definition.type == "boolean":
            if not isinstance(value, bool):
                raise ValueError(f"Parameter '{key}' must be a boolean.")
            normalized[key] = value
        elif definition.type == "integer":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"Parameter '{key}' must be an integer.")
            if isinstance(value, float) and not value.is_integer():
                raise ValueError(f"Parameter '{key}' must be an integer.")
            normalized[key] = int(value)
        else:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"Parameter '{key}' must be a number.")
            if not math.isfinite(float(value)):
                raise ValueError(f"Parameter '{key}' must be a finite number.")
            normalized[key] = float(value)
    return {"prompt": prompt, "name": name, "parameters": normalized}


def _best_valid_token(
    generated: str,
    logits: NDArray[np.float64],
    cache: DecoderCache,
    candidates: Sequence[int],
    functions: Sequence[FunctionDefinition],
    fast_string: bool = False,
) -> int:
    """Return the highest-logit token that keeps the prefix schema-valid."""
    candidate_ids = [
        token_id for token_id in candidates
        if 0 <= token_id < len(logits)
    ]
    if not candidate_ids:
        raise RuntimeError("Constrained decoding found no valid next token.")

    if fast_string:
        safe_count = len(cache.vocabulary_index.string_safe)
        safe = candidate_ids[:safe_count]
        best_id: int | None = None
        best_score = -np.inf
        if safe:
            safe_array = np.asarray(safe, dtype=np.int64)
            position = int(np.argmax(logits[safe_array]))
            best_id = int(safe_array[position])
            best_score = float(logits[best_id])

        for token_id in candidate_ids[safe_count:]:
            token_text = cache.vocabulary_index.tokens.get(token_id)
            if token_text is None or float(logits[token_id]) <= best_score:
                continue
            if _token_can_extend_prefix(generated, token_text, functions):
                best_id = token_id
                best_score = float(logits[token_id])
        if best_id is not None:
            return best_id

    candidate_array = np.asarray(candidate_ids, dtype=np.int64)
    order = np.argsort(logits[candidate_array])[::-1]
    for position in order:
        token_id = int(candidate_array[int(position)])
        token_text = cache.vocabulary_index.tokens.get(token_id)
        if token_text and _token_can_extend_prefix(
            generated, token_text, functions
        ):
            return token_id
    raise RuntimeError("Constrained decoding found no valid next token.")


def _select_token(
    generated: str,
    input_ids: list[int],
    cache: DecoderCache,
    function: FunctionDefinition | None = None,
) -> tuple[str, int]:
    """Select the highest-logit token allowed by the current schema state."""
    logits = np.asarray(
        cache.model.get_logits_from_input_ids(input_ids), dtype=float
    )
    if len(logits) != cache.vocabulary_index.size:
        raise RuntimeError("Vocabulary and model logits have different sizes.")

    candidates, state = _allowed_token_ids(generated, cache, function)
    if state == "complete":
        raise RuntimeError("The generated function call is already complete.")
    if state == "invalid" or not candidates:
        raise RuntimeError("Constrained decoding found no valid next token.")

    selected_function = function or _selected_function_from_prefix(
        generated, cache.functions
    )
    token_id = _best_valid_token(
        generated,
        logits,
        cache,
        candidates,
        (selected_function,)
        if selected_function is not None
        else cache.functions,
        fast_string=state == "string",
    )
    token_text = cache.vocabulary_index.tokens.get(token_id)
    if token_text is None:
        raise RuntimeError("The model selected an unknown vocabulary token.")
    return token_text, token_id


def generate_constrained_json(
    prompt_text: str,
    cache: DecoderCache,
    max_generated_tokens: int = 150,
) -> str:
    """Generate a function call with token-level schema constraints."""
    schemas = [
        function.model_dump(mode="json") for function in cache.functions
    ]
    prompt = (
        "You are a function calling system. Select the best function and "
        "provide all required arguments. Output only the JSON function call.\n"
        f"Available functions: {json.dumps(schemas, separators=(',', ':'))}\n"
        f"User: {prompt_text}\n"
        "Tool Call: "
    )

    generated = _JSON_PREFIX
    encoded = cache.model.encode(prompt + generated)
    input_ids = list(encoded.tolist()[0])

    for _ in range(max_generated_tokens):
        if _prefix_is_complete(generated, cache.functions):
            return generated

        selected = _selected_function_from_prefix(generated, cache.functions)
        token_text, token_id = _select_token(
            generated, input_ids, cache, selected
        )
        generated += token_text
        input_ids.append(token_id)

    if not _prefix_is_complete(generated, cache.functions):
        raise RuntimeError(
            f"Generation stopped after {max_generated_tokens} tokens "
            "without producing a complete valid function call."
        )
    return generated

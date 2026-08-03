import json
import numpy as np
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from .models import FunctionDefinition

DIGITS = set("0123456789")


def load_vocab(model: Small_LLM_Model) -> dict[int, str]:
    """Load the model vocabulary as an ``{id: token_text}`` mapping."""
    path = model.get_path_to_vocab_file()
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    id_to_text: dict[int, str] = {}
    for token_text, token_id in raw.items():
        id_to_text[token_id] = token_text.replace("Ġ", " ")
    return id_to_text


def choose_token(logits: list[float], allowed_ids: set[int]) -> int:
    """Return the highest-logit token id among ``allowed_ids``."""
    if len(allowed_ids) == 0:
        raise ValueError("no valid tokens")
    arr = np.array(logits)
    masked = np.full(arr.shape, -np.inf)
    allowed = list(allowed_ids)
    masked[allowed] = arr[allowed]
    return int(np.argmax(masked))


def is_digit_token(text: str) -> bool:
    """True if ``text`` is non-empty and made only of digits."""
    return text != "" and all(c in DIGITS for c in text)


def is_string_content_token(text: str) -> bool:
    """True if ``text`` is valid JSON string content (no quote/backslash)."""
    return text != "" and all(c != '"' and c != "\\" for c in text)


def literal_allowed_ids(remaining: str, vocab: dict[int, str]) -> set[int]:
    """Token ids whose text is a non-empty prefix of ``remaining``."""
    return {tid for tid, text in vocab.items()
            if text != "" and remaining.startswith(text)}


def number_allowed_ids(vocab: dict[int, str], has_digit: bool,
                       terminator_ids: set[int]) -> set[int]:
    """Digit tokens, plus the terminator once a digit was emitted."""
    allowed = {tid for tid, text in vocab.items() if is_digit_token(text)}
    if has_digit:
        allowed |= terminator_ids
    return allowed


def string_allowed_ids(vocab: dict[int, str],
                       terminator_ids: set[int]) -> set[int]:
    """String-content tokens, plus the closing-quote terminator."""
    allowed = {tid for tid, text in vocab.items()
               if is_string_content_token(text)}
    return allowed | terminator_ids


def funnel_allowed_ids(vocab: dict[int, str], produced: str,
                       candidates: list[str]) -> set[int]:
    """Tokens keeping ``produced`` a prefix of some still-possible
    candidate."""
    alive = [c for c in candidates if c.startswith(produced)]
    return {tid for tid, text in vocab.items()
            if text != "" and any(c.startswith(produced + text)
                                  for c in alive)}


class Decoder():
    """Constrained decoder: builds a valid function-call JSON
    token by token."""

    def __init__(self, model: Small_LLM_Model, vocab: dict[int, str],
                 functions: list[FunctionDefinition]) -> None:
        self.model = model
        self.vocab = vocab
        self.functions = functions
        self.ids: list[int] = []
        self.output = ""

    def emit(self, token_id: int) -> str:
        """Append one already-chosen token to the running ids/output."""
        text = self.vocab[token_id]
        self.ids.append(token_id)
        self.output += text
        return text

    def longest_prefix_token(self, remaining: str) -> int:
        """Return the id of the longest vocab token that starts `remaining`."""
        best_id = -1
        best_len = 0
        for token_id, text in self.vocab.items():
            if text == "" or not remaining.startswith(text):
                continue
            if len(text) > best_len:
                best_id = token_id
                best_len = len(text)
        return best_id

    def emit_literal(self, text: str) -> None:
        """Emit a fixed literal string — no model call (deterministic)."""
        remaining = text
        while remaining:
            token_id = self.longest_prefix_token(remaining)
            self.emit(token_id)
            remaining = remaining[len(self.vocab[token_id]):]

    def choose(self, allowed_ids: set[int]) -> str:
        """Pick a token from allowed_ids via the model."""
        if len(allowed_ids) == 1:
            token_id = next(iter(allowed_ids))
        else:
            logits = self.model.get_logits_from_input_ids(self.ids)
            token_id = choose_token(logits, allowed_ids)
        return self.emit(token_id)

    def emit_funnel(self, candidates: list[str]) -> str:
        """Let the model pick one full candidate via a prefix funnel."""
        produced = ""
        while produced not in candidates:
            alive = [c for c in candidates if c.startswith(produced)]
            if len(alive) == 1:
                self.emit_literal(alive[0][len(produced):])
                produced = alive[0]
            else:
                produced += self.choose(
                    funnel_allowed_ids(self.vocab, produced, candidates))
        return produced

    def emit_value(self, ptype: str, tail: str) -> None:
        """Emit a value of `ptype`, then the literal `tail`."""
        if ptype == "number":
            term = literal_allowed_ids(tail, self.vocab)
            has_digit = False
            length = 0
            while True:
                if has_digit and length > 20:
                    self.emit_literal(tail)
                    return
                got = self.choose(number_allowed_ids(self.vocab, has_digit,
                                                     term))
                if is_digit_token(got):
                    has_digit = True
                    length += len(got)
                else:
                    self.emit_literal(tail[len(got):])
                    return
        elif ptype == "string":
            self.emit_literal('"')
            full = '"' + tail
            term = literal_allowed_ids(full, self.vocab)
            length = 0
            while True:
                if length > 80:
                    self.emit_literal(full)
                    return
                got = self.choose(string_allowed_ids(self.vocab, term))
                if not is_string_content_token(got):
                    self.emit_literal(full[len(got):])
                    return
                length += len(got)
        elif ptype == "boolean":
            self.emit_funnel(["true", "false"])
            self.emit_literal(tail)
        else:
            raise ValueError(f"unsupported parameter type: {ptype}")

    def generate(self, prompt_text: str) -> str:
        """Constrained-decode one function call as a JSON string."""
        self.ids = self.model.encode(prompt_text)[0].tolist()
        self.output = ""
        names = [f.name for f in self.functions]
        by_name = {f.name: f for f in self.functions}

        self.emit_literal('{"name": "')
        name = self.emit_funnel(names)
        func = by_name[name]
        self.emit_literal('", "parameters": {')
        params = list(func.parameters.items())
        if not params:
            self.emit_literal('}}')
        else:
            for i, (key, spec) in enumerate(params):
                self.emit_literal(f'"{key}": ')
                tail = '}}' if i == len(params) - 1 else ', '
                self.emit_value(spec.type, tail)
        return self.output

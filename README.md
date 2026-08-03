*This project has been created as part of the 42 curriculum by csilva.*

# call me maybe

Function calling in LLMs via constrained decoding (Qwen3-0.6B).

## Description

This project turns a natural-language request into a structured function call.
Given a prompt like "What is the sum of 2 and 3?", it does not answer "5" — it
produces the function to call and its typed arguments:

    {"name": "fn_add_numbers", "parameters": {"a": 2, "b": 3}}

Small models like Qwen3-0.6B are unreliable at producing valid JSON when simply
asked to. Instead of trusting the model, the program uses **constrained
decoding**: at every step it masks the model's logits so that only tokens that
keep the output a valid function call (matching functions_definition.json) can
be chosen. This guarantees 100% valid JSON.

## Instructions

    make install        # uv sync
    make run            # uv run python -m src
    make lint           # flake8 + mypy

Custom paths:

    uv run python -m src \
        --functions_definition data/input/functions_definition.json \
        --input data/input/function_calling_tests.json \
        --output data/output/function_calling_results.json

The first run downloads the Qwen3-0.6B weights (~1.2 GB) and caches them.

## Algorithm — constrained decoding

The model generates one token at a time. At each step:

1. Ask the SDK for the raw logits of the next token.
2. Compute the set of token ids valid at the current position of the JSON.
3. Set every other logit to -infinity (masking) and take the argmax, so the
   chosen token is always the best *valid* one.
4. Append it, advance the position, repeat until the JSON is complete.

The output shape is `{"name": "<name>", "parameters": {"<key>": <value>, ...}}`.
Most of it is fixed text; only three positions are "open":

- **function name**: a prefix funnel over the available names — only tokens that
  continue a still-possible name are allowed, so an invalid name can never be
  built. The model picks which name (chosen by the LLM, not heuristics).
- **number value**: digit tokens are allowed; after one digit, the terminator
  (the text that follows) is allowed too, so the model decides when to stop.
- **string value**: any non-quote character is content; the closing quote ends
  it.

Token texts come from the vocabulary file (get_path_to_vocab_file), mapping
id -> text and turning the byte-level space marker "Ġ" into a real space.

## Design decisions

- **Fixed parts are emitted without calling the model** (they are deterministic),
  which cut model calls ~10x and brought the run under the time limit.
- **pydantic validation** of every generated JSON as a safety net.
- **Force-close on runaway values**: if a value grows too long (the small model
  sometimes rambles), it is forced closed, keeping the JSON valid.

## Performance analysis

- Function selection correct on all provided prompts; argument extraction correct
  except the regex field of one hard substitution prompt — above the 90% target.
- All prompts processed in ~2 minutes (limit is 5).
- 100% valid, schema-compliant JSON.

## Challenges faced

- **Speed**: a naive version called the model per character and blew past 5 min;
  skipping the model on deterministic parts fixed it.
- **Where a value ends**: solved with the "terminator" idea.
- **Real vocabulary**: the "Ġ" marker and multi-character tokens, handled by
  reading the raw vocab file.
- **A rambling model**: force-close keeps output valid on the hardest prompt.

## Testing strategy

Each piece was tested in isolation with a fake vocabulary and fake logits
(masking/argmax, fixed-literal generation, number/string states, name funnel)
before running the full pipeline on the real model and checking the output JSON.

## Resources

- Constrained decoding / structured generation.
- Byte-Pair Encoding and the byte-level tokenizer.
- Hugging Face Transformers and the provided llm_sdk.

### AI usage

I used an AI assistant (Claude) mainly as a **tutor**,
not as a code generator I copied blindly.

- **Learning the theory.** I went through the whole idea of constrained
  decoding step by step with the AI: what tokens and logits are, how masking
  with -infinity forces valid output, the state machine (fixed parts, the
  function-name funnel, number/string values), and the quirks of the real Qwen
  vocabulary (the "Ġ" space marker, multi-character tokens). I answered
  comprehension questions along the way to make sure I actually understood
  each part before moving on.

- **Writing the code.** I wrote the implementation myself (the loaders, the
  prompt builder, the vocabulary loader, the masking/argmax helper, the
  per-state "allowed tokens" functions, and the `Decoder` class). The AI
  reviewed what I wrote, explained the parts I did not understand, and helped
  me fix bugs.

- **Debugging and optimizing.** The AI helped me diagnose two real problems:
  the program going over the 5-minute limit (fixed by not calling the model on
  deterministic parts) and the small model rambling on a regex (fixed with the
  force-close mechanism). It also helped me get `flake8`/`mypy` clean.
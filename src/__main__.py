import argparse
import json
import os
import sys
from pydantic import ValidationError
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from .loader import load_func_defs, load_prompts
from .prompt import build_prompt
from .decoder import load_vocab, Decoder
from .models import FunctionCall


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions_definition",
                        default="data/input/functions_definition.json")
    parser.add_argument("--input",
                        default="data/input/function_calling_tests.json")
    parser.add_argument("--output",
                        default="data/output/function_calling_results.json")
    args = parser.parse_args()

    try:
        functions = load_func_defs(args.functions_definition)
        prompts = load_prompts(args.input)
    except (FileNotFoundError, ValueError, ValidationError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    model = Small_LLM_Model()
    vocab = load_vocab(model)
    decoder = Decoder(model, vocab, functions)

    results = []
    for item in prompts:
        print(f"processing: {item.prompt}", file=sys.stderr)
        try:
            prompt_text = build_prompt(functions, item.prompt)
            raw = decoder.generate(prompt_text)
            call = FunctionCall(**json.loads(raw))
            results.append({"prompt": item.prompt,
                            "name": call.name,
                            "parameters": call.parameters})
        except (ValueError, ValidationError) as e:
            print(f"Error on prompt {item.prompt!r}: {e}", file=sys.stderr)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

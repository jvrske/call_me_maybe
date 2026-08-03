from .models import FunctionDefinition


def build_prompt(functions: list[FunctionDefinition], user_prompt: str) -> str:
    instructions = ('You are a function-calling assistant. Respond ONLY with '
                    'JSON like {"name": ..., "parameters": {...}}.\n\n'
                    'Available functions:')

    lines = []
    for func in functions:
        params = ", ".join(f"{name}: {spec.type}"
                           for name, spec in func.parameters.items())
        line = f"- {func.name}({params}): {func.description}"
        lines.append(line)
    prompt = f"\nUser request: {user_prompt}\nResponse:"
    return instructions + "\n" + "\n".join(lines) + "\n" + prompt

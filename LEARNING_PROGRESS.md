# Progresso de Aprendizado — call_me_maybe

> Arquivo interno para retomar as sessões de estudo. NÃO faz parte da entrega do projeto.
> Última atualização: 2026-06-10

## Como estamos trabalhando

- **Modo professor.** O objetivo é que o ALUNO (jvrske) entenda e saiba justificar
  cada linha na defesa do 42. Nada de só colar código.
- **Nível do aluno:** iniciante total em LLMs (tokens, logits, decoding do zero).
- **Modo de código:** alternar entre os 3 conforme a dificuldade:
  1. Aluno escreve tudo sozinho (partes que ele já domina)
  2. Esqueleto + aluno preenche a lógica
  3. Pair programming (conceitos novos/difíceis, ex: constrained decoding)
  Sempre avisar qual modo estamos usando.
- **REGRA FIXA:** toda explicação termina com uma PERGUNTA pro aluno responder,
  pra confirmar que fixou o conceito.
- Branch de trabalho: `claude/sleepy-sagan-ywf1cw`
- Idioma: explicações em português, código/README em inglês.

## Resumo do projeto (subject)

Sistema de function calling: traduz prompt em linguagem natural -> chamada de função
estruturada em JSON. NÃO responde a pergunta, devolve {name, parameters}.
Modelo: Qwen/Qwen3-0.6B. Técnica obrigatória: **constrained decoding** (garantir
100% JSON válido + conformidade de schema). Proibido: dspy, pytorch, transformers,
outlines, huggingface no código principal. Permitido: numpy, pydantic, json, llm_sdk.
Escolha da função deve vir do LLM, não de heurística.

Comando de execução:
`uv run python -m src [--functions_definition <f>] [--input <f>] [--output <f>]`
Defaults: lê de data/input/, escreve em data/output/function_calling_results.json

Output: array de objetos com EXATAMENTE as chaves: prompt, name, parameters.

## Plano de sessões

| Sessão | Conteúdo | Status |
|---|---|---|
| 0 | Panorama geral + analogia dos ímãs de geladeira | FEITO |
| 1 | Fundamentos: tokens, logits, vocab + models.py + loader | a fazer |
| 2 | CLI / entrypoint (`__main__.py`) + pipeline básico | a fazer |
| 3 | Tokenização e vocab.json (BPE, o símbolo "Ġ") | a fazer |
| 4 | Constrained decoding — teoria (máquina de estados, mascarar logits) | a fazer |
| 5 | Constrained decoding — estrutura JSON (forçar {, ", chaves) | a fazer |
| 6 | Constrained decoding — tipos (number/string/boolean, schema) | a fazer |
| 7 | Seleção de função via LLM + integração | a fazer |
| 8 | Lint, mypy, testes, README completo | a fazer |

## Onde paramos

Terminamos a SESSÃO 0 (panorama). Expliquei:
- O que o programa faz (prompt -> JSON com name+parameters)
- Por que é difícil (modelo de 0.6B acerta JSON só ~30% sozinho)
- A ideia de constrained decoding (analogia: esconder os ímãs de geladeira
  inválidos; o modelo "escolhe" mas só sobram opções válidas)

**PERGUNTA PENDENTE deixada pro aluno** (responder no início da sessão 1):
> Na analogia da criança com os ímãs de geladeira — quem "escolhe" o próximo
> ímã, e qual é o nosso papel? Por que isso resolve o problema do modelo ser pequeno?

## Estado do código (o que já existe no repo)

- `src/models.py`: parcial — só tem `ParameterSpec` (Literal number/string/boolean)
- `main.py`: vazio (provavelmente vamos remover; entrypoint deve ser `src/__main__.py`)
- `Makefile`: completo (install/run/debug/clean/lint/lint-strict/test)
- `pyproject.toml`: completo (numpy, pydantic, llm-sdk como dep editável)
- `llm_sdk/`: SDK fornecido, completo. Métodos úteis:
  - `get_logits_from_input_ids(ids) -> list[float]`
  - `get_path_to_vocab_file() -> str`
  - `encode(text) -> Tensor`, `decode(ids) -> str`
  - extras: `get_path_to_merges_file()`, `get_path_to_tokenizer_file()`
- `data/input/`: tem function_calling_tests.json e functions_definition.json
- `README.md`, `.gitignore`: existem (revisar depois)

## Checklist de implementação (entrega)

- [ ] models.py: FunctionDefinition, FunctionCall, PromptInput (todos pydantic)
- [ ] loader: ler os 2 JSONs de input com tratamento de erro (JSON inválido / ausente)
- [ ] constrained decoder: vocab + máquina de estados JSON + mascarar logits + schema
- [ ] seleção de função via LLM (não heurística)
- [ ] `src/__main__.py`: argparse (--functions_definition/--input/--output) + defaults
- [ ] gerar data/output/function_calling_results.json
- [ ] type hints + passa mypy; docstrings PEP257; passa flake8
- [ ] README: algoritmo, decisões, performance, desafios, testes, exemplos
- [ ] NÃO commitar data/output/

## Metas de qualidade (subject V.5)

- 90%+ acerto na seleção de função e extração de args
- 100% JSON válido e schema-compliant
- Processar todos os prompts em < 5 min
- Tratamento de erro robusto (nunca crashar)

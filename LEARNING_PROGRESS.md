# Progresso de Aprendizado — call_me_maybe

> Arquivo interno para retomar as sessões de estudo. NÃO faz parte da entrega do projeto.
> Última atualização: 2026-06-11

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
- Branch de trabalho: `claude/pensive-gauss-gr03bv` (antiga: `claude/sleepy-sagan-ywf1cw`, PR #1)
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
| 1 | Fundamentos: tokens, logits, vocab + models.py + loader | EM ANDAMENTO |
| 2 | CLI / entrypoint (`__main__.py`) + pipeline básico | a fazer |
| 3 | Tokenização e vocab.json (BPE, o símbolo "Ġ") | a fazer |
| 4 | Constrained decoding — teoria (máquina de estados, mascarar logits) | a fazer |
| 5 | Constrained decoding — estrutura JSON (forçar {, ", chaves) | a fazer |
| 6 | Constrained decoding — tipos (number/string/boolean, schema) | a fazer |
| 7 | Seleção de função via LLM + integração | a fazer |
| 8 | Lint, mypy, testes, README completo | a fazer |

## Onde paramos

SESSÃO 0 fechada. Aluno entendeu a analogia dos ímãs e a mecânica de logits:
- Confirmou que QUEM escolhe o token é o MODELO, não a gente (corrigido erro inicial).
- Nosso papel = mascarar (rebaixar pra -inf) os tokens inválidos; modelo escolhe o de maior logit entre os que sobram.
- Exemplo validado: logit 9 (inválido, mascarado) vs logit 4 (válido) -> sai o 4.
- Termo correto introduzido: "máquina de estados" decide validade (não "parseamento").

SESSÃO 1 — PARTE TEÓRICA FEITA. Aluno entendeu e acertou:
- token = tijolinho de texto (palavra/pedaço/símbolo); vocab = tabela token<->ID; logits = lista de notas (1 por token do vocab).
- Índice do logits = ID do token; valor = nota/logit (quanto o modelo "quer" aquele token).
- Ciclo de geração: ids -> get_logits_from_input_ids -> [NOSSO mascaramento, logits[id_invalido]=-inf] -> escolhe maior -> append -> repete.

SESSÃO 1 — PARTE PRÁTICA (models.py): EM PAUSA, retomar AQUI.
Já existe em src/models.py: ParamType (Literal) + ParameterSpec (só campo `type`, extra="forbid").
Faltam 3 modelos pydantic a criar:
  1. FunctionDefinition: name:str, description:str, parameters:dict[str,ParameterSpec], returns:ParameterSpec
  2. FunctionCall (o que produzimos): name:str, parameters:dict[str, valor real]
  3. PromptInput: prompt:str

>>> PERGUNTA PENDENTE pro aluno (responder ao voltar, ANTES de escrever código):
    Por que `parameters` é um dict[str, ParameterSpec] e nao uma list?
    (dica: no functions_definition.json os params aparecem como chaves nomeadas "a","b"...)
>>> TAREFA pendente: aluno escreve a classe FunctionDefinition sozinho (modo 1), eu reviso.
    Depois seguimos pra FunctionCall, PromptInput, e o loader dos 2 JSONs (sessão 1/2).

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

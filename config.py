from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

IMAGENS_DIR = BASE_DIR / "imagens"
GABARITOS_DIR = BASE_DIR / "gabaritos"

# Saída de rodadas novas (ignorada pelo git). Cada chamada de main.py grava um
# par de CSVs NOVO, com o timestamp da execução no nome — nunca sobrescreve
# rodadas anteriores. O resumo traz uma linha por execução, só com métricas
# agregadas (schema fixo); o detalhe traz a comparação item a item (nó de
# tensão, ID de componente) em formato longo — ver avaliar_resposta_completa
# em main.py para o porquê dessa separação.
RESULTADOS_DIR = BASE_DIR / "resultados"
JSON_DIR = RESULTADOS_DIR / "respostas_json"
PREFIXO_CSV_RESULTADOS = "resultados"
PREFIXO_CSV_DETALHES = "resultados_detalhe"

# =========================================================
# Rodada oficial: os dados que sustentam os números do TFC, versionados para
# que a análise possa ser refeita sem chamar nenhuma API. Somente leitura —
# rodadas novas vão para RESULTADOS_DIR. A origem de cada arquivo está
# descrita no README.
# =========================================================
RODADA_OFICIAL_DIR = BASE_DIR / "rodada_oficial"
CSV_RODADA_OFICIAL = RODADA_OFICIAL_DIR / "resultados.csv"
JSON_RODADA_OFICIAL_DIR = RODADA_OFICIAL_DIR / "respostas_json"

# Saídas que vão para o texto do TFC (tabelas LaTeX e figuras). A pasta TFC/
# é um repositório separado; os scripts a criam se ela não existir.
TFC_DIR = BASE_DIR / "TFC"


def caminho_detalhe(caminho_resumo: Path) -> Path:
    """CSV de detalhe correspondente a um CSV de resumo, na mesma pasta
    ('resultados.csv' -> 'resultados_detalhe.csv';
    'resultados_AAAAMMDD_HHMMSS.csv' -> 'resultados_detalhe_AAAAMMDD_HHMMSS.csv')."""
    caminho_resumo = Path(caminho_resumo)
    return caminho_resumo.with_name(
        caminho_resumo.name.replace(PREFIXO_CSV_RESULTADOS, PREFIXO_CSV_DETALHES, 1))


# =========================================================
# Modelos avaliados
#
# Proprietário: Gemini 2.5 Flash, pela API do Google.
#
# Pesos abertos, pelo OpenRouter, escolhidos para variar um fator por vez:
#   - Qwen3-VL 32B x Qwen3-VL 8B: mesma família, geração e variante Instruct;
#     a diferença é o PORTE.
#   - Qwen3-VL 32B x Gemma 3 27B: portes próximos (32B x 27,4B), ambos densos;
#     a diferença é a FAMÍLIA.
#
# "provider" fixa o provedor (allow_fallbacks=False): sem isso o OpenRouter
# pode rotear para backends com precisões diferentes, e a rodada deixa de ser
# reprodutível. "rotulo" é o nome gravado no CSV. Preços em USD por 1M de
# tokens, do catálogo do provedor fixado, na época da rodada oficial.
# =========================================================
MODELO_GEMINI = "gemini-2.5-flash"
USAR_MODELO_GEMINI = True

# O mesmo Gemini 2.5 Flash com o raciocinio interno desligado
# (thinking_budget=0). Entra no CSV com rotulo proprio para ser tratado
# como um modelo a parte: assim a rodada pode ser consolidada com a
# oficial e comparada sem ambiguidade. Acionado por
#     python main.py --sem-raciocinio --somente-gemini
# Sem essa opcao, o Gemini roda no padrao dinamico da rodada oficial.
GEMINI_SEM_RACIOCINIO_BUDGET = 0
ROTULO_GEMINI_SEM_RACIOCINIO = f"{MODELO_GEMINI} (sem raciocinio)"

MODELOS_OPENROUTER = [
    {"id": "qwen/qwen3-vl-32b-instruct", "provider": "Alibaba",
     "rotulo": "qwen3-vl-32b (openrouter)",
     "preco_input_1m": 0.104, "preco_output_1m": 0.416},
    {"id": "qwen/qwen3-vl-8b-instruct", "provider": "Parasail",
     "rotulo": "qwen3-vl-8b-bf16 (openrouter)",
     "preco_input_1m": 0.25, "preco_output_1m": 0.75},
    # A Novita é o único provedor que serve o Gemma 3 27B em bf16, mesma
    # precisão do Qwen3-VL 8B, o que evita introduzir variação de precisão na
    # comparação entre famílias.
    {"id": "google/gemma-3-27b-it", "provider": "Novita",
     "rotulo": "gemma-3-27b-bf16 (openrouter)",
     "preco_input_1m": 0.10, "preco_output_1m": 0.20},
]
USAR_MODELO_OPENROUTER = True

# =========================================================
# Reexecução automática das respostas truncadas. Ao fim da rodada, main.py
# refaz as execuções cuja resposta foi cortada por tamanho / loop de
# repetição (marcadas como 'truncado_por_tamanho'). Com temperatura 0 a mesma
# chamada repetiria o mesmo loop, então a reexecução usa uma temperatura
# maior e uma semente diferente por tentativa. As linhas reexecutadas ficam
# marcadas na coluna 'reexecucoes' (>0) do CSV resumo, e a resposta truncada
# original é guardada como '<nome>_falha_original.json'. Na rodada oficial
# nenhuma resposta veio truncada; as 21 falhas (limite de taxa do provedor e
# JSON inválido) foram refeitas com refazer_falhas.py, em temperatura 0.
# =========================================================
REEXECUTAR_TRUNCADAS = True
MAX_REEXECUCOES = 2
TEMPERATURA_REEXECUCAO = 0.3

# =========================================================
# Idioma dos prompts enviados à IA — "pt", "en" ou "ambos" (roda cada
# circuito/nível de prompt uma vez em português e uma vez em inglês, no
# mesmo par de arquivos CSV, marcando cada linha com a coluna "idioma").
# =========================================================
IDIOMA_PROMPT = "ambos"  # "pt", "en" ou "ambos"

# =========================================================
# Avaliação — tolerância de engenharia usada para classificar uma
# comparação numérica (IA x gabarito) como 'correto'. Usada tanto em
# evaluator_numerico.py (tensões/correntes/potências/Vout/ganho) quanto em
# evaluator_componentes.py (valor dos componentes, ex: resistência).
# =========================================================
TOLERANCIA_PADRAO_PERCENTUAL = 5.0

# =========================================================
# Custos — preços do Gemini 2.5 Flash em USD por 1 milhão de tokens (os
# tokens de raciocínio interno são cobrados como saída) e cotação USD -> BRL,
# usada só para exibir o custo também em reais no relatório do terminal.
# =========================================================
PRECO_INPUT_1M = 0.30
PRECO_OUTPUT_1M = 2.50
COTACAO_USD_BRL = 5.50

# Preço (entrada, saída) em USD por 1M de tokens de cada modelo, pelo nome
# gravado no CSV. Usado para (re)calcular o custo a partir dos tokens.
PRECOS_POR_MODELO = {
    MODELO_GEMINI: (PRECO_INPUT_1M, PRECO_OUTPUT_1M),
    # Mesmo modelo e mesma tabela de precos; muda so o orcamento de
    # raciocinio, e os tokens pensados nao sao mais cobrados.
    ROTULO_GEMINI_SEM_RACIOCINIO: (PRECO_INPUT_1M, PRECO_OUTPUT_1M),
    **{cfg["rotulo"]: (cfg["preco_input_1m"], cfg["preco_output_1m"])
       for cfg in MODELOS_OPENROUTER},
}


def custo_usd(modelo: str, input_tokens: float, output_tokens: float) -> float:
    """Custo de uma execução em USD, a partir dos tokens e do preço do modelo."""
    preco_in, preco_out = PRECOS_POR_MODELO.get(modelo, (0.0, 0.0))
    return (input_tokens * preco_in + output_tokens * preco_out) / 1_000_000

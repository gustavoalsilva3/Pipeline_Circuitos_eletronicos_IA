from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

IMAGENS_DIR = BASE_DIR / "imagens"
GABARITOS_DIR = BASE_DIR / "gabaritos"
RESULTADOS_DIR = BASE_DIR / "resultados"
JSON_DIR = RESULTADOS_DIR / "respostas_json"

# Prefixos dos dois CSVs de saída. Cada chamada de main.py gera um par de
# arquivos NOVO, com o timestamp da execução no nome (ex:
# "resultados_20260912_153045.csv") — não acumula/sobrescreve entre
# rodadas, cada rodada fica com seu próprio par de arquivos.
# PREFIXO_CSV_RESULTADOS: resumo, uma linha por execução, só com métricas
# agregadas (schema fixo, não cresce com o dataset). PREFIXO_CSV_DETALHES:
# a comparação item a item (nó de tensão, ID de componente) em formato
# longo — uma linha por item em vez de uma coluna por item. Ver
# avaliar_resposta_completa em main.py para o porquê dessa separação.
PREFIXO_CSV_RESULTADOS = "resultados"
PREFIXO_CSV_DETALHES = "resultados_detalhe"

MODELO_GEMINI = "gemini-2.5-flash"
USAR_MODELO_GEMINI = True

# =========================================================
# Modelos servidos pelo LM Studio (servidor local, API compatível com a da
# OpenAI) — ver lmstudio_client.py. MODELOS_LMSTUDIO é uma LISTA: cada
# identificador vira um modelo separado na mesma rodada (útil pra comparar
# vários modelos de uma vez). O identificador é o que aparece na aba do
# servidor local do LM Studio (ex: "qwen/qwen3-vl-8b"); o LM Studio precisa
# ter cada modelo listado disponível/carregado. URL padrão
# http://localhost:1234; para outra máquina, LMSTUDIO_BASE_URL no .env.
# =========================================================
MODELOS_LMSTUDIO = ["qwen/qwen3-vl-8b"]
USAR_MODELO_LMSTUDIO = False

# =========================================================
# Modelos servidos pelo OpenRouter — ver openrouter_client.py. Servem a dois
# papéis no trabalho, ambos impossíveis na máquina local (8 GB de VRAM):
#
# 1. Qwen3-VL 32B: mesma família, geração e variante do modelo local de 8B,
#    com 4x os parâmetros. Isola o efeito do PORTE, separando-o do efeito de
#    ser aberto ou proprietário (a limitação declarada nas Conclusões).
#    Em Q4_K_M ocuparia ~19 GB, ou seja, uma GPU de consumo de 24 GB.
#
# 2. Qwen3-VL 8B em bf16: o MESMO modelo da rodada local, em precisão cheia.
#    Serve de controle para medir quanto a quantização de 4 bits custa, em
#    vez de apenas declará-la como limitação.
#
# "provider" fixa o provedor (allow_fallbacks=False): sem isso o OpenRouter
# pode rotear para backends com quantizações diferentes, e a rodada deixa de
# ser reprodutível. "rotulo" é o nome gravado no CSV — explicita a origem,
# para não confundir o 8B local com o 8B em nuvem na análise.
# Preços em USD por 1M de tokens, do catálogo do provedor fixado.
# =========================================================
MODELOS_OPENROUTER = [
    # Os dois Qwen ja foram executados na rodada oficial
    # (resultados_20260924_082751.csv). Ficam comentados para que a rodada
    # deste modelo novo nao repita 288 chamadas ja feitas; para reproduzir a
    # rodada completa do zero, basta descomentar os dois.
    #
    {"id": "qwen/qwen3-vl-32b-instruct",  "provider": "Alibaba",
     "rotulo": "qwen3-vl-32b (openrouter)",
     "preco_input_1m": 0.104, "preco_output_1m": 0.416},
    #
    {"id": "qwen/qwen3-vl-8b-instruct",   "provider": "Parasail",
     "rotulo": "qwen3-vl-8b-bf16 (openrouter)",
     "preco_input_1m": 0.25,  "preco_output_1m": 0.75},

    # Gemma 3 27B: segunda FAMILIA aberta, de porte proximo ao Qwen 32B
    # (27,4B contra 32B, ambos densos). Serve para verificar se as falhas
    # observadas nos Qwen — em especial a nao aplicacao da condicao de
    # circuito aberto do capacitor em regime CC — sao gerais aos modelos
    # abertos ou especificas daquela familia. A Novita e o unico provedor que
    # o serve em bf16, mesma precisao do 8B, o que evita introduzir variacao
    # de quantizacao na comparacao entre familias.
    {
        "id": "google/gemma-3-27b-it",
        "provider": "Novita",
        "rotulo": "gemma-3-27b-bf16 (openrouter)",
        "preco_input_1m": 0.10,
        "preco_output_1m": 0.20,
    },
]
USAR_MODELO_OPENROUTER = True

# Limite de tokens de SAÍDA por chamada ao LM Studio. Sem limite, um modelo
# pequeno pode entrar em loop de repetição até encher o contexto sem nunca
# fechar o JSON — visto em teste. Com limite, uma resposta em loop é cortada
# cedo e registrada como truncada (e reexecutada ao fim da rodada). O JSON
# completo tem ~400-700 tokens; 3000 dá folga. (A ~16 tokens/s, 3000 tokens
# levam ~3 min, abaixo do timeout padrão de 600 s.)
MAX_TOKENS_SAIDA_LOCAL = 3000

# Exceções por modelo (identificador do LM Studio -> valor), para um modelo
# que precise de limite de saída / timeout diferente do padrão (acima e
# 600 s). O limite precisa caber no contexto carregado no LM Studio menos a
# entrada (~2100 tokens), e o timeout precisa cobrir o tempo de gerá-lo.
MAX_TOKENS_SAIDA_POR_MODELO = {}
TIMEOUT_POR_MODELO_S = {}

# =========================================================
# Reexecução automática das respostas truncadas. Ao fim da rodada, main.py
# refaz as execuções cuja resposta foi cortada por tamanho / loop de
# repetição (marcadas como 'truncado_por_tamanho'). Com temperatura 0 a mesma
# chamada repetiria o mesmo loop, então a reexecução usa uma temperatura
# maior e uma semente diferente por tentativa. As linhas reexecutadas ficam
# marcadas na coluna 'reexecucoes' (>0) do CSV resumo, e a resposta truncada
# original é guardada como '<nome>_falha_original.json'.
# =========================================================
REEXECUTAR_TRUNCADAS = True
MAX_REEXECUCOES = 2
TEMPERATURA_REEXECUCAO = 0.3

# =========================================================
# Idioma dos prompts enviados à IA — "pt", "en" ou "ambos" (roda cada
# circuito/nível de prompt uma vez em português e uma vez em inglês, no
# mesmo par de arquivos CSV, marcando cada linha com a coluna "idioma").
# Não filtra quais imagens são processadas (isso é feito manualmente
# movendo arquivos para fora de IMAGENS_DIR, como já é feito num teste com
# poucos circuitos).
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
# Custos — preços do Gemini 2.5 Flash em USD por 1 milhão de tokens
# e cotação USD -> BRL usada só para exibir o custo estimado também
# em reais no relatório final. Ajuste aqui se os preços da API
# mudarem ou se quiser atualizar a cotação. Usadas em main.py.
# =========================================================
PRECO_INPUT_1M = 0.30
PRECO_OUTPUT_1M = 2.50
COTACAO_USD_BRL = 5.50
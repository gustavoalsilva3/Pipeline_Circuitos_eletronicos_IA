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
"""Módulo de avaliação de componentes identificados pela IA em um circuito.

Escopo reduzido de propósito, seguindo a mesma lógica aplicada em
evaluator_textual.py: avalia apenas o que tem valor de engenharia direto.

  - Identificação do componente (por 'id'): taxa de acerto simples
    (identificados corretamente / total esperado no gabarito), mais as
    listas de componentes "alucinados" (citados mas inexistentes) e "não
    identificados" (existem no circuito, IA não citou).
  - Atributo 'valor' (ex: "10 kΩ"): comparado numericamente, reaproveitando
    o parser de unidades de engenharia e a tolerância de 5% definidos em
    evaluator_numerico.py.

Duas simplificações em relação à versão anterior:
  - Sem comparação de 'modelo' (ex: "LT1057" vs "TL081"): o modelo
    específico de op-amp usado na simulação é praticamente arbitrário e não
    testa entendimento de engenharia do circuito — mesmo racional que
    tirou 'funcao'/'explicacao' do evaluator_textual.py.
  - Sem precision/recall/F1: substituídos por uma taxa de acerto única
    (mais simples de reportar e interpretar), com alucinados e não
    identificados reportados separadamente em vez de compostos numa métrica.
  - Categoria 'transistores' removida: fora do escopo do TCC (delimitação
    explícita — "transistores BJT e MOSFET não serão incluídos").
"""
print("EVALUATOR_COMPONENTES_V2_CARREGADO")

from evaluator_numerico import classificar_status


CATEGORIAS_PADRAO = [
    "amplificadores_operacionais",
    "resistores",
    "fontes",
    "capacitores",
    "diodos",
]

# Único atributo avaliado numericamente. 'modelo' e qualquer outro atributo
# descritivo não entram mais na pontuação (ver docstring do módulo).
ATRIBUTO_AVALIADO = "valor"

# Categorias em que só a presença/identificação do componente importa —
# nenhum atributo é comparado, mesmo que 'valor' esteja presente no JSON.
# 'diodos' entrou aqui porque o TCC não tem um valor de referência de
# engenharia estabelecido para diodo (ao contrário de resistor/fonte, onde
# o valor numérico é o dado central da análise).
CATEGORIAS_APENAS_PRESENCA = {"diodos"}


def _normalizar_id(id_bruto):
    """Normaliza o identificador do componente para comparação (ex: ' r1 ' -> 'R1')."""
    if id_bruto is None:
        return ""
    return str(id_bruto).strip().upper()


def _indexar_por_id(lista_componentes):
    """Converte uma lista [{'id': 'R1', ...}, ...] em dict {'R1': {...}},
    ignorando entradas sem 'id' (evita que uma resposta malformada quebre a
    avaliação inteira do circuito)."""
    indice = {}
    for componente in lista_componentes or []:
        if not isinstance(componente, dict):
            continue
        id_norm = _normalizar_id(componente.get("id"))
        if id_norm:
            indice[id_norm] = componente
    return indice


def avaliar_categoria_componentes(lista_ia, lista_real, avaliar_atributo=True):
    """Avalia uma única categoria (ex: todos os resistores do circuito).

    Retorna a taxa de acerto na identificação por 'id', mais a comparação
    do atributo 'valor' para os componentes corretamente identificados
    (quando avaliar_atributo=True; para categorias em CATEGORIAS_APENAS_PRESENCA,
    só a identificação é avaliada).
    """

    indice_ia = _indexar_por_id(lista_ia)
    indice_real = _indexar_por_id(lista_real)

    ids_ia = set(indice_ia.keys())
    ids_real = set(indice_real.keys())

    verdadeiros_positivos = ids_ia & ids_real
    falsos_positivos = ids_ia - ids_real   # alucinados: citados mas inexistentes
    falsos_negativos = ids_real - ids_ia   # existem no circuito, IA não citou

    # Taxa de acerto = quantos componentes reais o modelo identificou.
    # Sem componentes reais nesta categoria, não há o que acertar (None).
    taxa_acerto = round(len(verdadeiros_positivos) / len(ids_real), 3) if ids_real else None

    atributos = {}
    if avaliar_atributo:
        for id_componente in sorted(verdadeiros_positivos):
            valor_ia = indice_ia[id_componente].get(ATRIBUTO_AVALIADO)
            valor_real = indice_real[id_componente].get(ATRIBUTO_AVALIADO)

            if valor_ia is None and valor_real is None:
                continue

            atributos[id_componente] = {
                "esperado": valor_real,
                "obtido": valor_ia,
                "status": classificar_status(valor_real, valor_ia),
            }

    return {
        "taxa_acerto": taxa_acerto,
        "alucinados": sorted(falsos_positivos),
        "nao_identificados": sorted(falsos_negativos),
        "atributos": atributos,
    }


def avaliar_componentes(resposta: dict, gabarito: dict):
    """Avalia todas as categorias de componentes de um circuito.

    Recebe os dicionários completos de resposta e gabarito (mesmo padrão de
    avaliar_numerico em evaluator_numerico.py) e extrai a seção
    'componentes' internamente. Mantida com esse nome para integração direta
    com main.py, ao lado de avaliar_numerico e avaliar_textual.
    """

    componentes_ia = resposta.get("componentes", {}) or {}
    componentes_real = gabarito.get("componentes", {}) or {}

    categorias = sorted(
        set(CATEGORIAS_PADRAO) | set(componentes_ia.keys()) | set(componentes_real.keys())
    )

    resultado = {}
    for categoria in categorias:
        lista_ia = componentes_ia.get(categoria, [])
        lista_real = componentes_real.get(categoria, [])

        if not lista_ia and not lista_real:
            continue  # categoria vazia dos dois lados: fora do escopo deste circuito

        resultado[categoria] = avaliar_categoria_componentes(
            lista_ia, lista_real,
            avaliar_atributo=categoria not in CATEGORIAS_APENAS_PRESENCA,
        )

    return resultado


def agregar_componentes(lista_resultados):
    """Agrega a taxa de acerto de identificação por categoria de componente
    ao longo de vários circuitos, pronto para virar gráfico de barras."""

    por_categoria = {}

    for resultado_circuito in lista_resultados:
        for categoria, dados in resultado_circuito.items():
            bucket = por_categoria.setdefault(categoria, {
                "taxas": [], "alucinados": 0, "nao_identificados": 0,
            })

            if dados["taxa_acerto"] is not None:
                bucket["taxas"].append(dados["taxa_acerto"])

            bucket["alucinados"] += len(dados["alucinados"])
            bucket["nao_identificados"] += len(dados["nao_identificados"])

    return {
        categoria: {
            "taxa_acerto_media": round(sum(bucket["taxas"]) / len(bucket["taxas"]), 3) if bucket["taxas"] else None,
            "total_alucinados": bucket["alucinados"],
            "total_nao_identificados": bucket["nao_identificados"],
        }
        for categoria, bucket in por_categoria.items()
    }
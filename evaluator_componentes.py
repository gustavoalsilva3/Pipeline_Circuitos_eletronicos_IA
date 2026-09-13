"""
Módulo de avaliação da identificação de componentes pela IA.

A avaliação é dividida em duas partes:

1. Identificação dos componentes
   - Precision
   - Recall
   - F1-score

2. Valores dos componentes
   - Para cada componente corretamente identificado (TP), compara o valor
     informado pela IA com o valor presente no gabarito, com tolerância
     percentual de engenharia (mesmo parser numérico do evaluator_numerico,
     não igualdade exata de string).
   - Além da comparação detalhada por ID (guardada em 'valores'), o
     resultado inclui 'valor_taxa_acerto_percentual': dos componentes
     identificados corretamente, quantos também têm o valor certo. É uma
     pergunta diferente de precision/recall, que só olham identificação.

A identificação dos componentes é feita pelo campo 'id'. Só as categorias
em CATEGORIAS_PADRAO são avaliadas com precision/recall/f1; qualquer outra
chave que apareça em 'componentes' (alucinada pela IA ou um nome diferente
no gabarito) é listada em 'categorias_invalidas' em vez de virar uma
categoria nova.

Exemplo:

Gabarito:
    R1, R2, V1

IA:
    R1, R2, R3

Temos:
    TP = R1, R2
    FP = R3
    FN = V1

Logo:
    Precision = 2 / 3
    Recall    = 2 / 3
    F1        = 2 / 3
"""

from config import TOLERANCIA_PADRAO_PERCENTUAL
from evaluator_numerico import parse_numeric_value

# Categorias consideradas no dataset.
CATEGORIAS_PADRAO = [
    "amplificadores_operacionais",
    "resistores",
    "fontes",
    "capacitores",
    "diodos",
    "indutores"
]


# ---------------------------------------------------------------------------
# FUNÇÕES AUXILIARES
# ---------------------------------------------------------------------------

def _normalizar_id(id_bruto):
    """
    Normaliza o ID do componente.

    Exemplos:
        'r1'   -> 'R1'
        ' R1 ' -> 'R1'
        'V1'   -> 'V1'

    Isso evita que diferenças de maiúsculas/minúsculas ou espaços
    sejam consideradas como erro.
    """

    if id_bruto is None:
        return ""

    return str(id_bruto).strip().upper()


def _indexar_por_id(lista_componentes):
    """
    Converte uma lista de componentes em um dicionário indexado pelo ID.

    Entrada:
        [
            {"id": "R1", "valor": "10 kΩ"},
            {"id": "R2", "valor": "5 kΩ"}
        ]

    Saída:
        {
            "R1": {"id": "R1", "valor": "10 kΩ"},
            "R2": {"id": "R2", "valor": "5 kΩ"}
        }

    Componentes sem ID são ignorados.
    """

    indice = {}

    for componente in lista_componentes or []:

        # Ignora entradas que não sejam objetos/dicionários.
        if not isinstance(componente, dict):
            continue

        id_norm = _normalizar_id(componente.get("id"))

        # Ignora componentes sem ID válido.
        if not id_norm:
            continue

        indice[id_norm] = componente

    return indice


def _comparar_valor(valor_ia, valor_real, tolerancia_percentual=TOLERANCIA_PADRAO_PERCENTUAL):
    """
    Compara o valor informado pela IA com o valor do gabarito usando o
    mesmo parser tolerante a prefixos de engenharia do evaluator_numerico
    (ex: "10 V" == "+10 V", "10 kΩ" == "10000 Ω"), em vez de igualdade
    exata de string — que classificava como erro respostas corretas só
    por causa de formatação diferente (sinal, espaço, unidade).

        - valores numericamente iguais (dentro da tolerância) -> True
        - valores numericamente diferentes -> False
        - algum valor ausente -> None
        - nenhum dos dois é numérico -> cai para comparação de string
          (preserva a checagem para eventuais campos textuais)
    """

    if valor_ia is None or valor_real is None:
        return None

    valor_ia_n = parse_numeric_value(valor_ia)
    valor_real_n = parse_numeric_value(valor_real)

    if valor_ia_n is None or valor_real_n is None:
        return str(valor_ia).strip().lower() == str(valor_real).strip().lower()

    if abs(valor_real_n) < 1e-9:
        return abs(valor_ia_n) < 1e-9

    erro_percentual = abs(valor_ia_n - valor_real_n) / abs(valor_real_n) * 100
    return erro_percentual <= tolerancia_percentual


# ---------------------------------------------------------------------------
# AVALIAÇÃO DE UMA CATEGORIA
# ---------------------------------------------------------------------------

def avaliar_categoria_componentes(lista_ia, lista_real):
    """
    Avalia uma categoria específica de componentes.

    Exemplo:
        categoria = resistores

    A função calcula:

        TP -> componente existe no gabarito e foi identificado pela IA
        FP -> IA identificou um componente que não existe
        FN -> componente existe, mas não foi identificado

    A partir desses valores são calculados:

        Precision
        Recall
        F1-score

    Também registra a comparação dos valores dos componentes
    corretamente identificados.
    """

    # Indexa os componentes pelo ID.
    indice_ia = _indexar_por_id(lista_ia)
    indice_real = _indexar_por_id(lista_real)

    # Conjuntos de IDs.
    ids_ia = set(indice_ia.keys())
    ids_real = set(indice_real.keys())

    # -----------------------------------------------------------------------
    # IDENTIFICAÇÃO
    # -----------------------------------------------------------------------

    # Verdadeiros positivos:
    # componentes que existem no gabarito e foram identificados pela IA.
    verdadeiros_positivos = ids_ia & ids_real

    # Falsos positivos:
    # componentes citados pela IA que não existem no gabarito.
    falsos_positivos = ids_ia - ids_real

    # Falsos negativos:
    # componentes que existem no gabarito, mas não foram identificados.
    falsos_negativos = ids_real - ids_ia

    TP = len(verdadeiros_positivos)
    FP = len(falsos_positivos)
    FN = len(falsos_negativos)

    # -----------------------------------------------------------------------
    # PRECISION
    # -----------------------------------------------------------------------

    # Precision responde:
    #
    # "Dos componentes identificados pela IA, quantos realmente existem?"
    #
    # Se a IA não identificou nenhum componente, a métrica não é definida.
    if TP + FP > 0:
        precision = TP / (TP + FP)
    else:
        precision = None

    # -----------------------------------------------------------------------
    # RECALL
    # -----------------------------------------------------------------------

    # Recall responde:
    #
    # "Dos componentes que realmente existem, quantos a IA identificou?"
    #
    # Se o gabarito não possui componentes naquela categoria,
    # não existe o que recuperar.
    if TP + FN > 0:
        recall = TP / (TP + FN)
    else:
        recall = None

    # -----------------------------------------------------------------------
    # F1-SCORE
    # -----------------------------------------------------------------------

    # F1 combina Precision e Recall.
    if precision is not None and recall is not None:

        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

    else:
        f1 = None

    # -----------------------------------------------------------------------
    # COMPARAÇÃO DOS VALORES
    # -----------------------------------------------------------------------

    valores = {}

    # O valor só é comparado quando o componente foi corretamente
    # identificado pela IA.
    for id_componente in sorted(verdadeiros_positivos):

        componente_ia = indice_ia[id_componente]
        componente_real = indice_real[id_componente]

        valor_ia = componente_ia.get("valor")
        valor_real = componente_real.get("valor")

        valores[id_componente] = {
            "valor_esperado": valor_real,
            "valor_obtido": valor_ia,
            "valor_correto": _comparar_valor(
                valor_ia,
                valor_real
            )
        }

    # -----------------------------------------------------------------------
    # AGREGADO DA COMPARAÇÃO DE VALORES
    #
    # Mede uma pergunta diferente de precision/recall: "dos componentes que
    # a IA identificou corretamente (TP), ela também acertou o valor?".
    # Por isso o denominador é TP, não o total esperado no gabarito — um
    # componente não identificado já é penalizado no recall, não deve ser
    # penalizado de novo aqui. 'faltante' (valor_correto is None, quando
    # falta o campo 'valor' de um dos lados) entra no denominador para não
    # inflar a taxa artificialmente, no mesmo espírito do
    # 'taxa_acerto_percentual' do evaluator_numerico.
    # -----------------------------------------------------------------------

    valor_status_counts = {"correto": 0, "incorreto": 0, "faltante": 0}
    for info in valores.values():
        if info["valor_correto"] is None:
            valor_status_counts["faltante"] += 1
        elif info["valor_correto"]:
            valor_status_counts["correto"] += 1
        else:
            valor_status_counts["incorreto"] += 1

    # Se todos os itens caíram em 'faltante' (categoria sem campo 'valor'
    # aplicável, ex: amplificadores_operacionais, que só tem 'id'), não há
    # nada avaliável — retorna None em vez de 0.0, que sugeriria erro real.
    quantidade_avaliavel = valor_status_counts["correto"] + valor_status_counts["incorreto"]
    valor_taxa_acerto_percentual = (
        round(valor_status_counts["correto"] / TP * 100, 1)
        if TP > 0 and quantidade_avaliavel > 0
        else None
    )

    # -----------------------------------------------------------------------
    # RESULTADO
    # -----------------------------------------------------------------------

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,

        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1_score": round(f1, 4) if f1 is not None else None,

        "valor_taxa_acerto_percentual": valor_taxa_acerto_percentual,
        "valor_status_counts": valor_status_counts,

        # Informação diagnóstica.
        "componentes_identificados": sorted(verdadeiros_positivos),
        "componentes_alucinados": sorted(falsos_positivos),
        "componentes_nao_identificados": sorted(falsos_negativos),

        # Comparação dos valores.
        "valores": valores
    }


# ---------------------------------------------------------------------------
# AVALIAÇÃO COMPLETA DO CIRCUITO
# ---------------------------------------------------------------------------

def avaliar_componentes(resposta: dict, gabarito: dict):
    """
    Avalia os componentes de um circuito completo.

    A função percorre todas as categorias existentes na resposta e no
    gabarito e calcula Precision, Recall e F1-score para cada categoria.

    Também realiza a comparação dos valores dos componentes corretamente
    identificados.
    """

    # Obtém a seção de componentes da IA.
    componentes_ia = resposta.get("componentes", {}) or {}

    # Obtém a seção de componentes do gabarito.
    componentes_real = gabarito.get("componentes", {}) or {}

    # Lista de categorias FECHADA (só CATEGORIAS_PADRAO) — se a IA alucinar
    # uma chave fora do padrão (erro de digitação, categoria em inglês,
    # etc.), ela não vira uma coluna nova e imprevisível no CSV. As chaves
    # extras ficam registradas em 'categorias_invalidas' para diagnóstico,
    # sem receber o mesmo tratamento (precision/recall/f1) das válidas.
    categorias_invalidas = sorted(
        (set(componentes_ia.keys()) | set(componentes_real.keys()))
        - set(CATEGORIAS_PADRAO)
    )

    resultado = {}
    if categorias_invalidas:
        resultado["categorias_invalidas"] = categorias_invalidas

    # Avalia cada categoria separadamente.
    for categoria in CATEGORIAS_PADRAO:

        lista_ia = componentes_ia.get(categoria, []) or []
        lista_real = componentes_real.get(categoria, []) or []

        # Se nenhum dos dois possui componentes nessa categoria,
        # não há nada para avaliar.
        if not lista_ia and not lista_real:
            continue

        resultado[categoria] = avaliar_categoria_componentes(
            lista_ia,
            lista_real
        )

    return resultado
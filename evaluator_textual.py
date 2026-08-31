"""Módulo de avaliação textual/qualitativa das respostas da IA.

Escopo reduzido de propósito: avalia apenas 'regiao_operacao' e
'terra_virtual', os dois únicos campos de 'analise_textual' com valor de
referência objetivo (confirmável contra o gabarito/LTspice sem ambiguidade
semântica). Campos como 'funcao', 'explicacao', 'justificativa_regiao_operacao'
e 'topologia' foram deixados de fora — a complexidade de avaliar texto livre
(correspondência lexical imprecisa, ex: "amplificador" vs "amplifica" não
bater em um matching exato) não compensa para o escopo deste TCC, que é
focado em engenharia elétrica, não em NLP.
"""

CAMPOS_AVALIADOS = ("regiao_operacao", "terra_virtual")

def _normalizar_texto(texto):
    """Normaliza texto para comparação tolerante a espaços/caixa."""
    if texto is None:
        return ""
    return " ".join(str(texto).strip().lower().split())


def avaliar_campo_categorico(valor_real, valor_ia):
    """Hit rate exato para um campo categórico ou booleano isolado."""
    if isinstance(valor_real, bool) or isinstance(valor_ia, bool):
        acerto = valor_real == valor_ia
    else:
        acerto = _normalizar_texto(valor_real) == _normalizar_texto(valor_ia)

    return {"esperado": valor_real, "obtido": valor_ia, "acerto": acerto}


def avaliar_textual(textual_ia: dict, textual_real: dict):
    """Avalia 'regiao_operacao' e 'terra_virtual'.

    Recebe os dicionários já filtrados por response_splitter.separar_resposta
    (chave 'textual'), no mesmo padrão usado por avaliar_numerico. Mantida
    com esse nome para integração direta com evaluator_numerico.avaliar_resposta.
    """
    textual_ia = textual_ia or {}
    textual_real = textual_real or {}

    resultado = {}
    for campo in CAMPOS_AVALIADOS:
        if campo in textual_real or campo in textual_ia:
            resultado[campo] = avaliar_campo_categorico(textual_real.get(campo), textual_ia.get(campo))

    return resultado


def agregar_textual(lista_resultados):
    """Agrega a taxa de acerto de 'regiao_operacao' e 'terra_virtual' ao
    longo de vários circuitos — pronto para virar gráfico de barras."""

    acumulador = {}
    for resultado in lista_resultados:
        for campo, dados in resultado.items():
            acumulador.setdefault(campo, []).append(dados["acerto"])

    return {
        campo: round(sum(acertos) / len(acertos), 3)
        for campo, acertos in acumulador.items()
    }
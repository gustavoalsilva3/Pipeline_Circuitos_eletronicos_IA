"""Módulo de avaliação textual/qualitativa das respostas da IA.

Escopo reduzido de propósito: avalia apenas 'regiao_operacao' e
'terra_virtual', os dois únicos campos de 'analise_textual' com valor de
referência objetivo (confirmável contra o gabarito/LTspice sem ambiguidade
semântica). Campos como 'funcao', 'explicacao', 'justificativa_regiao_operacao'
e 'topologia' foram deixados de fora — a complexidade de avaliar texto livre
(correspondência lexical imprecisa, ex: "amplificador" vs "amplifica" não
bater em um matching exato) não compensa para o escopo deste TCC, que é
focado em engenharia elétrica, não em NLP.

'regiao_operacao' é um dict {ID_do_componente: estado}, não uma frase única
pro circuito inteiro — um circuito com vários diodos/amp-ops pode ter cada
um num estado diferente ao mesmo tempo (ex: D1 em condução, D2 em corte),
que uma frase só não descreve de forma comparável. Cada componente é
comparado por ESTADO CANÔNICO, não string exata (ver _canonicalizar_estado),
porque o mesmo estado elétrico é descrito de formas muito diferentes pela IA
(ex: "polarização direta (ON)" vs gabarito "Condução Direta") — com
comparação de string exata, esses casos contavam como erro mesmo estando
corretos. 'terra_virtual' continua um campo único pro circuito (é uma
propriedade da malha de realimentação, não de um componente isolado).
"""

import re

CAMPOS_AVALIADOS = ("regiao_operacao", "terra_virtual")


def _normalizar_texto(texto):
    """Normaliza texto para comparação tolerante a espaços/caixa."""
    if texto is None:
        return ""
    return " ".join(str(texto).strip().lower().split())


# =========================================================
# Canonicalização de 'regiao_operacao' em estados discretos
#
# Checados em ORDEM DE PRIORIDADE: assim que um padrão bate, esse é o
# estado do texto (não continua checando os demais). Isso resolve o caso
# de um texto mencionar mais de um termo (ex: "D1 em condução, D2 em
# corte" — o circuito como um todo é caracterizado pelo primeiro estado
# não-linear relevante encontrado, condução).
#
# Cuidado ao adicionar palavras-chave aqui: termos genéricos demais (ex:
# "aberto", "curto") são ambíguos entre diodo (corte/condução) e RLC
# (capacitor aberto, indutor em curto) — por isso os padrões abaixo usam
# frases mais específicas em vez de palavras soltas nesses casos.
# =========================================================
ESTADOS_CANONICOS = [
    ("saturacao_negativa", [
        r"satura[cç][aã]o\s*negativ", r"negative\s*saturation", r"-v(ee|cc)\b",
    ]),
    ("saturacao_positiva", [
        r"satura[cç][aã]o\s*positiv", r"positive\s*saturation", r"\+v(ee|cc)\b",
    ]),
    ("conducao", [
        r"condu[cç][aã]o", r"conduzindo", r"polariza[cç][aã]o\s*direta",
        r"polarizad[oa]\s*diretamente", r"\bon\b", r"\bforward\b",
        r"\bconduct(ing|ion)\b", r"forward[\s-]?bias",
    ]),
    ("corte", [
        r"\bcorte\b", r"bloquead", r"bloqueio", r"polariza[cç][aã]o\s*revers",
        r"polarizad[oa]\s*inversamente", r"\boff\b", r"\breverse\b",
        r"\bcut[\s-]?off\b", r"reverse[\s-]?bias",
    ]),
    ("linear", [
        r"\blinear\b", r"ponto\s*de\s*opera[cç][aã]o", r"dc\s*operating\s*point",
        r"capacitor.{0,15}abert", r"indutor.{0,15}curto", r"regime\s*permanente",
        r"capacitor.{0,15}open", r"inductor.{0,15}short", r"steady[\s-]?state",
    ]),
]


def _canonicalizar_estado(texto):
    """Retorna o rótulo canônico (ex: 'saturacao_negativa') do primeiro
    estado cujo padrão bate no texto, ou None se nenhum padrão bater."""
    texto_norm = _normalizar_texto(texto)
    for rotulo, padroes in ESTADOS_CANONICOS:
        for padrao in padroes:
            if re.search(padrao, texto_norm):
                return rotulo
    return None


def avaliar_campo_categorico(valor_real, valor_ia):
    """Hit rate exato para um campo categórico ou booleano isolado
    (usado para 'terra_virtual', que é booleano e não tem ambiguidade
    de fraseado)."""
    if isinstance(valor_real, bool) or isinstance(valor_ia, bool):
        acerto = valor_real == valor_ia
    else:
        acerto = _normalizar_texto(valor_real) == _normalizar_texto(valor_ia)

    return {"esperado": valor_real, "obtido": valor_ia, "acerto": acerto}


def _normalizar_id(id_bruto):
    """Normaliza o ID de um componente (maiúsculas, sem espaços nas pontas)
    — mesma lógica usada em evaluator_componentes._normalizar_id, para que
    'd1'/' D1 ' sejam reconhecidos como o mesmo componente 'D1'."""
    if id_bruto is None:
        return ""
    return str(id_bruto).strip().upper()


def avaliar_regiao_operacao_por_componente(dict_real, dict_ia):
    """Avalia 'regiao_operacao' componente a componente (diodos, amp-ops).

    Recebe {ID: texto_do_estado} dos dois lados, normaliza os IDs e compara
    o estado canônico de cada um (ver _canonicalizar_estado). Segue o mesmo
    vocabulário de status usado em evaluator_componentes:
      - 'faltante': ID existe no gabarito mas a IA não opinou sobre ele
      - 'alucinado': ID que a IA mencionou mas não existe no gabarito
      - 'correto'/'incorreto': ID presente nos dois lados, estado bate ou não

    Se o texto do gabarito não bater em nenhum padrão canônico conhecido
    (caso raro, sinaliza que ESTADOS_CANONICOS precisa de um padrão novo),
    cai para comparação de string exata nesse item, em vez de marcar como
    erro sem visibilidade.
    """
    dict_real = dict_real if isinstance(dict_real, dict) else {}
    dict_ia = dict_ia if isinstance(dict_ia, dict) else {}

    real_norm = {_normalizar_id(k): v for k, v in dict_real.items()}
    ia_norm = {_normalizar_id(k): v for k, v in dict_ia.items()}

    ids = sorted(set(real_norm.keys()) | set(ia_norm.keys()))

    valores = {}
    status_counts = {"correto": 0, "incorreto": 0, "faltante": 0, "alucinado": 0}

    for id_comp in ids:
        esperado = real_norm.get(id_comp)
        obtido = ia_norm.get(id_comp)
        estado_esperado = _canonicalizar_estado(esperado) if esperado is not None else None
        estado_obtido = _canonicalizar_estado(obtido) if obtido is not None else None

        if esperado is None:
            status = "alucinado"
        elif obtido is None:
            status = "faltante"
        elif estado_esperado is None:
            status = "correto" if _normalizar_texto(esperado) == _normalizar_texto(obtido) else "incorreto"
        else:
            status = "correto" if estado_esperado == estado_obtido else "incorreto"

        status_counts[status] += 1
        valores[id_comp] = {
            "esperado": esperado,
            "obtido": obtido,
            "estado_esperado": estado_esperado,
            "estado_obtido": estado_obtido,
            "status": status,
        }

    quantidade = len(ids)
    taxa_acerto_percentual = (
        round(status_counts["correto"] / quantidade * 100, 1) if quantidade > 0 else None
    )

    return {
        "quantidade": quantidade,
        "taxa_acerto_percentual": taxa_acerto_percentual,
        "status_counts": status_counts,
        "valores": valores,
    }


def avaliar_textual(textual_ia: dict, textual_real: dict):
    """Avalia 'regiao_operacao' (por componente) e 'terra_virtual' (único).

    Recebe os dicionários já filtrados por response_splitter.separar_resposta
    (chave 'textual'), no mesmo padrão usado por avaliar_numerico. Mantida
    com esse nome para integração direta com evaluator_numerico.avaliar_resposta.
    """
    textual_ia = textual_ia or {}
    textual_real = textual_real or {}

    resultado = {}
    for campo in CAMPOS_AVALIADOS:
        if campo in textual_real or campo in textual_ia:
            if campo == "regiao_operacao":
                resultado[campo] = avaliar_regiao_operacao_por_componente(textual_real.get(campo), textual_ia.get(campo))
            else:
                resultado[campo] = avaliar_campo_categorico(textual_real.get(campo), textual_ia.get(campo))

    return resultado
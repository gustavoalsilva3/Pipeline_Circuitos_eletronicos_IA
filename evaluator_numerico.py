"""Módulo de avaliação numérica para comparar respostas da IA com gabaritos.

Converte valores para um formato numérico comum, classifica cada comparação
como correta, incorreta ou faltante (usando tolerância percentual de
engenharia, não igualdade exata), calcula métricas de erro (ERM, MAE, RMSE)
e organiza o resultado em seções para Vout, ganho, tensões, correntes e
potências.
"""

import re
import math

from response_splitter import separar_resposta
from evaluator_textual import avaliar_textual


NUMERIC_PATTERN = re.compile(
    r"[-+]?[0-9]*[\.,]?[0-9]+(?:[eE][-+]?[0-9]+)?"
)

# Prefixos de engenharia (SI) reconhecidos logo após o número, ex: "10 kΩ",
# "2.2 µF", "100 nF", "-12 V". Case-sensitive de propósito: 'm' (mili,
# 1e-3) e 'M' (mega, 1e6) são multiplicadores muito diferentes, assim como
# 'k'/'K' (kilo). Isso permite avaliar valores de componentes (resistores,
# capacitores, fontes) sem exigir que o JSON já venha em unidade SI base.
ENGINEERING_PREFIXES = {
    "p": 1e-12,
    "n": 1e-9,
    "µ": 1e-6,
    "u": 1e-6,
    "m": 1e-3,
    "k": 1e3,
    "K": 1e3,
    "M": 1e6,
    "G": 1e9,
}

# Tolerância padrão para classificar uma resposta como "correta".
# Um resultado de IA raramente bate com o gabarito na precisão de ponto
# flutuante — o critério de engenharia é estar dentro de uma faixa de erro
# relativo aceitável.
TOLERANCIA_PADRAO_PERCENTUAL = 5.0

# Tolerância absoluta usada apenas para decidir se um valor "é zero"
# (evita comparação exata de float quando o gabarito é 0).
ZERO_ABS_TOL = 1e-9


# =========================================================
# Converte valores numéricos de strings em float
# =========================================================
def parse_numeric_value(valor):
    """Converte inteiros, floats e strings numéricas para float quando possível.

    Também reconhece prefixos de engenharia colados à unidade (ex: "10 kΩ"
    -> 10000.0, "2.2 µF" -> 2.2e-6, "-12 V" -> -12.0). Sem prefixo
    reconhecido, o valor volta como o número puro (comportamento anterior
    preservado)."""

    if valor is None or isinstance(valor, bool):
        return None

    if isinstance(valor, (int, float)):
        return float(valor) if math.isfinite(float(valor)) else None

    if not isinstance(valor, str):
        return None

    texto_original = valor.strip()
    texto_lower = texto_original.lower()

    if texto_lower in {
        "não determinado", "nao determinado", "nao_determinado",
        "não_determinado", "null", "n/a", "nan", "",
    }:
        return None

    # Preserva maiúsculas/minúsculas (necessário para distinguir 'm' de 'M',
    # 'k' de 'K'); só a checagem de "não determinado" acima usa lowercase.
    texto = texto_original.replace("ª", "").replace("º", "").replace(",", ".")
    match = NUMERIC_PATTERN.search(texto)

    if not match:
        return None

    try:
        valor_numerico = float(match.group(0))
    except ValueError:
        return None

    if not math.isfinite(valor_numerico):
        return None

    restante = texto[match.end():].strip()
    if restante:
        multiplicador = ENGINEERING_PREFIXES.get(restante[0])
        if multiplicador is not None:
            valor_numerico *= multiplicador

    return valor_numerico


def _round_if_not_none(valor, casas_decimais):
    """Arredonda valores apenas quando eles existem, preservando None para dados ausentes."""
    return None if valor is None else round(valor, casas_decimais)


# =========================================================
# Comparação central de um único par (IA x referência)
#
# Bug corrigido: a versão anterior usava math.isclose(rel_tol=1e-9,
# abs_tol=1e-9) para decidir "correto", o que exige igualdade quase exata
# e praticamente nunca é satisfeito por uma resposta gerada por IA. Isso
# fazia todo item cair em "incorreto", mesmo com erro de 1-2%. Agora o
# status usa o mesmo erro percentual reportado nos resultados, com uma
# tolerância de engenharia configurável (padrão 5%).
#
# Essa função também elimina a redundância de reparsear e recalcular o
# erro em três lugares diferentes (classificar_status, erro_absoluto,
# erro_percentual): agora tudo é calculado uma única vez aqui e reutilizado.
# =========================================================
def _comparar_valor(valor_ia, valor_real, tolerancia_percentual):
    """Compara um par (valor_ia, valor_real) e retorna erro absoluto, erro
    percentual e status ('correto', 'incorreto' ou 'faltante') em uma
    única passada, evitando reparsing redundante."""

    valor_ia_n = parse_numeric_value(valor_ia)
    valor_real_n = parse_numeric_value(valor_real)

    if valor_ia_n is None:
        return {
            "valor_ia_n": None,
            "valor_real_n": valor_real_n,
            "erro_absoluto": None,
            "erro_percentual": None,
            "referencia_zero": False,
            "status": "faltante",
        }

    if valor_real_n is None:
        return {
            "valor_ia_n": valor_ia_n,
            "valor_real_n": None,
            "erro_absoluto": None,
            "erro_percentual": None,
            "referencia_zero": False,
            "status": "incorreto",
        }

    erro_abs = abs(valor_ia_n - valor_real_n)
    referencia_zero = abs(valor_real_n) < ZERO_ABS_TOL

    if referencia_zero:
        # Erro percentual não é definido quando a referência é zero.
        # Classificamos "correto" apenas se a IA também previu ~zero;
        # o caso fica marcado em referencia_zero para não ser silenciosamente
        # descartado das estatísticas agregadas (ver avaliar_categoria).
        erro_pct = 0.0 if abs(valor_ia_n) < ZERO_ABS_TOL else None
        status = "correto" if abs(valor_ia_n) < ZERO_ABS_TOL else "incorreto"
    else:
        erro_pct = erro_abs / abs(valor_real_n) * 100
        status = "correto" if erro_pct <= tolerancia_percentual else "incorreto"

    return {
        "valor_ia_n": valor_ia_n,
        "valor_real_n": valor_real_n,
        "erro_absoluto": erro_abs,
        "erro_percentual": erro_pct,
        "referencia_zero": referencia_zero,
        "status": status,
    }


# =========================================================
# Wrappers públicos mantidos para compatibilidade com o resto da pipeline
# (main.py e outros módulos podem depender desses nomes de função)
# =========================================================
def classificar_status(valor_real, valor_ia, tolerancia_percentual=TOLERANCIA_PADRAO_PERCENTUAL):
    """Classifica uma comparação como 'correto', 'incorreto' ou 'faltante'
    usando tolerância percentual de engenharia (padrão 5%)."""
    return _comparar_valor(valor_ia, valor_real, tolerancia_percentual)["status"]


def erro_absoluto(valor_real, valor_ia):
    """Calcula o erro absoluto entre o valor real e o valor inferido pela IA."""
    return _comparar_valor(valor_ia, valor_real, TOLERANCIA_PADRAO_PERCENTUAL)["erro_absoluto"]


def erro_percentual(valor_real, valor_ia):
    """Calcula o erro percentual relativo ao valor de referência."""
    return _comparar_valor(valor_ia, valor_real, TOLERANCIA_PADRAO_PERCENTUAL)["erro_percentual"]


# =========================================================
# MAE / RMSE / Erro Relativo Médio
# =========================================================
def calcular_mae(valores_ia, valores_reais):
    """Calcula o Mean Absolute Error a partir dos erros absolutos disponíveis."""
    erros = [
        erro_absoluto(real, ia)
        for ia, real in zip(valores_ia, valores_reais)
    ]
    erros = [e for e in erros if e is not None]
    return sum(erros) / len(erros) if erros else None


def calcular_rmse(valores_ia, valores_reais):
    """Calcula o Root Mean Square Error usando os erros quadráticos válidos."""
    erros_quad = [
        e ** 2 for e in (
            erro_absoluto(real, ia) for ia, real in zip(valores_ia, valores_reais)
        ) if e is not None
    ]
    return math.sqrt(sum(erros_quad) / len(erros_quad)) if erros_quad else None


def calcular_erm(valores_ia, valores_reais):
    """Calcula o erro relativo médio percentual considerando apenas valores válidos
    (casos com referência zero não entram aqui — ver 'referencia_zero' em avaliar_categoria)."""
    erros = [
        erro_percentual(real, ia)
        for ia, real in zip(valores_ia, valores_reais)
    ]
    erros = [e for e in erros if e is not None]
    return sum(erros) / len(erros) if erros else None


# =========================================================
# Extrai pares IA x referência de um dicionário
# =========================================================
def extrair_pares_dicionario(valores_ia, valores_reais):
    """Gera pares chave -> valor_IA -> valor_real para comparar categorias inteiras."""
    chaves = sorted(set(valores_ia.keys()) | set(valores_reais.keys()))
    return [(chave, valores_ia.get(chave), valores_reais.get(chave)) for chave in chaves]


# =========================================================
# Avalia uma categoria de grandezas (tensões, correntes, potências)
# =========================================================
def avaliar_categoria(valores_ia, valores_reais, tolerancia_percentual=TOLERANCIA_PADRAO_PERCENTUAL):
    """Avalia uma categoria inteira, item a item, em uma única passada
    (a versão anterior iterava duas vezes sobre os pares para montar o
    mesmo resultado)."""

    pares = extrair_pares_dicionario(valores_ia, valores_reais)

    status_counts = {"correto": 0, "incorreto": 0, "faltante": 0}
    valores = {}
    valores_ia_validos = []
    valores_real_validos = []
    quantidade_referencia_zero = 0

    for nome, valor_ia, valor_real in pares:
        comp = _comparar_valor(valor_ia, valor_real, tolerancia_percentual)

        status_counts[comp["status"]] += 1
        if comp["referencia_zero"]:
            quantidade_referencia_zero += 1

        if comp["valor_ia_n"] is not None and comp["valor_real_n"] is not None:
            valores_ia_validos.append(comp["valor_ia_n"])
            valores_real_validos.append(comp["valor_real_n"])

        valores[nome] = {
            "IA": valor_ia,
            "REAL": valor_real,
            "erro_absoluto": comp["erro_absoluto"],
            "erro_percentual": comp["erro_percentual"],
            "status": comp["status"],
        }

    return {
        "quantidade": len(pares),
        "ERM_percentual": _round_if_not_none(calcular_erm(valores_ia_validos, valores_real_validos), 4),
        "MAE": _round_if_not_none(calcular_mae(valores_ia_validos, valores_real_validos), 8),
        "RMSE": _round_if_not_none(calcular_rmse(valores_ia_validos, valores_real_validos), 8),
        "status_counts": status_counts,
        "quantidade_referencia_zero": quantidade_referencia_zero,
        "valores": valores,
    }


# =========================================================
# Avalia uma grandeza escalar (ex: Vout, ganho)
# =========================================================
def avaliar_escalar(valor_ia, valor_real, tolerancia_percentual=TOLERANCIA_PADRAO_PERCENTUAL):
    """Avalia um valor escalar simples, como Vout ou ganho teórico."""
    comp = _comparar_valor(valor_ia, valor_real, tolerancia_percentual)

    return {
        "IA": valor_ia if comp["valor_ia_n"] is None else comp["valor_ia_n"],
        "REAL": valor_real if comp["valor_real_n"] is None else comp["valor_real_n"],
        "erro_absoluto": _round_if_not_none(comp["erro_absoluto"], 8),
        "erro_percentual": _round_if_not_none(comp["erro_percentual"], 4),
        "status": comp["status"],
    }


# =========================================================
# Função principal da avaliação NUMÉRICA
# =========================================================
def avaliar_numerico(resposta: dict, gabarito: dict, tolerancia_percentual=TOLERANCIA_PADRAO_PERCENTUAL):
    """Organiza a avaliação numérica das seções Vout, ganho, tensões, correntes e potências."""

    analise_ia = resposta.get("analise_numerica", resposta.get("analise_eletrica", {})) or {}
    analise_real = gabarito.get("analise_numerica", gabarito.get("analise_eletrica", {})) or {}

    resultado = {}

    vout_ia = analise_ia.get("Vout_real_V", analise_ia.get("Vout_V"))
    vout_real = analise_real.get("Vout_real_V", analise_real.get("Vout_V"))
    if vout_ia is not None or vout_real is not None:
        resultado["Vout"] = avaliar_escalar(vout_ia, vout_real, tolerancia_percentual)

    vout_teorico_ia = analise_ia.get("Vout_teorico_V")
    vout_teorico_real = analise_real.get("Vout_teorico_V")
    if vout_teorico_ia is not None or vout_teorico_real is not None:
        resultado["Vout_teorico"] = avaliar_escalar(vout_teorico_ia, vout_teorico_real, tolerancia_percentual)

    ganho_ia = analise_ia.get("ganho_teorico")
    ganho_real = analise_real.get("ganho_teorico")
    if ganho_ia is not None or ganho_real is not None:
        resultado["ganho"] = avaliar_escalar(ganho_ia, ganho_real, tolerancia_percentual)

    resultado["tensoes"] = avaliar_categoria(
        analise_ia.get("tensoes_V", {}) or {},
        analise_real.get("tensoes_V", {}) or {},
        tolerancia_percentual,
    )

    resultado["correntes"] = avaliar_categoria(
        analise_ia.get("correntes_A", {}) or {},
        analise_real.get("correntes_A", {}) or {},
        tolerancia_percentual,
    )

    resultado["potencias"] = avaliar_categoria(
        analise_ia.get("potencias_W", {}) or {},
        analise_real.get("potencias_W", {}) or {},
        tolerancia_percentual,
    )

    return resultado


# =========================================================
# Função principal
#
# Mantida com esse nome para facilitar a integração com o main.py existente.
# =========================================================
def avaliar_resposta(resposta: dict, gabarito: dict, tolerancia_percentual=TOLERANCIA_PADRAO_PERCENTUAL):
    """Executa a avaliação completa, separando os dados numéricos e textuais antes de comparar."""
    resposta_separada = separar_resposta(resposta)
    gabarito_separado = separar_resposta(gabarito)

    resultado_numerico = avaliar_numerico(
        resposta_separada.get("numerico", {}),
        gabarito_separado.get("numerico", {}),
        tolerancia_percentual,
    )

    resultado_textual = avaliar_textual(
        resposta_separada.get("textual", {}),
        gabarito_separado.get("textual", {}),
    )

    return {
        "numerico": resultado_numerico,
        "textual": resultado_textual,
    }
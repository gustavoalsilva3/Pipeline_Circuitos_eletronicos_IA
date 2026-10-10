"""Módulo de avaliação numérica para comparar respostas da IA com gabaritos.

Converte valores para um formato numérico comum, classifica cada comparação
como correta, incorreta, faltante (existe no gabarito, a IA não informou) ou
alucinada (a IA informou um nó/componente que não existe no gabarito) —
usando tolerância percentual de engenharia, não igualdade exata —, calcula
métricas de erro (ERM, MAE, RMSE, taxa_acerto_percentual) e organiza o
resultado em seções para Vout, ganho, tensões, correntes e potências. Mesmo
vocabulário de status usado em evaluator_componentes.py e
evaluator_textual.py, para que o detalhe (resultados_detalhe_*.csv) seja
consistente entre as três dimensões avaliadas.

IMPORTANTE sobre qual métrica usar como principal em correntes/potências:
ERM_percentual EXCLUI da média os itens com referência ~0 (ver
'referencia_zero' em _comparar_valor), porque erro percentual não é
definido quando o valor de referência é zero. Isso é correto item a item,
mas tem um efeito colateral sério ao AGREGAR por categoria de circuito:
circuitos com muitos ramos de corrente zero (comuns em amp-op saturado ou
diodo em corte) podem ficar com ERM_percentual vazio/None mesmo quando o
modelo errou TODOS os valores — e uma média feita ignorando esses None
silenciosamente exclui exatamente os circuitos onde o modelo mais erra,
inflando artificialmente o resultado agregado.

Por isso, 'taxa_acerto_percentual' (baseado em status_counts, que NUNCA
descarta nada) é a métrica recomendada como principal para comparações
agregadas entre categorias/prompts/modelos. Use ERM_percentual como
complementar (grandeza do erro nos casos onde ele é definido), não como
métrica isolada de desempenho.
"""

import re
import math

from config import TOLERANCIA_PADRAO_PERCENTUAL
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

# Tolerância absoluta usada apenas para decidir se um valor "é zero"
# (evita comparação exata de float quando o gabarito é 0). A tolerância
# PERCENTUAL usada para classificar uma comparação como "correta" (padrão
# 5%) é TOLERANCIA_PADRAO_PERCENTUAL, importada de config.py — um
# resultado de IA raramente bate com o gabarito na precisão exata de
# ponto flutuante, por isso o critério é uma faixa de erro relativo
# aceitável, não igualdade exata.
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


def _modulo(valor):
    """Módulo do valor numérico; valores não numéricos voltam inalterados."""
    valor_n = parse_numeric_value(valor)
    return valor if valor_n is None else abs(valor_n)


def _round_if_not_none(valor, casas_decimais):
    """Arredonda valores apenas quando eles existem, preservando None para dados ausentes."""
    return None if valor is None else round(valor, casas_decimais)


# =========================================================
# Comparação central de um único par (IA x referência) — usada por todas
# as funções públicas do módulo (avaliar_categoria, avaliar_escalar), para
# que status/erro_absoluto/erro_percentual sejam sempre calculados da
# mesma forma em um único lugar.
#
# O status usa tolerância percentual de engenharia (padrão 5%, ver
# TOLERANCIA_PADRAO_PERCENTUAL), não igualdade exata — uma resposta gerada
# por IA praticamente nunca bate com o gabarito na precisão exata de ponto
# flutuante, então exigir igualdade classificaria como "incorreto" até
# respostas com erro de 1-2%, que são aceitáveis do ponto de vista de
# engenharia.
# =========================================================
def _comparar_valor(valor_ia, valor_real, tolerancia_percentual):
    """Compara um par (valor_ia, valor_real) e retorna erro absoluto, erro
    percentual e status ('correto', 'incorreto', 'faltante' ou 'alucinado')
    em uma única passada."""

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
        # A IA informou um valor para um nó/componente que não existe no
        # gabarito (ou o gabarito não tem valor de referência para ele) —
        # mesmo status 'alucinado' usado em evaluator_componentes.py e
        # evaluator_textual.py para o mesmo tipo de caso, em vez de
        # 'incorreto' (que sugeriria haver uma referência para comparar).
        return {
            "valor_ia_n": valor_ia_n,
            "valor_real_n": None,
            "erro_absoluto": None,
            "erro_percentual": None,
            "referencia_zero": False,
            "status": "alucinado",
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
# MAE / RMSE / Erro Relativo Médio
#
# Recebem pares já numéricos (avaliar_categoria só passa aqui os itens
# onde IA e referência foram parseados com sucesso — ver
# 'valores_ia_validos'/'valores_real_validos'), por isso operam direto com
# aritmética simples em vez de passar de novo por _comparar_valor.
# =========================================================
def calcular_mae(valores_ia, valores_reais):
    """Calcula o Mean Absolute Error a partir de pares já numéricos."""
    if not valores_ia:
        return None
    erros = [abs(ia - real) for ia, real in zip(valores_ia, valores_reais)]
    return sum(erros) / len(erros)


def calcular_rmse(valores_ia, valores_reais):
    """Calcula o Root Mean Square Error a partir de pares já numéricos."""
    if not valores_ia:
        return None
    erros_quad = [(ia - real) ** 2 for ia, real in zip(valores_ia, valores_reais)]
    return math.sqrt(sum(erros_quad) / len(erros_quad))


def calcular_erm(valores_ia, valores_reais):
    """Calcula o erro relativo médio percentual a partir de pares já
    numéricos, excluindo pares com referência ~0 (erro percentual não é
    definido nesse caso — mesmo critério de 'referencia_zero' usado em
    _comparar_valor/avaliar_categoria), exceto quando a IA também previu
    ~0, que conta como erro 0%."""
    erros = []
    for ia, real in zip(valores_ia, valores_reais):
        if abs(real) < ZERO_ABS_TOL:
            if abs(ia) < ZERO_ABS_TOL:
                erros.append(0.0)
            continue
        erros.append(abs(ia - real) / abs(real) * 100)
    return sum(erros) / len(erros) if erros else None


# =========================================================
# Normaliza chave de nó/componente (mesma lógica de _normalizar_id em
# evaluator_componentes) — sem isso, "a" e "A" (ou " A ") seriam tratados
# como dois nós diferentes em vez do mesmo nó, inflando 'faltante'/
# 'incorreto' artificialmente e, se a IA variar a caixa entre execuções,
# criando entradas inconsistentes na comparação.
# =========================================================
def _normalizar_chave(chave):
    """Normaliza a chave de um nó (tensoes_V) ou componente (correntes_A/
    potencias_W) para maiúsculas e sem espaços nas pontas."""
    if chave is None:
        return ""
    return str(chave).strip().upper()


# =========================================================
# Extrai pares IA x referência de um dicionário
# =========================================================
def extrair_pares_dicionario(valores_ia, valores_reais):
    """Gera pares chave -> valor_IA -> valor_real para comparar categorias
    inteiras. As chaves são normalizadas antes da comparação (ver
    _normalizar_chave)."""
    ia_normalizado = {_normalizar_chave(chave): valor for chave, valor in valores_ia.items()}
    reais_normalizado = {_normalizar_chave(chave): valor for chave, valor in valores_reais.items()}

    chaves = sorted(set(ia_normalizado.keys()) | set(reais_normalizado.keys()))
    return [(chave, ia_normalizado.get(chave), reais_normalizado.get(chave)) for chave in chaves]


# =========================================================
# Avalia uma categoria de grandezas (tensões, correntes, potências)
# =========================================================
def avaliar_categoria(valores_ia, valores_reais, tolerancia_percentual=TOLERANCIA_PADRAO_PERCENTUAL):
    """Avalia uma categoria inteira (tensões, correntes ou potências) item
    a item, em uma única passada sobre os pares IA x referência."""

    pares = extrair_pares_dicionario(valores_ia, valores_reais)

    status_counts = {"correto": 0, "incorreto": 0, "faltante": 0, "alucinado": 0}
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

    # Sem nenhum item no gabarito (ex: circuito sem nós de tensão rotulados),
    # a taxa não é aplicável, mesmo que a IA tenha informado itens: eles
    # continuam registrados como 'alucinado', mas não viram uma taxa de 0%.
    # Sem isso, o mesmo circuito dava taxa indefinida a quem não informava
    # nada e 0% a quem inventava rótulos, tratando os modelos de forma desigual.
    tem_referencia = any(valor_real is not None for _, _, valor_real in pares)

    return {
        "quantidade": len(pares),
        "taxa_acerto_percentual": round(status_counts["correto"] / len(pares) * 100, 1) if tem_referencia else None,
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

    tensoes_ia = analise_ia.get("tensoes_V", {}) or {}
    tensoes_real = analise_real.get("tensoes_V", {}) or {}

    # 'Vout' é o rótulo da saída, que a imagem mostra e que já é avaliado à
    # parte (Vout_real_V). Os nós de 'tensoes_V' são as letras (A, B, C...) do
    # gabarito; se a IA acrescenta uma chave 'Vout' que o gabarito não tem, ela
    # não conta como nó alucinado em tensões (o modelo só foi além do pedido,
    # sem inventar nada). Só a resposta da IA é filtrada, e só quando o
    # gabarito não traz 'Vout' entre os nós — nenhum dado é alterado.
    if not any(_normalizar_chave(k) == "VOUT" for k in tensoes_real):
        tensoes_ia = {k: v for k, v in tensoes_ia.items() if _normalizar_chave(k) != "VOUT"}

    resultado["tensoes"] = avaliar_categoria(tensoes_ia, tensoes_real, tolerancia_percentual)

    # O gabarito registra as correntes em módulo (o LTspice não define o
    # sentido de referência que a imagem também não mostra), então um modelo
    # que adota o sentido oposto acerta a magnitude com o sinal trocado. As
    # correntes são comparadas em módulo: só a resposta da IA é convertida,
    # e valores não numéricos seguem como estão (viram 'faltante').
    correntes_ia = {k: _modulo(v) for k, v in (analise_ia.get("correntes_A", {}) or {}).items()}
    resultado["correntes"] = avaliar_categoria(
        correntes_ia,
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
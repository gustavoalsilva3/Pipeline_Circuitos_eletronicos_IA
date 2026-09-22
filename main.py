"""Pipeline principal para analisar circuitos com Gemini e avaliar as respostas.

Este módulo coordena as etapas de:
1. localizar imagens e gabaritos;
2. enviar a imagem ao modelo Gemini;
3. salvar a resposta bruta em JSON;
4. carregar o gabarito correspondente;
5. avaliar a resposta nas três dimensões (numérica, textual, componentes);
6. consolidar os resultados em dois CSVs: um resumo (uma linha por execução,
   só com métricas agregadas) e um detalhe (uma linha por item comparado —
   nó de tensão, ID de componente — ver avaliar_resposta_completa).
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import (
    GABARITOS_DIR,
    IMAGENS_DIR,
    RESULTADOS_DIR,
    JSON_DIR,
    PREFIXO_CSV_RESULTADOS,
    PREFIXO_CSV_DETALHES,
    MODELO_GEMINI,
    USAR_MODELO_GEMINI,
    MODELOS_LMSTUDIO,
    MAX_TOKENS_SAIDA_POR_MODELO,
    TIMEOUT_POR_MODELO_S,
    USAR_MODELO_LMSTUDIO,
    MODELOS_OPENROUTER,
    USAR_MODELO_OPENROUTER,
    PRECO_INPUT_1M,
    PRECO_OUTPUT_1M,
    COTACAO_USD_BRL,
    IDIOMA_PROMPT,
    REEXECUTAR_TRUNCADAS,
    MAX_REEXECUCOES,
    TEMPERATURA_REEXECUCAO,
)

from prompts import (
    PROMPT_ANALISE_CIRCUITO_1,
    PROMPT_ANALISE_CIRCUITO_2,
    PROMPT_ANALISE_CIRCUITO_3,
    PROMPT_ANALISE_CIRCUITO_1_EN,
    PROMPT_ANALISE_CIRCUITO_2_EN,
    PROMPT_ANALISE_CIRCUITO_3_EN,
)
from gemini_client import GeminiCircuitAnalyzer
from lmstudio_client import LMStudioCircuitAnalyzer
from openrouter_client import OpenRouterCircuitAnalyzer
from evaluator_numerico import avaliar_resposta as avaliar_numerico_textual
from evaluator_componentes import avaliar_componentes


# =========================================================
# Achata um dict aninhado em chaves separadas por '_' (ex: {"numerico":
# {"Vout": {"status": "correto"}}} -> {"numerico_Vout_status": "correto"}).
#
# Sem isso, **avaliacao no CSV jogaria dicts aninhados direto nas colunas
# do DataFrame — cada célula viraria uma string Python ilegível em vez de
# colunas separadas que dá pra usar em groupby/pivot no pandas depois.
# Listas (ex: uma eventual 'categorias_invalidas') viram uma string JSON
# compacta em vez de um objeto Python, pelo mesmo motivo.
# =========================================================
def _achatar(dados, prefixo=""):

    plano = {}

    for chave, valor in dados.items():
        chave_completa = f"{prefixo}{chave}" if not prefixo else f"{prefixo}_{chave}"

        if isinstance(valor, dict):
            plano.update(_achatar(valor, chave_completa))
        elif isinstance(valor, list):
            plano[chave_completa] = json.dumps(valor, ensure_ascii=False) if valor else ""
        else:
            plano[chave_completa] = valor

    return plano


# =========================================================
# Chaves com granularidade por item (nó de tensão, ID de componente) que
# NÃO entram no resumo achatado — cada uma vira, em vez disso, uma ou mais
# linhas na planilha de detalhe (formato longo). São elas que crescem sem
# limite conforme aparecem circuitos com nós/componentes diferentes; o
# resumo fica só com métricas agregadas (taxa_acerto, precision/recall/f1
# etc.), que têm sempre o mesmo formato não importa quantos itens existam.
# =========================================================
CHAVES_DETALHE_NUMERICO = {"valores"}
CHAVES_DETALHE_COMPONENTES = {
    "valores",
    "componentes_identificados",
    "componentes_alucinados",
    "componentes_nao_identificados",
}


def _resumo_numerico(resultado_numerico: dict) -> dict:
    """Remove o dict 'valores' (item a item) de cada categoria (tensoes/
    correntes/potencias), mantendo só os agregados (taxa_acerto_percentual,
    ERM, MAE, RMSE, status_counts...). Vout/Vout_teorico/ganho são escalares
    (não têm 'valores'), então passam intactos."""

    resumo = {}
    for chave, valor in resultado_numerico.items():
        if isinstance(valor, dict) and "valores" in valor:
            resumo[chave] = {k: v for k, v in valor.items() if k not in CHAVES_DETALHE_NUMERICO}
        else:
            resumo[chave] = valor
    return resumo


def _resumo_componentes(resultado_componentes: dict) -> dict:
    """Remove 'valores' e as listas de IDs (identificados/alucinados/
    não_identificados) de cada categoria, mantendo TP/FP/FN/precision/
    recall/f1_score/valor_taxa_acerto_percentual."""

    resumo = {}
    for categoria, info in resultado_componentes.items():
        if categoria == "categorias_invalidas" or not isinstance(info, dict):
            resumo[categoria] = info
            continue
        resumo[categoria] = {k: v for k, v in info.items() if k not in CHAVES_DETALHE_COMPONENTES}
    return resumo


def _resumo_textual(resultado_textual: dict) -> dict:
    """Remove o dict 'valores' (por componente) de 'regiao_operacao',
    mantendo só o agregado (quantidade, taxa_acerto_percentual,
    status_counts). 'terra_virtual' é um campo escalar único (não tem
    'valores'), passa intacto."""

    resumo = {}
    for chave, valor in resultado_textual.items():
        if isinstance(valor, dict) and "valores" in valor:
            resumo[chave] = {k: v for k, v in valor.items() if k != "valores"}
        else:
            resumo[chave] = valor
    return resumo


def _extrair_detalhes_textual(resultado_textual: dict, id_execucao: int) -> list[dict]:
    """Converte o dict 'valores' (por componente) de 'regiao_operacao' em
    linhas de detalhe, mesmo formato longo do numérico/componentes."""

    linhas = []
    bloco = resultado_textual.get("regiao_operacao")
    if isinstance(bloco, dict):
        for item, info in bloco.get("valores", {}).items():
            linhas.append({
                "id_execucao": id_execucao,
                "origem": "textual",
                "categoria": "regiao_operacao",
                "item": item,
                "valor_ia": info.get("obtido"),
                "valor_real": info.get("esperado"),
                "status": info.get("status"),
            })
    return linhas


def _extrair_detalhes_numerico(resultado_numerico: dict, id_execucao: int) -> list[dict]:
    """Converte o dict 'valores' de tensoes/correntes/potencias em linhas
    de detalhe (uma por nó/componente), no formato longo."""

    linhas = []
    for categoria in ("tensoes", "correntes", "potencias"):
        bloco = resultado_numerico.get(categoria, {})
        if not isinstance(bloco, dict):
            continue
        for item, info in bloco.get("valores", {}).items():
            linhas.append({
                "id_execucao": id_execucao,
                "origem": "numerico",
                "categoria": categoria,
                "item": item,
                "valor_ia": info.get("IA"),
                "valor_real": info.get("REAL"),
                "status": info.get("status"),
            })
    return linhas


def _extrair_detalhes_componentes(resultado_componentes: dict, id_execucao: int) -> list[dict]:
    """Converte 'valores' (comparação de valor) e as listas de IDs
    (alucinados/não identificados) de cada categoria de componente em
    linhas de detalhe."""

    linhas = []
    for categoria, info in resultado_componentes.items():
        if categoria == "categorias_invalidas" or not isinstance(info, dict):
            continue

        for item, val in info.get("valores", {}).items():
            valor_ia = val.get("valor_obtido")
            valor_real = val.get("valor_esperado")

            # Categorias sem campo 'valor' aplicável no schema (amp-ops e
            # diodos só têm 'id') fazem os dois lados virem None aqui — não
            # é uma comparação "faltante" de verdade (o componente já foi
            # corretamente identificado, é um TP), é simplesmente um campo
            # que não existe para essa categoria. Sem esse filtro, TODO
            # amp-op/diodo identificado certo virava uma linha 'faltante'
            # no detalhe, poluindo status_geral/status_por_categoria com um
            # falso 100% de falha nessas duas categorias.
            if valor_ia is None and valor_real is None:
                continue

            valor_correto = val.get("valor_correto")
            status = "faltante" if valor_correto is None else ("correto" if valor_correto else "incorreto")
            linhas.append({
                "id_execucao": id_execucao,
                "origem": "componentes",
                "categoria": categoria,
                "item": item,
                "valor_ia": valor_ia,
                "valor_real": valor_real,
                "status": status,
            })

        for item in info.get("componentes_alucinados", []):
            linhas.append({
                "id_execucao": id_execucao, "origem": "componentes", "categoria": categoria,
                "item": item, "valor_ia": "", "valor_real": "", "status": "alucinado",
            })

        for item in info.get("componentes_nao_identificados", []):
            linhas.append({
                "id_execucao": id_execucao, "origem": "componentes", "categoria": categoria,
                "item": item, "valor_ia": "", "valor_real": "", "status": "nao_identificado",
            })

    return linhas


def avaliar_resposta_completa(resposta: dict, gabarito: dict, id_execucao: int) -> tuple[dict, list[dict]]:
    """Avalia uma resposta contra o gabarito usando os três módulos e
    devolve (resumo achatado, lista de linhas de detalhe).

    O resumo tem sempre o mesmo formato de colunas (métricas agregadas);
    o detalhe é quem carrega a comparação item a item (nó de tensão, ID de
    componente), em formato longo — uma linha por item em vez de colunas
    novas por item.

    Se o gabarito estiver ausente (dict vazio), NÃO avalia — evita gerar
    'incorreto'/'faltante' em tudo, o que poluiria as médias agregadas
    sem deixar claro que a causa foi gabarito ausente, não erro do modelo.
    """

    if not gabarito:
        return {"gabarito_encontrado": False}, []

    resultado_num_textual = avaliar_numerico_textual(resposta, gabarito)
    resultado_componentes = avaliar_componentes(resposta, gabarito)

    combinado = {
        "gabarito_encontrado": True,
        "numerico": _resumo_numerico(resultado_num_textual["numerico"]),
        "textual": _resumo_textual(resultado_num_textual["textual"]),
        "componentes": _resumo_componentes(resultado_componentes),
    }

    detalhes = (
        _extrair_detalhes_numerico(resultado_num_textual["numerico"], id_execucao)
        + _extrair_detalhes_componentes(resultado_componentes, id_execucao)
        + _extrair_detalhes_textual(resultado_num_textual["textual"], id_execucao)
    )

    return _achatar(combinado), detalhes


# =========================================================
# Níveis de prompt (direto / guiado / estruturado), por idioma. As chaves
# JSON pedidas na resposta são sempre em português (ver prompts.py); só o
# texto de instrução muda entre os dois dicionários abaixo.
# =========================================================
NIVEIS_PROMPT_POR_IDIOMA = {
    "pt": {
        1: PROMPT_ANALISE_CIRCUITO_1,
        2: PROMPT_ANALISE_CIRCUITO_2,
        3: PROMPT_ANALISE_CIRCUITO_3,
    },
    "en": {
        1: PROMPT_ANALISE_CIRCUITO_1_EN,
        2: PROMPT_ANALISE_CIRCUITO_2_EN,
        3: PROMPT_ANALISE_CIRCUITO_3_EN,
    },
}

# =========================================================
# Idiomas efetivamente rodados nesta execução, conforme config.IDIOMA_PROMPT
# ("pt", "en" ou "ambos" — roda os dois, um após o outro, no mesmo par de
# CSVs, marcando cada linha com a coluna "idioma").
# =========================================================
if IDIOMA_PROMPT == "ambos":
    IDIOMAS_EXECUCAO = ["pt", "en"]
elif IDIOMA_PROMPT in ("pt", "en"):
    IDIOMAS_EXECUCAO = [IDIOMA_PROMPT]
else:
    raise ValueError(
        f"config.IDIOMA_PROMPT inválido: {IDIOMA_PROMPT!r} — use 'pt', 'en' ou 'ambos'."
    )


# =========================================================
# Modelos avaliados. Cada entrada precisa de um "analisador" com o método
# analisar_imagem(image_path, prompt) -> (resposta, input_tokens,
# output_tokens), no mesmo formato do GeminiCircuitAnalyzer, e de
# "preco_input_1m"/"preco_output_1m" (USD por 1M tokens, usados só para o
# relatório de custo — 0.0 para um modelo local/gratuito como o do LM Studio).
#
# Cada modelo é ligado/desligado por um toggle em config.py
# (USAR_MODELO_GEMINI / USAR_MODELO_LMSTUDIO / USAR_MODELO_OPENROUTER), mesmo
# espírito do IDIOMA_PROMPT — permite rodar só alguns (ex: só um modelo
# local, sem gastar a API do Gemini) sem comentar/descomentar código. O
# LM Studio aceita uma lista de modelos (config.MODELOS_LMSTUDIO), cada um
# vira uma entrada separada.
# =========================================================
def montar_modelos():
    """Instancia os analisadores de cada modelo avaliado."""
    modelos = []

    if USAR_MODELO_GEMINI:
        modelos.append({
            "nome": MODELO_GEMINI,
            "analisador": GeminiCircuitAnalyzer(model_name=MODELO_GEMINI),
            "preco_input_1m": PRECO_INPUT_1M,
            "preco_output_1m": PRECO_OUTPUT_1M,
        })

    if USAR_MODELO_LMSTUDIO:
        for modelo_id in MODELOS_LMSTUDIO:
            modelos.append({
                "nome": modelo_id,
                "analisador": LMStudioCircuitAnalyzer(
                    model_name=modelo_id,
                    max_tokens=MAX_TOKENS_SAIDA_POR_MODELO.get(modelo_id),
                    timeout=TIMEOUT_POR_MODELO_S.get(modelo_id),
                ),
                "preco_input_1m": 0.0,
                "preco_output_1m": 0.0,
            })

    if USAR_MODELO_OPENROUTER:
        for cfg in MODELOS_OPENROUTER:
            modelos.append({
                "nome": cfg.get("rotulo") or cfg["id"],
                "analisador": OpenRouterCircuitAnalyzer(
                    model_name=cfg["id"],
                    provider=cfg.get("provider"),
                    max_tokens=cfg.get("max_tokens"),
                    timeout=cfg.get("timeout"),
                ),
                "preco_input_1m": cfg.get("preco_input_1m", 0.0),
                "preco_output_1m": cfg.get("preco_output_1m", 0.0),
            })

    if not modelos:
        raise ValueError(
            "Nenhum modelo ativo — ligue pelo menos um em config.py "
            "(USAR_MODELO_GEMINI, USAR_MODELO_LMSTUDIO e/ou USAR_MODELO_OPENROUTER)."
        )

    return modelos


# =========================================================
# Cria as pastas necessárias caso não existam
# =========================================================
def preparar_pastas():
    """Cria as pastas utilizadas para guardar resultados, respostas JSON e demais artefatos."""

    RESULTADOS_DIR.mkdir(exist_ok=True)
    JSON_DIR.mkdir(exist_ok=True)


# =========================================================
# Salva o DataFrame em CSV com fallback para outro nome
# caso o arquivo padrão esteja travado ou inacessível
# =========================================================
def salvar_resultados_csv(df_final: pd.DataFrame, caminho: Path) -> Path:
    """Salva o DataFrame em CSV e usa um nome alternativo se o arquivo padrão estiver bloqueado."""
    try:
        df_final.to_csv(
            caminho,
            index=False,
            sep=";",
            decimal=",",
            encoding="utf-8-sig",
        )
        return caminho
    except PermissionError:
        nome_fallback = caminho.with_name(
            f"{caminho.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{caminho.suffix}"
        )
        print(
            f"Atenção: não foi possível gravar {caminho.name} por permissão."
            f" Salvando em {nome_fallback.name}."
        )
        df_final.to_csv(
            nome_fallback,
            index=False,
            sep=";",
            decimal=",",
            encoding="utf-8-sig",
        )
        return nome_fallback
    except Exception as exc:
        nome_fallback = caminho.with_name(
            f"{caminho.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{caminho.suffix}"
        )
        print(f"Atenção: não foi possível salvar {caminho.name}: {exc}")
        print(f"Tentando salvar em {nome_fallback.name}.")
        df_final.to_csv(
            nome_fallback,
            index=False,
            sep=";",
            decimal=",",
            encoding="utf-8-sig",
        )
        return nome_fallback


# =========================================================
# Busca todas as imagens do dataset
# =========================================================
def listar_imagens():
    """Busca no diretório de imagens todos os arquivos suportados pelo pipeline."""

    extensoes = ["*.png", "*.jpg", "*.jpeg", "*.webp"]

    imagens = []

    for ext in extensoes:

        imagens.extend(IMAGENS_DIR.glob(ext))

    return sorted(imagens)


# =========================================================
# Carrega um JSON de gabarito encontrado para a imagem atual
# =========================================================
def carregar_json(caminho: Path) -> dict:
    """Carrega um arquivo JSON de forma segura, retornando um dicionário vazio em caso de erro."""
    try:
        with open(caminho, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _identificar_circuito(nome: str) -> tuple[str, int] | None:
    """Extrai a família e o número do circuito a partir do nome do arquivo."""
    nome_normalizado = nome.lower()

    if "amp" in nome_normalizado:
        familia = "amp"
    elif "diodo" in nome_normalizado or "dio" in nome_normalizado:
        familia = "dio"
    elif "rlc" in nome_normalizado:
        familia = "rlc"
    else:
        return None

    numeros = re.findall(r"\d+", nome_normalizado)
    if not numeros:
        return None

    return familia, int(numeros[-1])


def _nomes_batem(imagem_path: Path, arquivo_gabarito: Path) -> bool:
    """Compara estritamente a família e o número do circuito."""
    circuito_imagem = _identificar_circuito(imagem_path.stem)
    circuito_gabarito = _identificar_circuito(arquivo_gabarito.stem)

    return (
        circuito_imagem is not None
        and circuito_imagem == circuito_gabarito
    )


def carregar_gabarito(imagem_path: Path) -> tuple[dict, Path | None]:
    """Procura o JSON de gabarito mais adequado para a imagem atual, com fallback para o único arquivo disponível."""
    if not GABARITOS_DIR.exists():
        return {}, None

    caminho_exato = GABARITOS_DIR / f"{imagem_path.stem}.json"
    if caminho_exato.exists():
        return carregar_json(caminho_exato), caminho_exato

    arquivos = sorted(GABARITOS_DIR.glob("*.json"))
    if len(arquivos) == 1:
        return carregar_json(arquivos[0]), arquivos[0]

    for arquivo in arquivos:
        if _nomes_batem(imagem_path, arquivo):
            return carregar_json(arquivo), arquivo

    return {}, None


# =========================================================
# Sanitiza um nome de modelo (ex: "qwen3.5:4b") para uso em nome de
# arquivo. Caracteres como ':' são reservados no Windows/NTFS — usar um
# nome de arquivo com ':' no meio não dá erro visível, mas o Windows trata
# tudo depois do ':' como um Alternate Data Stream (ADS) oculto em vez de
# parte do nome: o arquivo "...qwen3.5:4b.json" vira um arquivo vazio
# "...qwen3.5" com o conteúdo de verdade escondido num stream "4b.json"
# que não aparece no Explorer nem é lido por open()/pandas normalmente.
# =========================================================
_CARACTERES_INVALIDOS_ARQUIVO = re.compile(r'[<>:"/\\|?*]')


def _sanitizar_nome_arquivo(texto: str) -> str:
    return _CARACTERES_INVALIDOS_ARQUIVO.sub("-", texto)


# =========================================================
# Salva a resposta da IA em JSON individual
# =========================================================
def salvar_json(nome_arquivo, resposta):
    """Salva a resposta bruta da IA em um arquivo JSON individual para rastreio."""

    caminho = JSON_DIR / f"{nome_arquivo}.json"

    with open(caminho, "w", encoding="utf-8") as f:

        json.dump(
            resposta,
            f,
            ensure_ascii=False,
            indent=4
        )

    return caminho


# =========================================================
# Linha do CSV resumo de uma execução (usada na execução normal e na
# reexecução de respostas truncadas, para as duas terem o mesmo formato).
# =========================================================
def _montar_linha(id_execucao, imagem_path, idioma, nivel_prompt, modelo_cfg, tempo_execucao,
                  input_tokens, output_tokens, custo_usd, json_path, gabarito_path, avaliacao,
                  reexecucoes=0):
    return {
        "id_execucao": id_execucao,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "imagem": imagem_path.name,
        "modelo": modelo_cfg["nome"],
        "idioma": idioma,
        "nivel_prompt": nivel_prompt,
        "tempo_execucao_s": tempo_execucao,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "custo_usd": round(custo_usd, 6),
        "arquivo_json": str(json_path),
        "arquivo_gabarito": str(gabarito_path) if gabarito_path else "",
        "reexecucoes": reexecucoes,
        **avaliacao,
    }


def _foi_truncada(resposta) -> bool:
    """True se a resposta foi cortada por tamanho (limite de tokens ou loop
    de repetição) — marcada como 'truncado_por_tamanho' pelos clientes."""
    return isinstance(resposta, dict) and bool(resposta.get("truncado_por_tamanho"))


# =========================================================
# Reexecuta, ao fim da rodada, as execuções que deram resposta truncada.
#
# Com temperatura 0 a mesma chamada repetiria o mesmo loop, então cada
# tentativa usa TEMPERATURA_REEXECUCAO e uma semente diferente. Se alguma
# tentativa der certo, ela substitui a linha do resumo e as linhas do
# detalhe daquela execução (a resposta truncada original é guardada como
# '<nome>_falha_original.json'); se todas falharem, a falha original é
# mantida e só a coluna 'reexecucoes' registra quantas tentativas houve.
# Devolve (input_tokens, output_tokens, custo_usd, recuperadas) gastos.
# =========================================================
def reexecutar_truncadas(pendentes, resultados, detalhes_resultados):
    gasto_in = gasto_out = 0
    gasto_custo = 0.0
    recuperadas = 0

    print("\n" + "=" * 55)
    print(f"🔁 Reexecutando {len(pendentes)} resposta(s) truncada(s) "
          f"(até {MAX_REEXECUCOES} tentativa(s) cada, temperatura {TEMPERATURA_REEXECUCAO})")
    print("=" * 55)

    for p in pendentes:
        modelo_cfg = p["modelo_cfg"]
        resposta = None

        for n in range(1, MAX_REEXECUCOES + 1):
            print(f"Reexecução {n}/{MAX_REEXECUCOES} — execução {p['id_execucao']}: "
                  f"{p['imagem_path'].name} | {p['idioma']} | nível {p['nivel_prompt']} | {modelo_cfg['nome']}")
            inicio = time.time()
            resposta, in_tok, out_tok = modelo_cfg["analisador"].analisar_imagem(
                image_path=p["imagem_path"],
                prompt=p["texto_prompt"],
                temperatura=TEMPERATURA_REEXECUCAO,
                seed=n,
            )
            tempo = round(time.time() - inicio, 2)
            custo = (in_tok * modelo_cfg["preco_input_1m"] + out_tok * modelo_cfg["preco_output_1m"]) / 1_000_000
            gasto_in += in_tok
            gasto_out += out_tok
            gasto_custo += custo

            if not _foi_truncada(resposta):
                break

        if _foi_truncada(resposta):
            linha = next(l for l in resultados if l["id_execucao"] == p["id_execucao"])
            linha["reexecucoes"] = n
            print(f"❌ Execução {p['id_execucao']}: continuou truncada após {n} reexecução(ões); falha original mantida.\n")
            continue

        recuperadas += 1
        original = JSON_DIR / f"{p['nome_base']}.json"
        if original.exists():
            original.replace(JSON_DIR / f"{p['nome_base']}_falha_original.json")
        json_path = salvar_json(p["nome_base"], resposta)

        avaliacao, detalhes = avaliar_resposta_completa(resposta, p["gabarito"], p["id_execucao"])
        nova = _montar_linha(p["id_execucao"], p["imagem_path"], p["idioma"], p["nivel_prompt"], modelo_cfg,
                             tempo, in_tok, out_tok, custo, json_path, p["gabarito_path"], avaliacao,
                             reexecucoes=n)
        pos = next(i for i, l in enumerate(resultados) if l["id_execucao"] == p["id_execucao"])
        resultados[pos] = nova
        detalhes_resultados[:] = [d for d in detalhes_resultados if d["id_execucao"] != p["id_execucao"]] + detalhes
        print(f"✅ Execução {p['id_execucao']}: recuperada na reexecução {n}.\n")

    return gasto_in, gasto_out, gasto_custo, recuperadas


# =========================================================
# Programa principal
# =========================================================

def main():
    """Executa o fluxo completo: análise da imagem, avaliação e exportação dos resultados."""

    preparar_pastas()

    modelos = montar_modelos()
    imagens = listar_imagens()

    # =====================================================
    # Cada chamada de main() gera um par de arquivos NOVO (resumo + detalhe),
    # com o timestamp desta execução no nome — não acumula/sobrescreve o
    # resultado de rodadas anteriores. Dentro desta mesma chamada, os
    # arquivos ainda são salvos a cada iteração (não só no final), então se
    # o processo travar ou for interrompido, o que já rodou não se perde.
    # =====================================================
    timestamp_execucao = datetime.now().strftime("%Y%m%d_%H%M%S")
    caminho_saida = RESULTADOS_DIR / f"{PREFIXO_CSV_RESULTADOS}_{timestamp_execucao}.csv"
    caminho_detalhe_saida = RESULTADOS_DIR / f"{PREFIXO_CSV_DETALHES}_{timestamp_execucao}.csv"

    # Linhas do CSV resumo (uma por execução) desta chamada do script.
    resultados = []

    # Linhas do CSV detalhe (uma por nó/componente comparado, formato longo
    # — ver avaliar_resposta_completa) desta chamada do script.
    detalhes_resultados = []

    # Acumuladores de tokens e custos para o relatório final
    acumulado_input_tokens = 0
    acumulado_output_tokens = 0
    acumulado_custo_usd = 0.0

    total_execucoes = len(imagens) * len(IDIOMAS_EXECUCAO) * 3 * len(modelos)
    print(f"\nTotal de imagens: {len(imagens)} | idiomas: {IDIOMAS_EXECUCAO} | níveis de prompt: 3 "
          f"| modelos: {len(modelos)} | total de execuções: {total_execucoes}\n")

    indice = 0

    # Execuções cuja resposta veio truncada (candidatas à reexecução no fim).
    pendentes = []

    # =====================================================
    # Loop principal: imagem x idioma x nível de prompt x modelo
    # =====================================================
    for imagem_path in imagens:

        # Carrega o gabarito uma vez por imagem (não muda entre idiomas/prompts/modelos)
        gabarito, gabarito_path = carregar_gabarito(imagem_path)
        if not gabarito:
            print(f"Atenção: gabarito não encontrado para {imagem_path.name} — "
                  f"todas as execuções desta imagem serão salvas com "
                  f"'gabarito_encontrado=False' e sem métricas.")

        for idioma in IDIOMAS_EXECUCAO:
            niveis_prompt = NIVEIS_PROMPT_POR_IDIOMA[idioma]

            for nivel_prompt, texto_prompt in niveis_prompt.items():
                for modelo_cfg in modelos:

                    indice += 1

                    print("=" * 50)
                    print(f"Execução {indice}/{total_execucoes}")
                    print(f"Imagem: {imagem_path.name} | Idioma: {idioma} | Prompt: nível {nivel_prompt} | Modelo: {modelo_cfg['nome']}")
                    print("=" * 50)

                    inicio = time.time()

                    # =================================================
                    # Envia imagem para o modelo e desempacota resposta + tokens
                    # =================================================
                    resposta, input_tokens, output_tokens = modelo_cfg["analisador"].analisar_imagem(
                        image_path=imagem_path,
                        prompt=texto_prompt,
                    )

                    tempo_execucao = round(time.time() - inicio, 2)

                    # =================================================
                    # Cálculo financeiro da requisição atual — usa o preço
                    # do modelo desta execução, não um preço fixo (o LM Studio
                    # é gratuito, preco_input_1m/preco_output_1m = 0.0).
                    # =================================================
                    custo_input = (input_tokens * modelo_cfg["preco_input_1m"]) / 1_000_000
                    custo_output = (output_tokens * modelo_cfg["preco_output_1m"]) / 1_000_000
                    custo_execucao_usd = custo_input + custo_output

                    # Acumula nos totais globais
                    acumulado_input_tokens += input_tokens
                    acumulado_output_tokens += output_tokens
                    acumulado_custo_usd += custo_execucao_usd

                    # Nome base único por (imagem, idioma, nível de prompt, modelo).
                    # Nome do modelo sanitizado (ver _sanitizar_nome_arquivo) —
                    # nomes como "qwen3.5:4b" têm ':', reservado no Windows.
                    nome_base = f"{imagem_path.stem}_{idioma}_prompt{nivel_prompt}_{_sanitizar_nome_arquivo(modelo_cfg['nome'])}"

                    # =================================================
                    # Salva JSON individual da resposta
                    # =================================================
                    json_path = salvar_json(nome_base, resposta)

                    # =================================================
                    # Avaliação automática da resposta
                    # =================================================
                    avaliacao, detalhes_execucao = avaliar_resposta_completa(resposta, gabarito, indice)

                    # =================================================
                    # Estrutura da linha do CSV (com adição das métricas de consumo)
                    # =================================================
                    linha = _montar_linha(indice, imagem_path, idioma, nivel_prompt, modelo_cfg,
                                          tempo_execucao, input_tokens, output_tokens,
                                          custo_execucao_usd, json_path, gabarito_path, avaliacao)

                    resultados.append(linha)
                    if _foi_truncada(resposta):
                        pendentes.append({
                            "id_execucao": indice, "imagem_path": imagem_path, "idioma": idioma,
                            "nivel_prompt": nivel_prompt, "texto_prompt": texto_prompt,
                            "modelo_cfg": modelo_cfg, "gabarito": gabarito,
                            "gabarito_path": gabarito_path, "nome_base": nome_base,
                        })
                    detalhes_resultados.extend(detalhes_execucao)

                    # =================================================
                    # Salva os dois CSVs IMEDIATAMENTE após esta execução, não só
                    # no final do loop — se o processo travar, o que já rodou
                    # nesta chamada não se perde. Cada chamada de main() começa
                    # com uma lista vazia (sem carregar rodadas anteriores), então
                    # isso sempre reescreve só o arquivo desta execução, criado
                    # com o timestamp calculado no início de main().
                    # =================================================
                    df_final = pd.DataFrame(resultados)
                    caminho_saida = salvar_resultados_csv(df_final, caminho_saida)

                    df_detalhe_final = pd.DataFrame(detalhes_resultados)
                    caminho_detalhe_saida = salvar_resultados_csv(df_detalhe_final, caminho_detalhe_saida)

                    # Output detalhado no terminal
                    print(f"⏱️ Tempo: {tempo_execucao}s")
                    print(f"📥 Input Tokens:  {input_tokens:,}")
                    print(f"📤 Output Tokens: {output_tokens:,}")
                    print(f"💵 Custo estimado: ${custo_execucao_usd:.6f} USD (R$ {custo_execucao_usd * COTACAO_USD_BRL:.4f})")
                    print(f"💾 Progresso salvo em: {caminho_saida.name} (resumo) e {caminho_detalhe_saida.name} (detalhe)\n")

    # =====================================================
    # Reexecução das respostas truncadas (ver reexecutar_truncadas)
    # =====================================================
    if REEXECUTAR_TRUNCADAS and pendentes:
        gi, go, gc, recuperadas = reexecutar_truncadas(pendentes, resultados, detalhes_resultados)
        acumulado_input_tokens += gi
        acumulado_output_tokens += go
        acumulado_custo_usd += gc
        caminho_saida = salvar_resultados_csv(pd.DataFrame(resultados), caminho_saida)
        caminho_detalhe_saida = salvar_resultados_csv(pd.DataFrame(detalhes_resultados), caminho_detalhe_saida)
        print(f"🔁 Reexecuções: {recuperadas} de {len(pendentes)} truncada(s) recuperada(s).")

    # =====================================================
    # RELATÓRIO FINAL DE CONSUMO E CUSTOS DA PIPELINE
    # =====================================================
    print("\n" + "=" * 55)
    print("📊 RESUMO FINANCEIRO E DE CONSUMO DA PIPELINE")
    print("=" * 55)
    print(f" Execuções realizadas:        {total_execucoes}")
    print(f" Total Input Tokens (Prompt):  {acumulado_input_tokens:,}")
    print(f" Total Output Tokens (JSON):  {acumulado_output_tokens:,}")
    print(f" Total Geral de Tokens:       {(acumulado_input_tokens + acumulado_output_tokens):,}")
    print("-" * 55)
    print(f" 💵 Custo Total Estimado (USD): ${acumulado_custo_usd:.4f}")
    print(f" 🇧🇷 Custo Total Estimado (BRL): R$ {acumulado_custo_usd * COTACAO_USD_BRL:.2f}")
    print("=" * 55)
    print(f"\nResumo salvo com sucesso em: {caminho_saida}")
    print(f"Detalhe (item a item) salvo com sucesso em: {caminho_detalhe_saida}\n")


# =========================================================
# Executa o programa
# =========================================================
if __name__ == "__main__":

    main()
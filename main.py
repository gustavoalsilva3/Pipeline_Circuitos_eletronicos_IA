"""Pipeline principal para analisar circuitos com Gemini e avaliar as respostas.

Este módulo coordena as etapas de:
1. localizar imagens e gabaritos;
2. enviar a imagem ao modelo Gemini;
3. salvar a resposta bruta em JSON;
4. carregar o gabarito correspondente;
5. avaliar a resposta nas três dimensões (numérica, textual, componentes);
6. consolidar os resultados em um CSV para análise posterior.
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
    CSV_RESULTADOS,
    MODELO_GEMINI,
)

from prompts import (
    PROMPT_ANALISE_CIRCUITO_1,
    PROMPT_ANALISE_CIRCUITO_2,
    PROMPT_ANALISE_CIRCUITO_3,
)
from gemini_client import GeminiCircuitAnalyzer
from evaluator_numerico import avaliar_resposta as avaliar_numerico_textual
from evaluator_componentes import avaliar_componentes


# =========================================================
# Combina os três evaluators (numérico+textual via evaluator_numerico,
# componentes via evaluator_componentes) num resultado único, achatado em
# colunas planas prontas para uma linha de CSV/DataFrame.
#
# Sem isso, **avaliacao no CSV jogaria dicts aninhados direto nas colunas
# do DataFrame — cada célula viraria uma string Python ilegível em vez de
# colunas separadas (ex: numerico_Vout_status, componentes_resistores_
# taxa_acerto) que dá pra usar em groupby/pivot no pandas depois.
# =========================================================
def _achatar(dados, prefixo=""):
    """Achata um dict aninhado em chaves separadas por '_'. Listas (ex:
    'alucinados': ['R4']) viram uma string JSON compacta em vez de um
    objeto Python, para ficar legível numa célula de CSV."""

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


def avaliar_resposta_completa(resposta: dict, gabarito: dict) -> dict:
    """Avalia uma resposta contra o gabarito usando os três módulos e
    devolve um dict plano, pronto para `linha = {..., **avaliacao}`.

    Se o gabarito estiver ausente (dict vazio), NÃO avalia — evita gerar
    'incorreto'/'faltante' em tudo, o que poluiria as médias agregadas
    sem deixar claro que a causa foi gabarito ausente, não erro do modelo.
    """

    if not gabarito:
        return {"gabarito_encontrado": False}

    resultado_num_textual = avaliar_numerico_textual(resposta, gabarito)
    resultado_componentes = avaliar_componentes(resposta, gabarito)

    combinado = {
        "gabarito_encontrado": True,
        "numerico": resultado_num_textual["numerico"],
        "textual": resultado_num_textual["textual"],
        "componentes": resultado_componentes,
    }

    return _achatar(combinado)


# =========================================================
# Níveis de prompt (direto / guiado / estruturado)
# =========================================================
NIVEIS_PROMPT = {
    1: PROMPT_ANALISE_CIRCUITO_1,
    2: PROMPT_ANALISE_CIRCUITO_2,
    3: PROMPT_ANALISE_CIRCUITO_3,
}


# =========================================================
# Modelos avaliados. Cada entrada precisa de um "analisador" com o método
# analisar_imagem(image_path, prompt) -> dict, no mesmo formato do
# GeminiCircuitAnalyzer. Hoje só o Gemini está configurado; quando o
# segundo modelo (código aberto) estiver pronto, basta adicionar uma nova
# entrada aqui — nenhum outro trecho do main.py precisa mudar.
# =========================================================
def montar_modelos():
    """Instancia os analisadores de cada modelo avaliado."""
    return [
        {"nome": MODELO_GEMINI, "analisador": GeminiCircuitAnalyzer(model_name=MODELO_GEMINI)},
        # {"nome": "nome-do-modelo-aberto", "analisador": OutroAnalyzer(...)},
    ]


# =========================================================
# Cria as pastas necessárias caso não existam
# =========================================================
def preparar_pastas():
    """Cria as pastas utilizadas para guardar resultados, respostas JSON e demais artefatos."""

    RESULTADOS_DIR.mkdir(exist_ok=True)
    JSON_DIR.mkdir(exist_ok=True)


# =========================================================
# Carrega o CSV anterior de forma tolerante a arquivos
# antigos ou malformados
# =========================================================
def carregar_resultados_csv(caminho: Path) -> pd.DataFrame:
    """Lê um CSV antigo de forma tolerante a arquivos malformados ou com linhas quebradas."""
    if not caminho.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(
            caminho,
            sep=";",
            engine="python",
            on_bad_lines="skip",
            keep_default_na=False,
            dtype=str,
        )
    except Exception as exc:
        print(f"Atenção: não foi possível ler {caminho.name}: {exc}")
        print("Iniciando o CSV com um estado limpo.")
        return pd.DataFrame()


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


def _normalizar_nome_arquivo(nome: str) -> str:
    """Normaliza nomes de arquivos para comparar imagem e gabarito de maneira mais robusta."""
    if not nome:
        return ""

    nome = nome.lower()
    nome = re.sub(r"\.(json|png|jpg|jpeg|webp)$", "", nome)
    nome = re.sub(r"[^a-z0-9]+", " ", nome)

    tokens = []
    for token in nome.split():
        token = token.strip()
        if not token:
            continue

        for prefixo in ["resposta", "gabarito", "circuito", "img", "image", "arquivo"]:
            if token.startswith(prefixo):
                token = token[len(prefixo):]
                break

        token = re.sub(r"\d+$", "", token)
        if token:
            tokens.append(token)

    return " ".join(tokens)


def _nomes_batem(imagem_path: Path, arquivo_gabarito: Path) -> bool:
    """Compara nome da imagem e do arquivo de gabarito usando regras flexíveis de correspondência."""
    imagem_stem = imagem_path.stem.lower()
    nome_gabarito = arquivo_gabarito.stem.lower()

    if imagem_stem == nome_gabarito:
        return True

    imagem_norm = _normalizar_nome_arquivo(imagem_stem)
    gabarito_norm = _normalizar_nome_arquivo(nome_gabarito)

    if imagem_norm and gabarito_norm:
        if imagem_norm == gabarito_norm:
            return True
        if imagem_norm in gabarito_norm or gabarito_norm in imagem_norm:
            return True

        tokens_imagem = set(imagem_norm.split())
        tokens_gabarito = set(gabarito_norm.split())
        if tokens_imagem & tokens_gabarito:
            return True

    return any(parte in nome_gabarito for parte in imagem_stem.split("_"))


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
# Programa principal
# =========================================================
def main():
    """Executa o fluxo completo: análise da imagem, avaliação e exportação dos resultados."""

    preparar_pastas()

    modelos = montar_modelos()
    imagens = listar_imagens()

    # Lista que armazenará todas as linhas do CSV
    resultados = []

    total_execucoes = len(imagens) * len(NIVEIS_PROMPT) * len(modelos)
    print(f"\nTotal de imagens: {len(imagens)} | níveis de prompt: {len(NIVEIS_PROMPT)} "
          f"| modelos: {len(modelos)} | total de execuções: {total_execucoes}\n")

    indice = 0

    # =====================================================
    # Loop principal: imagem x nível de prompt x modelo
    # =====================================================
    for imagem_path in imagens:

        # Carrega o gabarito uma vez por imagem (não muda entre prompts/modelos)
        gabarito, gabarito_path = carregar_gabarito(imagem_path)
        if not gabarito:
            print(f"Atenção: gabarito não encontrado para {imagem_path.name} — "
                  f"todas as execuções desta imagem serão salvas com "
                  f"'gabarito_encontrado=False' e sem métricas.")

        for nivel_prompt, texto_prompt in NIVEIS_PROMPT.items():
            for modelo_cfg in modelos:

                indice += 1

                print("=" * 50)
                print(f"Execução {indice}/{total_execucoes}")
                print(f"Imagem: {imagem_path.name} | Prompt: nível {nivel_prompt} | Modelo: {modelo_cfg['nome']}")
                print("=" * 50)

                inicio = time.time()

                # =================================================
                # Envia imagem para o modelo
                # =================================================
                resposta = modelo_cfg["analisador"].analisar_imagem(
                    image_path=imagem_path,
                    prompt=texto_prompt,
                )

                tempo_execucao = round(time.time() - inicio, 2)

                # Nome base único por (imagem, nível de prompt, modelo) —
                # sem isso, execuções diferentes sobrescreveriam o mesmo
                # arquivo JSON.
                nome_base = f"{imagem_path.stem}_prompt{nivel_prompt}_{modelo_cfg['nome']}"

                # =================================================
                # Salva JSON individual da resposta
                # =================================================
                json_path = salvar_json(nome_base, resposta)

                # =================================================
                # Avaliação automática da resposta (numérico + textual + componentes)
                # =================================================
                avaliacao = avaliar_resposta_completa(resposta, gabarito)

                # =================================================
                # Estrutura da linha do CSV
                # =================================================
                linha = {

                    "id_execucao": indice,

                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

                    "imagem": imagem_path.name,

                    "modelo": modelo_cfg["nome"],
                    "nivel_prompt": nivel_prompt,

                    "tempo_execucao_s": tempo_execucao,

                    "arquivo_json": str(json_path),
                    "arquivo_gabarito": str(gabarito_path) if gabarito_path else "",

                    # Resposta bruta em texto
                    "resposta_modelo": json.dumps(resposta, ensure_ascii=False),

                    # Métricas automáticas (numérico + textual + componentes, já achatadas)
                    **avaliacao
                }

                resultados.append(linha)

                print(f"Tempo: {tempo_execucao}s")
                print("Resposta salva com sucesso.\n")

    # =====================================================
    # Cria DataFrame das respostas atuais
    # =====================================================
    df_novo = pd.DataFrame(resultados)

    # =====================================================
    # Se CSV já existir:
    # adiciona novas linhas sem sobrescrever
    # =====================================================
    if CSV_RESULTADOS.exists():

        df_antigo = carregar_resultados_csv(CSV_RESULTADOS)

        if df_antigo.empty:
            df_final = df_novo
        else:
            df_final = pd.concat(
                [df_antigo, df_novo],
                ignore_index=True,
                sort=False
            )

    else:

        df_final = df_novo

    # =====================================================
    # Salva CSV final
    # =====================================================
    caminho_saida = salvar_resultados_csv(df_final, CSV_RESULTADOS)

    print("=" * 50)
    print("PROCESSAMENTO FINALIZADO")
    print("=" * 50)

    print(f"Resultados salvos em:")
    print(caminho_saida)


# =========================================================
# Executa o programa
# =========================================================
if __name__ == "__main__":

    main()
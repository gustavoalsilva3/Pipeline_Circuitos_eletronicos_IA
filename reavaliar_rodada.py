"""Reavalia as respostas JSON já salvas de uma rodada, sem chamar nenhum
modelo, e grava um novo par de CSVs (resumo + detalhe).

Serve para dois usos:

  1. refazer os números do TFC a partir dos dados versionados, sem precisar de
     chave de API nenhuma:        python reavaliar_rodada.py
  2. aplicar uma correção de gabarito ou de avaliador a uma rodada que está em
     resultados/:                 python reavaliar_rodada.py 20261010_144312

Os metadados de cada execução (tempo, tokens, custo, provedor, quantização,
ids) são mantidos da rodada de origem; só as colunas de avaliação são
recalculadas, a partir do JSON bruto e do gabarito atual. O par de CSVs novo
vai sempre para resultados/ — a rodada oficial é só leitura.
"""

import sys
from datetime import datetime

import pandas as pd

from config import (CSV_RODADA_OFICIAL, GABARITOS_DIR, JSON_DIR,
                    JSON_RODADA_OFICIAL_DIR, RESULTADOS_DIR, caminho_detalhe)
from main import avaliar_resposta_completa, carregar_json, salvar_resultados_csv

# Colunas que não vêm da avaliação e são copiadas da rodada original.
METADADOS = ["id_execucao", "timestamp", "imagem", "modelo", "idioma", "nivel_prompt",
             "tempo_execucao_s", "input_tokens", "output_tokens", "custo_usd",
             "arquivo_json", "arquivo_gabarito", "reexecucoes",
             "id_execucao_original", "rodada_origem", "fonte", "provedor", "quantizacao"]


def origem(carimbo):
    """(CSV de resumo, pasta de JSONs) da rodada pedida.

    'oficial' é a pasta versionada, que acompanha o repositório e por isso não
    depende de resultados/ (ignorada pelo git). Um carimbo AAAAMMDD_HHMMSS
    aponta para a rodada correspondente em resultados/.
    """
    if carimbo == "oficial":
        return CSV_RODADA_OFICIAL, JSON_RODADA_OFICIAL_DIR
    return RESULTADOS_DIR / f"resultados_{carimbo}.csv", JSON_DIR


def main(carimbo="oficial"):
    csv_origem, json_dir = origem(carimbo)
    if not csv_origem.exists():
        raise SystemExit(f"rodada não encontrada: {csv_origem}")

    original = pd.read_csv(csv_origem, sep=";", engine="python",
                           on_bad_lines="skip", dtype=str)

    linhas, detalhes = [], []
    for _, r in original.iterrows():
        # Caminhos refeitos a partir dos diretórios atuais: os gravados no CSV
        # são absolutos e deixariam de valer se a pasta fosse movida.
        resposta = carregar_json(json_dir / r.arquivo_json.replace("\\", "/").split("/")[-1])
        gabarito = carregar_json(GABARITOS_DIR / r.arquivo_gabarito.replace("\\", "/").split("/")[-1])
        avaliacao, det = avaliar_resposta_completa(resposta, gabarito, int(r.id_execucao))
        linhas.append({**{c: r[c] for c in METADADOS if c in original.columns}, **avaliacao})
        detalhes.extend(det)

    # Mesma ordem de colunas da rodada original; colunas novas, se houver, ao fim.
    resumo = pd.DataFrame(linhas)
    resumo = resumo[[c for c in original.columns if c in resumo.columns]
                    + [c for c in resumo.columns if c not in original.columns]]

    destino = RESULTADOS_DIR / f"resultados_{datetime.now():%Y%m%d_%H%M%S}.csv"
    s = salvar_resultados_csv(resumo, destino)
    t = salvar_resultados_csv(pd.DataFrame(detalhes), caminho_detalhe(destino))
    print(f"{len(resumo)} execuções reavaliadas (origem: {carimbo})\n  {s.name}\n  {t.name}")


if __name__ == "__main__":
    main(*sys.argv[1:])

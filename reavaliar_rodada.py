"""Reavalia as respostas JSON já salvas de uma rodada consolidada, sem chamar
nenhum modelo, e grava um novo par de CSVs (resumo + detalhe).

Serve para aplicar correções de gabarito ou de avaliador a respostas que já
existem: os metadados de cada execução (tempo, tokens, custo, provedor,
quantização, ids) são mantidos da rodada original, e só as colunas de
avaliação são recalculadas a partir do JSON bruto e do gabarito atual.

Uso:
    python reavaliar_rodada.py [carimbo]    (padrão: 20261010_144312)
"""

import sys
from datetime import datetime

import pandas as pd

from config import GABARITOS_DIR, JSON_DIR, RESULTADOS_DIR
from main import avaliar_resposta_completa, carregar_json, salvar_resultados_csv

# Colunas que não vêm da avaliação e são copiadas da rodada original.
METADADOS = ["id_execucao", "timestamp", "imagem", "modelo", "idioma", "nivel_prompt",
             "tempo_execucao_s", "input_tokens", "output_tokens", "custo_usd",
             "arquivo_json", "arquivo_gabarito", "reexecucoes",
             "id_execucao_original", "rodada_origem", "fonte", "provedor", "quantizacao"]


def main(carimbo="20261010_144312"):
    original = pd.read_csv(RESULTADOS_DIR / f"resultados_{carimbo}.csv", sep=";",
                           engine="python", on_bad_lines="skip", dtype=str)

    linhas, detalhes = [], []
    for _, r in original.iterrows():
        # Caminhos refeitos a partir dos diretórios atuais: os gravados no CSV
        # são absolutos e deixariam de valer se a pasta fosse movida.
        resposta = carregar_json(JSON_DIR / r.arquivo_json.replace("\\", "/").split("/")[-1])
        gabarito = carregar_json(GABARITOS_DIR / r.arquivo_gabarito.replace("\\", "/").split("/")[-1])
        avaliacao, det = avaliar_resposta_completa(resposta, gabarito, int(r.id_execucao))
        linhas.append({**{c: r[c] for c in METADADOS if c in original.columns}, **avaliacao})
        detalhes.extend(det)

    # Mesma ordem de colunas da rodada original; colunas novas, se houver, ao fim.
    resumo = pd.DataFrame(linhas)
    resumo = resumo[[c for c in original.columns if c in resumo.columns]
                    + [c for c in resumo.columns if c not in original.columns]]

    novo = datetime.now().strftime("%Y%m%d_%H%M%S")
    s = salvar_resultados_csv(resumo, RESULTADOS_DIR / f"resultados_{novo}.csv")
    t = salvar_resultados_csv(pd.DataFrame(detalhes), RESULTADOS_DIR / f"resultados_detalhe_{novo}.csv")
    print(f"{len(resumo)} execuções reavaliadas\n  {s.name}\n  {t.name}")


if __name__ == "__main__":
    main(*sys.argv[1:])

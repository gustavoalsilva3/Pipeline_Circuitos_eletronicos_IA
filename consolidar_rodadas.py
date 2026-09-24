"""Reúne as rodadas dos quatro modelos em um único par de CSVs, que é o que
o Analise_erros.py consome (ele sempre lê o par mais recente).

Necessário porque cada chamada de main.py numera suas execuções a partir de
1: juntar os arquivos sem tratamento faria ids de rodadas diferentes
colidirem, e o CSV de detalhe — que referencia id_execucao — passaria a
apontar para a linha errada. Aqui os ids são renumerados de forma contínua e
o id antigo fica guardado em 'id_execucao_original', junto da rodada de
origem, seguindo a convenção da consolidação anterior.

Acrescenta também duas colunas de reprodutibilidade que não existiam:

  provedor    — quem executou o modelo (Google, Alibaba, Parasail, local)
  quantizacao — precisão em que o modelo foi servido

Elas importam porque a comparação entre o 8B local (Q4_K_M) e o 8B em nuvem
(bf16) é justamente o controle de quantização do trabalho; sem registrar a
precisão no CSV, o dado não é auditável a partir da planilha.

Uso:
    python consolidar_rodadas.py
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import RESULTADOS_DIR

# (resumo, detalhe) de cada rodada a reunir, na ordem em que serão numeradas
RODADAS = [
    ("resultados_20260924_082751.csv", "resultados_detalhe_20260924_082751.csv"),
    ("resultados_20260924_091447.csv", "resultados_detalhe_20260924_091447.csv"),
    ("resultados_20260924_105846.csv", "resultados_detalhe_20260924_105846.csv"),
]

# provedor/quantização por modelo. O local é o único em que a quantização é
# escolha do autor; os demais dependem de quem serve o modelo.
ORIGEM = {
    "gemini-2.5-flash": ("Google (API)", "nao divulgada"),
    "gemma-3-27b-bf16 (openrouter)": ("Novita (OpenRouter)", "bf16"),
        "qwen3-vl-32b (openrouter)": ("Alibaba (OpenRouter)", "nao declarada"),
    "qwen3-vl-8b-bf16 (openrouter)": ("Parasail (OpenRouter)", "bf16"),
}


def main():
    resumos, detalhes = [], []
    proximo_id = 1

    for arq_resumo, arq_detalhe in RODADAS:
        r = pd.read_csv(RESULTADOS_DIR / arq_resumo, sep=";", engine="python",
                        on_bad_lines="skip", dtype=str)
        d = pd.read_csv(RESULTADOS_DIR / arq_detalhe, sep=";", engine="python",
                        on_bad_lines="skip", dtype=str)
        carimbo = arq_resumo.replace("resultados_", "").replace(".csv", "")

        # renumera de forma contínua e leva o mapa para o detalhe
        r = r.reset_index(drop=True)
        novos = {velho: str(proximo_id + i)
                 for i, velho in enumerate(r["id_execucao"].tolist())}
        proximo_id += len(r)

        if "id_execucao_original" not in r.columns:
            r["id_execucao_original"] = r["id_execucao"]
        if "rodada_origem" not in r.columns:
            r["rodada_origem"] = carimbo
        r["rodada_origem"] = r["rodada_origem"].fillna(carimbo)
        if "fonte" not in r.columns:
            r["fonte"] = "json_reavaliado"

        r["id_execucao"] = r["id_execucao"].map(novos)
        d["id_execucao"] = d["id_execucao"].map(novos)
        faltando = d["id_execucao"].isna().sum()
        if faltando:
            raise SystemExit(f"{faltando} linhas de detalhe de {arq_detalhe} sem id correspondente.")

        resumos.append(r)
        detalhes.append(d)

    resumo = pd.concat(resumos, ignore_index=True)
    detalhe = pd.concat(detalhes, ignore_index=True)

    desconhecidos = set(resumo["modelo"]) - set(ORIGEM)
    if desconhecidos:
        raise SystemExit(f"Modelos sem provedor/quantizacao definidos: {desconhecidos}")
    resumo["provedor"] = resumo["modelo"].map(lambda m: ORIGEM[m][0])
    resumo["quantizacao"] = resumo["modelo"].map(lambda m: ORIGEM[m][1])

    carimbo = datetime.now().strftime("%Y%m%d_%H%M%S")
    p_res = RESULTADOS_DIR / f"resultados_{carimbo}.csv"
    p_det = RESULTADOS_DIR / f"resultados_detalhe_{carimbo}.csv"
    resumo.to_csv(p_res, sep=";", index=False, encoding="utf-8-sig")
    detalhe.to_csv(p_det, sep=";", index=False, encoding="utf-8-sig")

    print(f"resumo : {p_res.name}  ({len(resumo)} execucoes)")
    print(f"detalhe: {p_det.name}  ({len(detalhe)} itens)")
    print(f"\nids: {resumo.id_execucao.astype(int).min()} a {resumo.id_execucao.astype(int).max()}"
          f" | unicos: {resumo.id_execucao.nunique()}")
    print("\npor modelo:")
    for m, n in resumo["modelo"].value_counts().items():
        print(f"   {m:<32} {n:>3}  [{ORIGEM[m][0]}, {ORIGEM[m][1]}]")
    orfas = set(detalhe.id_execucao) - set(resumo.id_execucao)
    print(f"\nlinhas de detalhe orfas: {len(orfas)}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Intervalos de confiança (bootstrap por circuito) das taxas de acerto numérico.

Gera TFC/tab_intervalos.tex, a tabela da seção 'Incerteza das Estimativas'."""
import os
import pandas as pd, numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(BASE, "resultados", "resultados_20260922_202645.csv"),
                sep=";", engine="python", on_bad_lines="skip", dtype=str)
d["circ"] = d["imagem"].str.replace(r"\.\w+$", "", regex=True)
d["fam"] = d["imagem"].str.extract(r"(Amp|Dio|RLC)", expand=False).map({"Amp": "Amplificador operacional", "Dio": "Diodo", "RLC": "RLC"})
cols = {"Tensões": "numerico_tensoes_taxa_acerto_percentual",
        "Correntes": "numerico_correntes_taxa_acerto_percentual",
        "Potências": "numerico_potencias_taxa_acerto_percentual"}
for c in cols.values():
    d[c] = pd.to_numeric(d[c].str.replace(",", "."), errors="coerce")
d["idioma"] = d["idioma"].map({"pt": "Português", "en": "Inglês"}).fillna(d["idioma"])

B = 5000
rng = np.random.default_rng(42)


def ic(sub, col):
    circs = sub["circ"].unique()
    grp = {c: sub.loc[sub["circ"] == c, col].dropna().values for c in circs}
    meds = []
    for _ in range(B):
        amostra = rng.choice(circs, len(circs))
        v = np.concatenate([grp[c] for c in amostra])
        if len(v):
            meds.append(v.mean())
    lo, hi = np.percentile(meds, [2.5, 97.5])
    m = sub[col].dropna().mean()
    return m, lo, hi


def fmt(x):
    return f"{x:.1f}".replace(".", ",")


def celula(sub, col):
    m, lo, hi = ic(sub, col)
    return f"{fmt(m)} [{fmt(lo)}; {fmt(hi)}]"


modelos = [("gemini-2.5-flash", "Gemini 2.5 Flash"), ("qwen3-vl-32b (openrouter)", "Qwen3-VL 32B"), ("qwen/qwen3-vl-8b", "Qwen3-VL 8B (local, Q4\_K\_M)")]
grupos = [("Geral", lambda x: x),
          ("Nível 1 --- Direto", lambda x: x[x.nivel_prompt.astype(str) == "1"]),
          ("Nível 2 --- Guiado", lambda x: x[x.nivel_prompt.astype(str) == "2"]),
          ("Nível 3 --- Estruturado", lambda x: x[x.nivel_prompt.astype(str) == "3"]),
          ("Português", lambda x: x[x.idioma == "Português"]),
          ("Inglês", lambda x: x[x.idioma == "Inglês"]),
          ("Amplificador operacional", lambda x: x[x.fam == "Amplificador operacional"]),
          ("Diodo", lambda x: x[x.fam == "Diodo"]),
          ("RLC", lambda x: x[x.fam == "RLC"])]

linhas = []
for i, (mid, nome) in enumerate(modelos):
    dm = d[d.modelo == mid]
    for j, (gn, f) in enumerate(grupos):
        sub = f(dm)
        cel = [celula(sub, c) for c in cols.values()]
        primeiro = nome if j == 0 else ""
        linhas.append(f"\t\t{primeiro} & {gn} & " + " & ".join(cel) + r" \\")
        if gn in ("Geral", "Nível 3 --- Estruturado", "Inglês"):
            linhas.append(r"\cmidrule{2-5}")
    if i == 0:
        linhas.append(r"\midrule")
linhas = [l for l in linhas]
# remove o último cmidrule redundante antes de bottomrule não necessário

tex = r"""\begin{table}[H]
	\centering
	\caption{Taxa média de acerto (\%) e intervalo de confiança de 95\% por \textit{bootstrap} sobre os circuitos, por modelo e subconjunto.}
	\small
	\begin{tabular}{l l c c c}
		\toprule
		Modelo & Subconjunto & Tensões & Correntes & Potências \\
		\midrule
""" + "\n".join(linhas) + r"""
		\bottomrule
	\end{tabular}
	\label{tab:intervalos}
\end{table}
"""
open(os.path.join(BASE, "TFC", "tab_intervalos.tex"), "w", encoding="utf-8").write(tex)
print(tex)

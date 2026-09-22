"""Controle de quantização: Qwen3-VL 8B em Q4_K_M (local) x bf16 (nuvem).

É o mesmo modelo, com os mesmos prompts e circuitos, mudando apenas a
precisão em que os pesos são servidos. Serve para medir — e não apenas
declarar — quanto a quantização de 4 bits custa, já que o desempenho do
modelo local só pode ser lido como representativo dos pesos originais se
essa diferença for pequena.

Usa o mesmo bootstrap por circuito da seção de incerteza: reamostra os 24
circuitos com reposição, porque as seis execuções de um mesmo circuito
(dois idiomas x três níveis de prompt) não são independentes.

Gera TFC/tab_quantizacao.tex.
"""

import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(BASE, "resultados", "resultados_20260922_202645.csv")
LOCAL, NUVEM = "qwen/qwen3-vl-8b", "qwen3-vl-8b-bf16 (openrouter)"
B, SEMENTE = 5000, 42

METRICAS = [
    ("numerico_tensoes_taxa_acerto_percentual", "Tensões (\\%)"),
    ("numerico_correntes_taxa_acerto_percentual", "Correntes (\\%)"),
    ("numerico_potencias_taxa_acerto_percentual", "Potências (\\%)"),
    ("textual_regiao_operacao_taxa_acerto_percentual", "Região de operação (\\%)"),
]

d = pd.read_csv(CSV, sep=";", engine="python", on_bad_lines="skip", dtype=str)
d["circ"] = d["imagem"].str.replace(r"\.\w+$", "", regex=True)
for c, _ in METRICAS:
    d[c] = pd.to_numeric(d[c].astype(str).str.replace(",", "."), errors="coerce")
d["tv"] = d["textual_terra_virtual_acerto"].astype(str).str.lower().isin(["true", "1"]) * 100.0

rng = np.random.default_rng(SEMENTE)


def ic(sub, col):
    circs = sub["circ"].unique()
    grp = {c: sub.loc[sub["circ"] == c, col].dropna().values for c in circs}
    meds = []
    for _ in range(B):
        v = np.concatenate([grp[c] for c in rng.choice(circs, len(circs))])
        if len(v):
            meds.append(v.mean())
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return sub[col].dropna().mean(), lo, hi


def ic_dif(a, b, col):
    """IC da DIFERENÇA (bf16 - Q4), reamostrando os MESMOS circuitos nos dois
    braços — é um pareamento: cada circuito entra ou sai das duas amostras
    junto, o que remove a variação entre circuitos da estimativa."""
    circs = sorted(set(a["circ"]) & set(b["circ"]))
    ga = {c: a.loc[a["circ"] == c, col].dropna().values for c in circs}
    gb = {c: b.loc[b["circ"] == c, col].dropna().values for c in circs}
    difs = []
    for _ in range(B):
        am = rng.choice(circs, len(circs))
        va = np.concatenate([ga[c] for c in am])
        vb = np.concatenate([gb[c] for c in am])
        if len(va) and len(vb):
            difs.append(vb.mean() - va.mean())
    return np.mean(difs), *np.percentile(difs, [2.5, 97.5])


def f(x):
    return f"{x:+.1f}".replace(".", ",") if abs(x) < 1000 else str(x)


def g(x):
    return f"{x:.1f}".replace(".", ",")


q4, bf = d[d.modelo == LOCAL], d[d.modelo == NUVEM]
print(f"Q4_K_M: {len(q4)} execucoes | bf16: {len(bf)} execucoes\n")
print(f"{'metrica':<26}{'Q4_K_M (local)':>24}{'bf16 (nuvem)':>24}{'diferenca bf16-Q4':>26}")

linhas = []
for col, rot in METRICAS + [("tv", "Terra virtual (\\%)")]:
    ma, loa, hia = ic(q4, col)
    mb, lob, hib = ic(bf, col)
    md, lod, hid = ic_dif(q4, bf, col)
    zero_dentro = lod <= 0 <= hid
    print(f"{rot.replace(chr(92)+'%','%'):<26}{g(ma)+' ['+g(loa)+'; '+g(hia)+']':>24}"
          f"{g(mb)+' ['+g(lob)+'; '+g(hib)+']':>24}"
          f"{f(md)+' ['+f(lod)+'; '+f(hid)+']':>26}"
          f"{'  (inclui 0)' if zero_dentro else '  (NAO inclui 0)'}")
    linhas.append(f"\t\t{rot} & {g(ma)} [{g(loa)}; {g(hia)}] & {g(mb)} [{g(lob)}; {g(hib)}] & "
                  f"{f(md)} [{f(lod)}; {f(hid)}] \\\\")

tex = (r"""\begin{table}[H]
	\centering
	\caption{Controle de quantização: Qwen3-VL 8B servido em Q4\_K\_M (execução local) e em bf16 (execução em nuvem), com intervalos de confiança de 95\% por \textit{bootstrap} sobre os circuitos. A última coluna traz o intervalo da diferença, obtido com reamostragem pareada dos mesmos circuitos nos dois braços.}
	\small
	\begin{tabular}{l c c c}
		\toprule
		Métrica & Q4\_K\_M (local) & bf16 (nuvem) & Diferença (bf16 $-$ Q4\_K\_M) \\
		\midrule
""" + "\n".join(linhas) + r"""
		\bottomrule
	\end{tabular}
	\label{tab:quantizacao}
\end{table}
""")
open(os.path.join(BASE, "TFC", "tab_quantizacao.tex"), "w", encoding="utf-8").write(tex)
print("\nTFC/tab_quantizacao.tex gerado.")

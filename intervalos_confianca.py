# -*- coding: utf-8 -*-
"""Intervalos de confiança (bootstrap por circuito) das taxas de acerto numérico.

Gera TFC/tab_intervalos.tex, a tabela da seção 'Incerteza das Estimativas'."""
import argparse
import os
import pandas as pd, numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))

# Mesma origem versionada de gerar_tabelas_cap4.py, pelo mesmo motivo. Com
# --carimbo le uma rodada de resultados/ e escreve com o carimbo no nome do
# arquivo, para que tab_intervalos.tex e tab_pareado.tex do texto nao sejam
# sobrescritos por uma rodada diferente da oficial.
_ap = argparse.ArgumentParser(
    description="Intervalos de confianca das taxas de acerto numerico.")
_ap.add_argument("--carimbo", default=None,
                 help="carimbo de uma rodada em resultados/ "
                      "(AAAAMMDD_HHMMSS); sem isso usa a rodada oficial")
_args = _ap.parse_args()

if _args.carimbo:
    CARIMBO = _args.carimbo
    FONTE = os.path.join(BASE, "resultados", f"resultados_{CARIMBO}.csv")
    SUF = f"_{CARIMBO}"
else:
    CARIMBO = "20261010_144312"
    FONTE = os.path.join(BASE, "rodada_oficial", "resultados.csv")
    SUF = ""

d = pd.read_csv(FONTE, sep=";", engine="python",
                on_bad_lines="skip", dtype=str)
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


# Ordem de referencia; os modelos ausentes da rodada caem fora na linha
# seguinte, de modo que a rodada oficial gere exatamente a mesma tabela.
MODELOS_BASE = [("gemini-2.5-flash", "Gemini 2.5 Flash"),
                ("gemini-2.5-flash (sem raciocinio)",
                 "Gemini sem raciocínio"),
                ("qwen3-vl-32b (openrouter)", "Qwen3-VL 32B"),
                ("gemma-3-27b-bf16 (openrouter)", "Gemma 3 27B"),
                ("qwen3-vl-8b-bf16 (openrouter)", "Qwen3-VL 8B")]
_presentes = set(d["modelo"].dropna())
modelos = [(mid, nome) for mid, nome in MODELOS_BASE if mid in _presentes]
# Uma linha por modelo, sobre o conjunto completo das execucoes. Os
# subconjuntos por nivel de prompt, idioma e familia saem daqui: as
# comparacoes entre eles sao feitas pela tabela pareada, que mede o intervalo
# da propria diferenca em vez de depender da sobreposicao de dois intervalos.
linhas = []
for mid, nome in modelos:
    dm = d[d.modelo == mid]
    cel = [celula(dm, c) for c in cols.values()]
    linhas.append(f"\t\t{nome} & " + " & ".join(cel) + r" \\")

tex = r"""\begin{table}[H]
	\centering
	\caption{Taxa média de acerto (\%) e intervalo de confiança de 95\% por \textit{bootstrap} sobre os circuitos, por modelo, no conjunto completo das execuções.}
	\footnotesize
	\begin{tabular}{l c c c}
		\toprule
		Modelo & Tensões & Correntes & Potências \\
		\midrule
""" + "\n".join(linhas) + r"""
		\bottomrule
	\end{tabular}
	\label{tab:intervalos}
\end{table}
"""
open(os.path.join(BASE, "TFC", f"tab_intervalos{SUF}.tex"), "w",
     encoding="utf-8").write(tex)
print(tex)


# =========================================================
# Intervalo PAREADO da diferenca entre dois subconjuntos
# =========================================================
# O criterio de sobreposicao usado acima e conservador: para dois intervalos
# se separarem, a distancia entre as medias precisa superar a SOMA das duas
# margens. Aqui mede-se diretamente a diferenca, cuja margem e menor. Os dois
# subconjuntos sao reamostrados sobre os MESMOS circuitos, o que ainda cancela
# parte da variacao comum.
def ic_pareado(sub_a, sub_b, col):
    circs = sorted(set(sub_a["circ"]) & set(sub_b["circ"]))
    ga = {c: sub_a.loc[sub_a["circ"] == c, col].dropna().values for c in circs}
    gb = {c: sub_b.loc[sub_b["circ"] == c, col].dropna().values for c in circs}
    circs = [c for c in circs if len(ga[c]) and len(gb[c])]
    if not circs:
        return None
    r = np.random.default_rng(42)
    difs = np.empty(B)
    for k in range(B):
        am = r.choice(circs, size=len(circs), replace=True)
        difs[k] = (np.concatenate([ga[c] for c in am]).mean()
                   - np.concatenate([gb[c] for c in am]).mean())
    obs = (np.concatenate([ga[c] for c in circs]).mean()
           - np.concatenate([gb[c] for c in circs]).mean())
    lo, hi = np.percentile(difs, [2.5, 97.5])
    return obs, lo, hi


NIVEIS = {"1": "Direto", "2": "Guiado", "3": "Estruturado"}
ORDEM_M = [nome for _, nome in modelos]
d["m"] = d["modelo"].map(dict(modelos))

comparacoes = [
    ("Porte", "Qwen3-VL 32B $-$ Qwen3-VL 8B",
     d[d.m == "Qwen3-VL 32B"], d[d.m == "Qwen3-VL 8B"]),
    ("Família", "Qwen3-VL 32B $-$ Gemma 3 27B",
     d[d.m == "Qwen3-VL 32B"], d[d.m == "Gemma 3 27B"]),
    # Porte e familia variam juntos neste par: e ele que testa se o porte
    # prediz desempenho ENTRE familias (27,4 bilhoes contra 8).
    ("Porte entre famílias", "Gemma 3 27B $-$ Qwen3-VL 8B",
     d[d.m == "Gemma 3 27B"], d[d.m == "Qwen3-VL 8B"]),
]

# Raciocinio interno: so existe quando a rodada traz as duas variantes do
# mesmo Gemini. E o unico par em que modelo, provedor e prompt sao iguais e
# so o raciocinio muda, de modo que ele isola esse fator.
if {"Gemini 2.5 Flash", "Gemini sem raciocínio"} <= set(ORDEM_M):
    comparacoes.insert(0, ("Raciocínio", "Gemini com $-$ Gemini sem",
                           d[d.m == "Gemini 2.5 Flash"],
                           d[d.m == "Gemini sem raciocínio"]))
for m in ORDEM_M:
    dm = d[d.m == m]
    for i, j in (("1", "2"), ("1", "3"), ("2", "3")):
        comparacoes.append((f"\\textit{{Prompt}}: {m}",
                            f"{NIVEIS[i]} $-$ {NIVEIS[j]}",
                            dm[dm.nivel_prompt == i], dm[dm.nivel_prompt == j]))
for m in ORDEM_M:
    dm = d[d.m == m]
    comparacoes.append((f"Idioma: {m}", "Português $-$ Inglês",
                        dm[dm.idioma == "Português"], dm[dm.idioma == "Inglês"]))

linhas_p, eixo_ant, n_dem, n_tot = [], None, 0, 0
for eixo, rot, a, b in comparacoes:
    cel = []
    for _, col in cols.items():
        r = ic_pareado(a, b, col)
        if r is None:
            cel.append("---")
            continue
        o, lo, hi = r
        n_tot += 1
        forte = lo > 0 or hi < 0
        n_dem += forte
        txt = (f"{o:+.1f} [{lo:+.1f}; {hi:+.1f}]").replace(".", ",")
        cel.append("\\textbf{" + txt + "}" if forte else txt)
    primeiro = eixo if eixo != eixo_ant else ""
    if eixo != eixo_ant and eixo_ant is not None:
        linhas_p.append("\\cmidrule{2-5}")
    eixo_ant = eixo
    linhas_p.append(f"{primeiro} & {rot} & " + " & ".join(cel) + r" \\")

tab_p = ("\\begin{table}[H]\n\t\\centering\n"
         "\t\\caption{Intervalos de confiança de 95\\% para a diferença entre subconjuntos, "
         "obtidos por reamostragem pareada dos mesmos circuitos. Em negrito, os intervalos que "
         "não contêm zero.}\n\t\\scriptsize\n\t\\setlength{\\tabcolsep}{2pt}\n"
         "\t\\begin{tabular}{>{\\raggedright\\arraybackslash}p{3.0cm} l c c c}\n\t\t\\toprule\n"
         "\t\tEixo & Comparação & Tensões & Correntes & Potências \\\\\n\t\t\\midrule\n"
         + "\n".join("\t\t" + l for l in linhas_p)
         + "\n\t\t\\bottomrule\n\t\\end{tabular}\n\t\\label{tab:pareado}\n\\end{table}\n")
open(os.path.join(BASE, "TFC", f"tab_pareado{SUF}.tex"), "w",
     encoding="utf-8").write(tab_p)
print(f"gerado: TFC/tab_pareado{SUF}.tex  "
      f"({n_dem} de {n_tot} intervalos excluem zero)")


# =========================================================
# Grafico dos intervalos, para o texto
#
# Ponto com intervalo, e nao barra: a barra sugere area a partir de zero e
# nao tem onde mostrar a incerteza, que e o assunto da figura. Uma cor so,
# porque o eixo ja nomeia cada configuracao e pintar cada linha de um tom
# gastaria o canal de cor com o que o rotulo ja diz.
# =========================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AZUL = "#2a78d6"
fig, eixos = plt.subplots(1, 3, figsize=(7.1, 2.7), sharey=True, sharex=True)  # ~largura do texto,
                                                # para a figura nao ser
                                                # reduzida no documento
for ax, (titulo, col) in zip(eixos, cols.items()):
    ys = list(range(len(modelos)))[::-1]
    for y, (mid, nome) in zip(ys, modelos):
        m_, lo, hi = ic(d[d["modelo"] == mid], col)
        ax.plot([lo, hi], [y, y], color=AZUL, linewidth=2, solid_capstyle="round")
        ax.plot([m_], [y], "o", color=AZUL, markersize=7,
                markeredgecolor="white", markeredgewidth=1.2, zorder=3)
        ax.annotate(f"{m_:.1f}".replace(".", ","), (m_, y), fontsize=8,
                    textcoords="offset points", xytext=(0, 8), ha="center",
                    color="#333333")
    ax.set_title(titulo, fontsize=11, fontweight="bold")
    ax.set_xlim(0, 100)
    # folga no topo: sem ela o rotulo da primeira linha encosta no titulo
    ax.set_ylim(-0.7, len(modelos) - 1 + 0.9)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Taxa de acerto (%)", fontsize=9)
eixos[0].set_yticks(list(range(len(modelos)))[::-1])
eixos[0].set_yticklabels([nome for _, nome in modelos], fontsize=9)
fig.tight_layout()
dest = os.path.join(BASE, "TFC", "figuras", "comparativo", "intervalos.png")
os.makedirs(os.path.dirname(dest), exist_ok=True)
fig.savefig(dest, dpi=150)
plt.close(fig)
print(f"gerado: {dest}")

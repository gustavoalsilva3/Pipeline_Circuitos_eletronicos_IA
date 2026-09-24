# -*- coding: utf-8 -*-
"""Gera as tres figuras comparativas do Capitulo 4, com os quatro modelos
lado a lado, a partir do CSV consolidado da rodada oficial.

As demais figuras do capitulo saem do Analise_erros.py, que trabalha um
modelo por vez; estas precisam dos quatro no mesmo eixo e por isso sao
geradas aqui. Escreve em TFC/figuras/comparativo/:

    taxa_acerto_por_prompt_modelos.png
    taxa_acerto_por_idioma_modelos.png
    taxa_acerto_por_familia_modelos.png
"""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(BASE, "resultados", "resultados_20260924_121301.csv")
SAIDA = os.path.join(BASE, "TFC", "figuras", "comparativo")

MODELOS = [
    ("gemini-2.5-flash", "Gemini 2.5 Flash"),
    ("qwen3-vl-32b (openrouter)", "Qwen3-VL 32B"),
    ("gemma-3-27b-bf16 (openrouter)", "Gemma 3 27B"),
    ("qwen3-vl-8b-bf16 (openrouter)", "Qwen3-VL 8B"),
]
COLUNAS = {
    "Tensões": "numerico_tensoes_taxa_acerto_percentual",
    "Correntes": "numerico_correntes_taxa_acerto_percentual",
    "Potências": "numerico_potencias_taxa_acerto_percentual",
}
CORES = ["#4C72B0", "#DD8452", "#55A868"]

d = pd.read_csv(CSV, sep=";", engine="python", on_bad_lines="skip", dtype=str)
d["fam"] = d["imagem"].str.extract(r"(Amp|Dio|RLC)", expand=False).map(
    {"Amp": "Amp. operacional", "Dio": "Diodo", "RLC": "RLC"})
# o CSV usa virgula decimal; sem a troca, to_numeric devolveria NaN
for c in COLUNAS.values():
    d[c] = pd.to_numeric(d[c].str.replace(",", "."), errors="coerce")
d["idioma"] = d["idioma"].map({"pt": "Português", "en": "Inglês"}).fillna(d["idioma"])


def grafico(chave, ordem, rotulo_x, arquivo, titulo):
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.4), sharey=True)
    for ax, (mid, nome) in zip(axs, MODELOS):
        sub = d[d.modelo == mid]
        x = np.arange(len(ordem))
        w = 0.26
        for i, (rot, col) in enumerate(COLUNAS.items()):
            vals = [sub[sub[chave].astype(str) == str(o)][col].mean() for o in ordem]
            ax.bar(x + (i - 1) * w, vals, w, label=rot, color=CORES[i])
            for xi, v in zip(x + (i - 1) * w, vals):
                ax.text(xi, v + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([str(o) for o in ordem])
        ax.set_xlabel(rotulo_x)
        ax.set_title(nome, fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel("Taxa de acerto (%)")
    axs[0].set_ylim(0, 100)
    axs[3].legend(loc="upper right", fontsize=8)
    fig.suptitle(titulo, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(SAIDA, arquivo), dpi=150)
    plt.close(fig)
    print("  " + arquivo)


os.makedirs(SAIDA, exist_ok=True)
print(f"lendo {os.path.basename(CSV)} ({len(d)} execuções)")
grafico("nivel_prompt", [1, 2, 3], "Nível de prompt",
        "taxa_acerto_por_prompt_modelos.png",
        "Taxa de acerto numérico por nível de prompt")
grafico("idioma", ["Português", "Inglês"], "Idioma do prompt",
        "taxa_acerto_por_idioma_modelos.png",
        "Taxa de acerto numérico por idioma do prompt")
grafico("fam", ["Amp. operacional", "Diodo", "RLC"], "Família de circuito",
        "taxa_acerto_por_familia_modelos.png",
        "Taxa de acerto numérico por família de circuito")

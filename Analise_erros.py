"""Análise automática de erros da pipeline.

Lê resultados.csv (resumo, uma linha por execução) e resultados_detalhe.csv
(detalhe item a item) e gera tabelas (CSV) + gráficos (PNG) resumindo o
desempenho do modelo por nível de prompt, por família de circuito, e os
piores casos — para usar direto na análise do TCC sem precisar recalcular
tudo manualmente a cada rodada da pipeline.

Uso:
    python Analise_erros.py

Saída (em resultados/analise/):
    tabelas/*.csv   — resumo_por_prompt, resumo_por_familia, resumo_por_idioma,
                       piores_vout, componentes_por_categoria,
                       componentes_por_idioma, status_geral_detalhe,
                       status_por_categoria_detalhe, status_geral_por_idioma,
                       status_por_categoria_por_idioma
    graficos/*.png  — taxa_acerto_por_prompt, taxa_acerto_por_familia,
                       taxa_acerto_por_idioma, regiao_operacao_por_prompt,
                       regiao_operacao_por_idioma, terra_virtual_por_idioma,
                       status_geral, status_por_categoria,
                       status_geral_por_idioma, status_por_categoria_por_idioma,
                       componentes_por_idioma

Tudo que termina em '_por_idioma' compara PT x EN — só é gerado se a coluna
'idioma' estiver presente no CSV (rodadas feitas com
config.IDIOMA_PROMPT = "ambos") e houver mais de um idioma nos dados. O
detalhe (dfd) não tem 'idioma' própria — carregar_dados() a traz do resumo
via id_execucao, para permitir separar por idioma também as métricas de
'status' item a item e de identificação de componentes.
"""

import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import PREFIXO_CSV_RESULTADOS, PREFIXO_CSV_DETALHES, RESULTADOS_DIR

PASTA_ANALISE = RESULTADOS_DIR / "analise"
PASTA_TABELAS = PASTA_ANALISE / "tabelas"
PASTA_GRAFICOS = PASTA_ANALISE / "graficos"

CORES_PROMPT = ["#4C72B0", "#DD8452", "#55A868"]
CATEGORIAS_COMPONENTES = [
    "amplificadores_operacionais", "resistores", "fontes",
    "diodos", "capacitores", "indutores",
]


# =========================================================
# Cada rodada de main.py gera um par de arquivos NOVO, com o timestamp da
# execução no nome (ex: "resultados_20260912_153045.csv") — por padrão,
# analisamos sempre a rodada mais recente. O regex ancora exatamente no
# formato AAAAMMDD_HHMMSS pra não confundir "resultados_*" com
# "resultados_detalhe_*" (que também começa com "resultados_").
# =========================================================
def _timestamp_mais_recente(prefixo):
    padrao = re.compile(rf"^{re.escape(prefixo)}_(\d{{8}}_\d{{6}})\.csv$")
    timestamps = [m.group(1) for m in (padrao.match(p.name) for p in RESULTADOS_DIR.glob("*.csv")) if m]
    if not timestamps:
        raise FileNotFoundError(
            f"Nenhum arquivo '{prefixo}_AAAAMMDD_HHMMSS.csv' encontrado em {RESULTADOS_DIR} "
            "— rode main.py pelo menos uma vez antes."
        )
    return max(timestamps)  # formato AAAAMMDD_HHMMSS ordena cronologicamente como string


def _carregar_csv(caminho):
    if not caminho.exists():
        raise FileNotFoundError(f"{caminho} não existe.")
    return pd.read_csv(caminho, sep=";", engine="python", on_bad_lines="skip", dtype=str)


def _para_numerico(df, colunas):
    for c in colunas:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    return df


def carregar_dados(timestamp=None):
    """Carrega resumo e detalhe da rodada mais recente (ou de 'timestamp',
    se informado, no formato AAAAMMDD_HHMMSS), já com as colunas numéricas
    convertidas e a coluna 'familia' derivada do nome da imagem (prefixo
    antes do número)."""
    if timestamp is None:
        timestamp = _timestamp_mais_recente(PREFIXO_CSV_RESULTADOS)

    caminho_resumo = RESULTADOS_DIR / f"{PREFIXO_CSV_RESULTADOS}_{timestamp}.csv"
    caminho_detalhe = RESULTADOS_DIR / f"{PREFIXO_CSV_DETALHES}_{timestamp}.csv"
    print(f"Analisando a rodada: {timestamp}")

    df = _carregar_csv(caminho_resumo)
    dfd = _carregar_csv(caminho_detalhe)

    colunas_numericas = [c for c in df.columns if any(
        chave in c for chave in [
            "taxa_acerto_percentual", "erro_percentual", "erro_absoluto", "RMSE", "MAE",
            "ERM_percentual", "precision", "recall", "f1_score", "custo_usd",
            "tempo_execucao_s", "input_tokens", "output_tokens", "quantidade",
            "status_counts", "TP", "FP", "FN",
        ]
    )]
    df = _para_numerico(df, colunas_numericas)

    df["familia"] = df["imagem"].apply(lambda nome: re.sub(r"_?\d+\.(png|jpg|jpeg|webp)$", "", str(nome), flags=re.IGNORECASE))
    df["circuito"] = df["imagem"].str.replace(r"\.(png|jpg|jpeg|webp)$", "", regex=True)

    # dfd (detalhe) não tem coluna própria de idioma — traz do resumo via
    # id_execucao, para poder separar por idioma também as métricas de
    # status item a item e de identificação de componentes.
    if "idioma" in df.columns and "id_execucao" in dfd.columns:
        dfd = dfd.merge(df[["id_execucao", "idioma"]], on="id_execucao", how="left")

    return df, dfd


# =========================================================
# TABELAS
# =========================================================
def gerar_tabelas(df, dfd):
    PASTA_TABELAS.mkdir(parents=True, exist_ok=True)
    tabelas = {}

    # --- Resumo por nível de prompt ---
    metricas_prompt = [
        "numerico_tensoes_taxa_acerto_percentual",
        "numerico_correntes_taxa_acerto_percentual",
        "numerico_potencias_taxa_acerto_percentual",
        "numerico_Vout_erro_percentual",
        "textual_regiao_operacao_taxa_acerto_percentual",
        "tempo_execucao_s",
        "custo_usd",
    ]
    metricas_prompt = [m for m in metricas_prompt if m in df.columns]
    resumo_prompt = df.groupby("nivel_prompt")[metricas_prompt].mean().round(3)
    if "textual_terra_virtual_acerto" in df.columns:
        # A coluna vem como texto do CSV ("True"/"False"), não booleano —
        # dtype=str em _carregar_csv preserva isso; comparar contra a
        # string evita o bug de 's == True' nunca bater.
        resumo_prompt["terra_virtual_taxa_acerto_percentual"] = (
            df.groupby("nivel_prompt")["textual_terra_virtual_acerto"]
            .apply(lambda s: round((s == "True").sum() / s.notna().sum() * 100, 1) if s.notna().sum() else None)
        )
    tabelas["resumo_por_prompt"] = resumo_prompt

    # --- Resumo por família de circuito ---
    resumo_familia = df.groupby("familia")[metricas_prompt].mean().round(3)
    tabelas["resumo_por_familia"] = resumo_familia

    # --- Resumo por idioma (PT x EN) ---
    # Só existe quando a rodada foi feita com config.IDIOMA_PROMPT = "ambos"
    # (ou quando há mais de um CSV combinado manualmente) — com um só
    # idioma nos dados, a comparação não diz nada, então a tabela nem é
    # gerada.
    if "idioma" in df.columns and df["idioma"].nunique() > 1:
        resumo_idioma = df.groupby("idioma")[metricas_prompt].mean().round(3)
        if "textual_terra_virtual_acerto" in df.columns:
            resumo_idioma["terra_virtual_taxa_acerto_percentual"] = (
                df.groupby("idioma")["textual_terra_virtual_acerto"]
                .apply(lambda s: round((s == "True").sum() / s.notna().sum() * 100, 1) if s.notna().sum() else None)
            )
        tabelas["resumo_por_idioma"] = resumo_idioma

    # --- Piores casos de Vout (status incorreto) ---
    if "numerico_Vout_status" in df.columns:
        piores = df[df["numerico_Vout_status"] == "incorreto"][
            [c for c in ["circuito", "nivel_prompt", "numerico_Vout_IA", "numerico_Vout_REAL",
                         "numerico_Vout_erro_percentual"] if c in df.columns]
        ].sort_values("numerico_Vout_erro_percentual", ascending=False, na_position="first")
        tabelas["piores_vout"] = piores

    # --- Componentes por categoria (médias) ---
    linhas_comp = []
    for cat in CATEGORIAS_COMPONENTES:
        linha = {"categoria": cat}
        # Razões (precision/recall/f1/valor_taxa_acerto): ficam em branco
        # (None) quando não há dado — 0% sugeriria erro real, quando na
        # verdade é "não avaliável" (ex: categoria sem campo 'valor', ou
        # que nunca apareceu em nenhum circuito desta rodada).
        for metrica in ["precision", "recall", "f1_score", "valor_taxa_acerto_percentual"]:
            col = f"componentes_{cat}_{metrica}"
            if col in df.columns:
                valido = df[col].dropna()
                linha[metrica] = round(valido.mean(), 4) if len(valido) else None
        # Contagens (TP/FP/FN): sempre um número, mesmo 0 — "a categoria
        # nunca apareceu nesta rodada" É zero identificado/alucinado/
        # faltante, não um dado ausente. Usa .get(col, 0) porque a coluna
        # pode nem existir no CSV se nenhuma execução preencheu a categoria.
        for metrica in ["TP", "FP", "FN"]:
            col = f"componentes_{cat}_{metrica}"
            linha[f"{metrica}_total"] = int(df[col].fillna(0).sum()) if col in df.columns else 0
        linhas_comp.append(linha)
    tabelas["componentes_por_categoria"] = pd.DataFrame(linhas_comp).set_index("categoria")

    # --- Componentes por categoria, separado por idioma (PT x EN) ---
    if "idioma" in df.columns and df["idioma"].nunique() > 1:
        linhas_comp_idioma = []
        for idioma, df_idioma in df.groupby("idioma"):
            for cat in CATEGORIAS_COMPONENTES:
                linha = {"idioma": idioma, "categoria": cat}
                for metrica in ["precision", "recall", "f1_score", "valor_taxa_acerto_percentual"]:
                    col = f"componentes_{cat}_{metrica}"
                    if col in df_idioma.columns:
                        valido = df_idioma[col].dropna()
                        linha[metrica] = round(valido.mean(), 4) if len(valido) else None
                for metrica in ["TP", "FP", "FN"]:
                    col = f"componentes_{cat}_{metrica}"
                    linha[f"{metrica}_total"] = int(df_idioma[col].fillna(0).sum()) if col in df_idioma.columns else 0
                linhas_comp_idioma.append(linha)
        tabelas["componentes_por_idioma"] = pd.DataFrame(linhas_comp_idioma).set_index(["idioma", "categoria"])

    # --- Status geral do detalhe (todas as comparações item a item) ---
    if "status" in dfd.columns:
        tabelas["status_geral_detalhe"] = dfd["status"].value_counts().to_frame("quantidade")
        tabelas["status_por_categoria_detalhe"] = (
            dfd.groupby("categoria")["status"].value_counts().unstack(fill_value=0)
        )

    # --- Status geral e por categoria, separado por idioma (PT x EN) ---
    if "idioma" in dfd.columns and dfd["idioma"].nunique() > 1:
        tabelas["status_geral_por_idioma"] = (
            dfd.groupby("idioma")["status"].value_counts().unstack(fill_value=0)
        )
        tabelas["status_por_categoria_por_idioma"] = (
            dfd.groupby(["idioma", "categoria"])["status"].value_counts().unstack(fill_value=0)
        )

    for nome, tabela in tabelas.items():
        caminho = PASTA_TABELAS / f"{nome}.csv"
        tabela.to_csv(caminho, sep=";", decimal=",", encoding="utf-8-sig")
        print(f"Tabela salva: {caminho}")

    return tabelas


# =========================================================
# GRÁFICOS
# =========================================================
def _barras_agrupadas(ax, df_plot, categorias, cores, titulo, ylabel, fmt="{:.1f}"):
    n_grupos = len(df_plot)
    n_series = len(categorias)
    largura = 0.8 / n_series
    x = range(n_grupos)

    for i, (col, label) in enumerate(categorias):
        valores = df_plot[col].values
        posicoes = [xi + i * largura - 0.4 + largura / 2 for xi in x]
        barras = ax.bar(posicoes, valores, width=largura, label=label, color=cores[i % len(cores)])
        ax.bar_label(barras, labels=[fmt.format(v) if pd.notna(v) else "" for v in valores], fontsize=8, padding=2)

    ax.set_xticks(list(x))
    ax.set_xticklabels(df_plot.index.astype(str))
    ax.set_title(titulo, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def gerar_graficos(df, dfd):
    PASTA_GRAFICOS.mkdir(parents=True, exist_ok=True)

    categorias_num = [
        ("numerico_tensoes_taxa_acerto_percentual", "Tensões"),
        ("numerico_correntes_taxa_acerto_percentual", "Correntes"),
        ("numerico_potencias_taxa_acerto_percentual", "Potências"),
    ]
    categorias_num = [(c, l) for c, l in categorias_num if c in df.columns]

    # --- Gráfico 1: taxa de acerto numérica por nível de prompt ---
    if categorias_num:
        resumo_prompt = df.groupby("nivel_prompt")[[c for c, _ in categorias_num]].mean()
        fig, ax = plt.subplots(figsize=(9, 5))
        _barras_agrupadas(ax, resumo_prompt, categorias_num, CORES_PROMPT,
                           "Taxa de acerto numérico por nível de prompt", "Taxa de acerto (%)", "{:.1f}%")
        ax.set_xlabel("Nível de prompt")
        fig.tight_layout()
        caminho = PASTA_GRAFICOS / "taxa_acerto_por_prompt.png"
        fig.savefig(caminho, dpi=150)
        plt.close(fig)
        print(f"Gráfico salvo: {caminho}")

        # --- Gráfico 2: taxa de acerto numérica por família ---
        resumo_familia = df.groupby("familia")[[c for c, _ in categorias_num]].mean()
        fig, ax = plt.subplots(figsize=(9, 5))
        _barras_agrupadas(ax, resumo_familia, categorias_num, CORES_PROMPT,
                           "Taxa de acerto numérico por família de circuito", "Taxa de acerto (%)", "{:.1f}%")
        ax.set_xlabel("Família")
        fig.tight_layout()
        caminho = PASTA_GRAFICOS / "taxa_acerto_por_familia.png"
        fig.savefig(caminho, dpi=150)
        plt.close(fig)
        print(f"Gráfico salvo: {caminho}")

        # --- Gráfico 2b: taxa de acerto numérica por idioma (PT x EN) ---
        if "idioma" in df.columns and df["idioma"].nunique() > 1:
            resumo_idioma = df.groupby("idioma")[[c for c, _ in categorias_num]].mean()
            fig, ax = plt.subplots(figsize=(7, 5))
            _barras_agrupadas(ax, resumo_idioma, categorias_num, CORES_PROMPT,
                               "Taxa de acerto numérico: português x inglês", "Taxa de acerto (%)", "{:.1f}%")
            ax.set_xlabel("Idioma do prompt")
            fig.tight_layout()
            caminho = PASTA_GRAFICOS / "taxa_acerto_por_idioma.png"
            fig.savefig(caminho, dpi=150)
            plt.close(fig)
            print(f"Gráfico salvo: {caminho}")

    # --- Gráfico 3: região de operação por prompt ---
    if "textual_regiao_operacao_taxa_acerto_percentual" in df.columns:
        serie = df.groupby("nivel_prompt")["textual_regiao_operacao_taxa_acerto_percentual"].mean()
        fig, ax = plt.subplots(figsize=(7, 5))
        barras = ax.bar(serie.index.astype(str), serie.values, color=CORES_PROMPT[:len(serie)])
        ax.bar_label(barras, labels=[f"{v:.1f}%" if pd.notna(v) else "n/d" for v in serie.values], fontweight="bold")
        ax.set_title("Acerto de região de operação por nível de prompt", fontsize=13, fontweight="bold")
        ax.set_xlabel("Nível de prompt")
        ax.set_ylabel("Taxa de acerto (%)")
        ax.set_ylim(0, 110)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        caminho = PASTA_GRAFICOS / "regiao_operacao_por_prompt.png"
        fig.savefig(caminho, dpi=150)
        plt.close(fig)
        print(f"Gráfico salvo: {caminho}")

        # --- Gráfico 3b: região de operação por idioma (PT x EN) ---
        if "idioma" in df.columns and df["idioma"].nunique() > 1:
            serie = df.groupby("idioma")["textual_regiao_operacao_taxa_acerto_percentual"].mean()
            fig, ax = plt.subplots(figsize=(6, 5))
            barras = ax.bar(serie.index.astype(str), serie.values, color=CORES_PROMPT[:len(serie)])
            ax.bar_label(barras, labels=[f"{v:.1f}%" if pd.notna(v) else "n/d" for v in serie.values], fontweight="bold")
            ax.set_title("Acerto de região de operação: português x inglês", fontsize=13, fontweight="bold")
            ax.set_xlabel("Idioma do prompt")
            ax.set_ylabel("Taxa de acerto (%)")
            ax.set_ylim(0, 110)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()
            caminho = PASTA_GRAFICOS / "regiao_operacao_por_idioma.png"
            fig.savefig(caminho, dpi=150)
            plt.close(fig)
            print(f"Gráfico salvo: {caminho}")

    # --- Gráfico 3c: acerto de terra virtual por idioma (PT x EN) ---
    if "textual_terra_virtual_acerto" in df.columns and "idioma" in df.columns and df["idioma"].nunique() > 1:
        serie = (
            df.groupby("idioma")["textual_terra_virtual_acerto"]
            .apply(lambda s: round((s == "True").sum() / s.notna().sum() * 100, 1) if s.notna().sum() else None)
        )
        fig, ax = plt.subplots(figsize=(6, 5))
        barras = ax.bar(serie.index.astype(str), serie.values, color=CORES_PROMPT[:len(serie)])
        ax.bar_label(barras, labels=[f"{v:.1f}%" if pd.notna(v) else "n/d" for v in serie.values], fontweight="bold")
        ax.set_title("Acerto de terra virtual: português x inglês", fontsize=13, fontweight="bold")
        ax.set_xlabel("Idioma do prompt")
        ax.set_ylabel("Taxa de acerto (%)")
        ax.set_ylim(0, 110)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        caminho = PASTA_GRAFICOS / "terra_virtual_por_idioma.png"
        fig.savefig(caminho, dpi=150)
        plt.close(fig)
        print(f"Gráfico salvo: {caminho}")

    # --- Gráfico 4: distribuição geral de status (detalhe) ---
    if "status" in dfd.columns:
        contagem = dfd["status"].value_counts()
        cores_status = {
            "correto": "#2F9E5C", "incorreto": "#D14343", "faltante": "#C99A2E",
            "alucinado": "#B23368", "nao_identificado": "#8A6FD1",
        }
        fig, ax = plt.subplots(figsize=(8, 2.2))
        esquerda = 0
        total = contagem.sum()
        for status, valor in contagem.items():
            largura = valor
            ax.barh(0, largura, left=esquerda, color=cores_status.get(status, "#888888"), label=status)
            if largura / total > 0.03:
                ax.text(esquerda + largura / 2, 0, f"{status}\n{valor}", ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")
            esquerda += largura
        ax.set_xlim(0, total)
        ax.set_yticks([])
        ax.set_title(f"Status geral das {total} comparações do detalhe", fontsize=12, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout()
        caminho = PASTA_GRAFICOS / "status_geral.png"
        fig.savefig(caminho, dpi=150)
        plt.close(fig)
        print(f"Gráfico salvo: {caminho}")

        # --- Gráfico 5: status por categoria ---
        tabela_cat = dfd.groupby("categoria")["status"].value_counts().unstack(fill_value=0)
        status_presentes = [s for s in ["correto", "incorreto", "faltante", "alucinado", "nao_identificado"] if s in tabela_cat.columns]
        fig, ax = plt.subplots(figsize=(9, 0.5 * len(tabela_cat) + 1.5))
        esquerda = pd.Series(0, index=tabela_cat.index)
        for status in status_presentes:
            valores = tabela_cat[status]
            ax.barh(tabela_cat.index, valores, left=esquerda, color=cores_status.get(status, "#888888"), label=status)
            esquerda = esquerda + valores
        ax.set_title("Status das comparações por categoria", fontsize=13, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8, ncol=len(status_presentes))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        caminho = PASTA_GRAFICOS / "status_por_categoria.png"
        fig.savefig(caminho, dpi=150)
        plt.close(fig)
        print(f"Gráfico salvo: {caminho}")

        # --- Gráfico 6: status geral, separado por idioma (PT x EN) ---
        if "idioma" in dfd.columns and dfd["idioma"].nunique() > 1:
            tabela_idioma = dfd.groupby("idioma")["status"].value_counts().unstack(fill_value=0)
            status_presentes_idioma = [s for s in ["correto", "incorreto", "faltante", "alucinado", "nao_identificado"] if s in tabela_idioma.columns]
            fig, ax = plt.subplots(figsize=(9, 0.6 * len(tabela_idioma) + 1.5))
            esquerda = pd.Series(0, index=tabela_idioma.index)
            for status in status_presentes_idioma:
                valores = tabela_idioma[status]
                barras = ax.barh(tabela_idioma.index, valores, left=esquerda, color=cores_status.get(status, "#888888"), label=status)
                esquerda = esquerda + valores
            ax.set_title("Status geral das comparações: português x inglês", fontsize=13, fontweight="bold")
            ax.set_ylabel("Idioma do prompt")
            ax.legend(loc="upper right", fontsize=8, ncol=len(status_presentes_idioma))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()
            caminho = PASTA_GRAFICOS / "status_geral_por_idioma.png"
            fig.savefig(caminho, dpi=150)
            plt.close(fig)
            print(f"Gráfico salvo: {caminho}")

            # --- Gráfico 7: status por categoria, separado por idioma (PT x EN) ---
            idiomas_presentes = sorted(dfd["idioma"].dropna().unique())
            fig, eixos = plt.subplots(1, len(idiomas_presentes), figsize=(9 * len(idiomas_presentes), 0.5 * dfd["categoria"].nunique() + 1.5), sharey=True)
            if len(idiomas_presentes) == 1:
                eixos = [eixos]
            for ax_i, idioma in zip(eixos, idiomas_presentes):
                tabela_cat_idioma = (
                    dfd[dfd["idioma"] == idioma].groupby("categoria")["status"].value_counts().unstack(fill_value=0)
                )
                status_presentes_cat = [s for s in ["correto", "incorreto", "faltante", "alucinado", "nao_identificado"] if s in tabela_cat_idioma.columns]
                esquerda = pd.Series(0, index=tabela_cat_idioma.index)
                for status in status_presentes_cat:
                    valores = tabela_cat_idioma[status]
                    ax_i.barh(tabela_cat_idioma.index, valores, left=esquerda, color=cores_status.get(status, "#888888"), label=status)
                    esquerda = esquerda + valores
                ax_i.set_title(f"Idioma: {idioma}", fontsize=12, fontweight="bold")
                ax_i.spines["top"].set_visible(False)
                ax_i.spines["right"].set_visible(False)
            eixos[-1].legend(loc="upper right", fontsize=8)
            fig.suptitle("Status das comparações por categoria: português x inglês", fontsize=13, fontweight="bold")
            fig.tight_layout()
            caminho = PASTA_GRAFICOS / "status_por_categoria_por_idioma.png"
            fig.savefig(caminho, dpi=150)
            plt.close(fig)
            print(f"Gráfico salvo: {caminho}")

    # --- Gráfico 8: identificação de componentes (precision/recall/f1/valor) por idioma ---
    if "idioma" in df.columns and df["idioma"].nunique() > 1:
        # precision/recall/f1 vêm em escala 0-1 e valor_taxa_acerto_percentual
        # em 0-100 — convertidas todas para % aqui (só para o gráfico, a
        # tabela 'componentes_por_idioma' mantém a escala original de cada
        # métrica) pra caberem no mesmo eixo sem achatar as três primeiras.
        metricas_comp = ["precision", "recall", "f1_score", "valor_taxa_acerto_percentual"]
        linhas = []
        for idioma, df_idioma in df.groupby("idioma"):
            linha = {"idioma": idioma}
            for metrica in metricas_comp:
                cols = [f"componentes_{cat}_{metrica}" for cat in CATEGORIAS_COMPONENTES if f"componentes_{cat}_{metrica}" in df_idioma.columns]
                if cols:
                    valido = df_idioma[cols].mean(axis=1).dropna()
                    media = valido.mean() if len(valido) else None
                    if media is not None and metrica != "valor_taxa_acerto_percentual":
                        media *= 100
                    linha[metrica] = media
            linhas.append(linha)
        resumo_comp_idioma = pd.DataFrame(linhas).set_index("idioma")
        rotulos_comp = {
            "precision": "Precision (%)", "recall": "Recall (%)",
            "f1_score": "F1-score (%)", "valor_taxa_acerto_percentual": "Valor correto (%)",
        }
        categorias_comp_plot = [(m, rotulos_comp[m]) for m in metricas_comp if m in resumo_comp_idioma.columns]
        if categorias_comp_plot:
            # 4 séries neste gráfico (as demais chamadas de _barras_agrupadas
            # usam só 3) — cores próprias para não repetir cor entre
            # precision e valor_taxa_acerto_percentual (índices 0 e 3 mod 3).
            cores_4 = CORES_PROMPT + ["#8172B2"]
            fig, ax = plt.subplots(figsize=(9, 5))
            _barras_agrupadas(ax, resumo_comp_idioma, categorias_comp_plot, cores_4,
                               "Identificação de componentes: português x inglês", "Percentual (%)", "{:.1f}%")
            ax.set_xlabel("Idioma do prompt")
            ax.set_ylim(0, 110)
            fig.tight_layout()
            caminho = PASTA_GRAFICOS / "componentes_por_idioma.png"
            fig.savefig(caminho, dpi=150)
            plt.close(fig)
            print(f"Gráfico salvo: {caminho}")


def main():
    df, dfd = carregar_dados()
    print(f"\nCarregado: {len(df)} execuções (resumo), {len(dfd)} comparações (detalhe)\n")
    gerar_tabelas(df, dfd)
    gerar_graficos(df, dfd)
    print(f"\nAnálise completa em: {PASTA_ANALISE}\n")


if __name__ == "__main__":
    main()

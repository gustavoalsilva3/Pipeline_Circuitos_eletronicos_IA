"""Análise automática de erros da pipeline.

Lê resultados.csv (resumo, uma linha por execução) e resultados_detalhe.csv
(detalhe item a item) e gera tabelas (CSV) + gráficos (PNG) resumindo o
desempenho do modelo por nível de prompt, por família de circuito, e os
piores casos — para usar direto na análise do TCC sem precisar recalcular
tudo manualmente a cada rodada da pipeline.

Uso:
    python Analise_erros.py

Saída (em resultados/analise/): UMA pasta por modelo (ex: gemini-2.5-flash/,
qwen-qwen3-vl-8b/), cada uma com tabelas/ e graficos/ calculados SÓ com as
execuções daquele modelo (o nome do modelo aparece no canto de cada
gráfico), e a pasta comparativo/ só com as tabelas e gráficos que comparam
os modelos (_por_modelo). Sem essa separação, um CSV com dois modelos
misturaria os dois em médias por prompt/família/idioma. Em cada pasta:
    tabelas/*.csv   — resumo_por_prompt, resumo_por_familia, resumo_por_idioma,
                       resumo_por_modelo, piores_vout, componentes_por_categoria,
                       componentes_por_idioma, componentes_por_modelo,
                       status_geral_detalhe, status_por_categoria_detalhe,
                       status_geral_por_idioma, status_por_categoria_por_idioma,
                       status_geral_por_modelo, status_por_categoria_por_modelo
    graficos/*.png  — taxa_acerto_por_prompt, taxa_acerto_por_familia,
                       taxa_acerto_por_idioma, taxa_acerto_por_modelo,
                       regiao_operacao_por_prompt, regiao_operacao_por_idioma,
                       regiao_operacao_por_modelo, terra_virtual_por_idioma,
                       terra_virtual_por_modelo, status_geral, status_por_categoria,
                       status_geral_por_idioma, status_por_categoria_por_idioma,
                       status_geral_por_modelo, status_por_categoria_por_modelo,
                       componentes_por_idioma, componentes_por_modelo

Tudo que termina em '_por_idioma' compara PT x EN (só gerado com mais de um
idioma nos dados, ver config.IDIOMA_PROMPT) e tudo que termina em
'_por_modelo' compara os modelos avaliados na rodada (ex: Gemini x um
modelo local via LM Studio — só gerado com mais de um modelo nos dados, ver
config.USAR_MODELO_LMSTUDIO). O detalhe (dfd) não tem colunas próprias de
'idioma'/'modelo' — carregar_dados() as traz do resumo via id_execucao,
para permitir separar por elas também as métricas de 'status' item a item e
de identificação de componentes.
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

# Estado da análise em andamento (definido por _definir_saida, ver main): pasta
# de saída, rótulo do modelo impresso em cada gráfico e se só as saídas
# '_por_modelo' (comparativo entre modelos) devem ser gravadas.
_ROTULO_MODELO = ""
_APENAS_COMPARATIVO = False


def _definir_saida(pasta, rotulo="", apenas_comparativo=False):
    global PASTA_TABELAS, PASTA_GRAFICOS, _ROTULO_MODELO, _APENAS_COMPARATIVO
    PASTA_TABELAS = pasta / "tabelas"
    PASTA_GRAFICOS = pasta / "graficos"
    _ROTULO_MODELO = rotulo
    _APENAS_COMPARATIVO = apenas_comparativo


def _nome_pasta(texto):
    return re.sub(r'[<>:"/\\|?*]', "-", str(texto))


def _quebrar_rotulo(texto, largura=18):
    """Quebra um rótulo longo em várias linhas, para que nomes de modelo
    (ex: 'qwen3-vl-8b-bf16 (openrouter)') não se sobreponham no eixo X
    quando há vários modelos lado a lado."""
    palavras = str(texto).replace("/", "/ ").split()
    linhas, atual = [], ""
    for p in palavras:
        if atual and len(atual) + 1 + len(p) > largura:
            linhas.append(atual)
            atual = p
        else:
            atual = f"{atual} {p}".strip()
    if atual:
        linhas.append(atual)
    return "\n".join(l.replace("/ ", "/") for l in linhas)


def _salvar_grafico(fig, nome):
    """Grava o gráfico marcando o modelo no canto superior esquerdo, para que a
    imagem sozinha diga de qual modelo é. No comparativo só saem os gráficos
    '_por_modelo'."""
    if _APENAS_COMPARATIVO and not nome.endswith("_por_modelo.png"):
        plt.close(fig)
        return
    if _ROTULO_MODELO:
        fig.text(0.01, 0.995, _ROTULO_MODELO, ha="left", va="top", fontsize=9,
                 fontweight="bold", color="#444444")
    caminho = PASTA_GRAFICOS / nome
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"Gráfico salvo: {caminho}")

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


def _ignorar_vout_em_tensoes(df, dfd):
    """Aplica, na ANÁLISE, a mesma regra de evaluator_numerico para rodadas
    já salvas: a chave 'Vout' dentro de 'tensoes_V' não é um nó do gabarito
    (a saída já é avaliada em Vout_real_V), então não conta como nó alucinado
    em tensões. Os CSVs e JSONs originais NÃO são alterados — o filtro só age
    nos dataframes em memória: remove essas linhas do detalhe e recalcula,
    para as execuções afetadas, a taxa/quantidade de tensões do resumo."""
    if "categoria" not in dfd.columns or "item" not in dfd.columns:
        return df, dfd
    mascara = (dfd["categoria"] == "tensoes") & (dfd["item"].astype(str).str.strip().str.upper() == "VOUT")
    removidos = dfd[mascara]
    if removidos.empty:
        return df, dfd

    dfd = dfd[~mascara].copy()
    tens = dfd[dfd["categoria"] == "tensoes"]
    total = tens.groupby("id_execucao").size()
    corretos = tens[tens["status"] == "correto"].groupby("id_execucao").size()

    for id_exec, n_rem in removidos.groupby("id_execucao").size().items():
        linhas = df["id_execucao"] == id_exec
        n_total = total.get(id_exec, 0)
        taxa = round(corretos.get(id_exec, 0) / n_total * 100, 1) if n_total else float("nan")
        df.loc[linhas, "numerico_tensoes_taxa_acerto_percentual"] = taxa
        if "numerico_tensoes_quantidade" in df.columns:
            df.loc[linhas, "numerico_tensoes_quantidade"] = n_total
        for status, n in removidos[removidos["id_execucao"] == id_exec].groupby("status").size().items():
            coluna = f"numerico_tensoes_status_counts_{status}"
            if coluna in df.columns:
                df.loc[linhas, coluna] = df.loc[linhas, coluna] - n

    print(f"Ignorados {len(removidos)} itens 'Vout' em tensões (rótulo de saída, avaliado à parte) "
          f"em {removidos['id_execucao'].nunique()} execuções — dados originais intactos.")
    return df, dfd


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

    # dfd (detalhe) não tem colunas próprias de idioma/modelo — traz do
    # resumo via id_execucao, para poder separar por elas também as
    # métricas de status item a item e de identificação de componentes.
    colunas_extra = [c for c in ("idioma", "modelo") if c in df.columns]
    if colunas_extra and "id_execucao" in dfd.columns:
        dfd = dfd.merge(df[["id_execucao", *colunas_extra]], on="id_execucao", how="left")

    df, dfd = _ignorar_vout_em_tensoes(df, dfd)

    return df, dfd


# =========================================================
# TABELAS
# =========================================================
def _totais_componentes(df_sub, categorias):
    """Soma TP, FP e FN das categorias dadas no dataframe (execuções)."""
    tp = fp = fn = 0.0
    for cat in categorias:
        for nome, acumula in (("TP", 0), ("FP", 1), ("FN", 2)):
            col = f"componentes_{cat}_{nome}"
            if col in df_sub.columns:
                v = float(df_sub[col].fillna(0).sum())
                if acumula == 0:
                    tp += v
                elif acumula == 1:
                    fp += v
                else:
                    fn += v
    return tp, fp, fn


def _prf_agregado(df_sub, categorias):
    """precision/recall/F1 a partir dos TOTAIS de TP/FP/FN (métrica agregada).

    A média das razões por execução distorce: execuções em que a razão é
    indefinida (ex: sem componente da categoria no gabarito) saem da média
    de um lado e ficam do outro, gerando F1 maior que precision e recall ao
    mesmo tempo. Somar TP/FP/FN antes de dividir evita isso e é consistente
    com as contagens TP_total/FP_total/FN_total das tabelas."""
    tp, fp, fn = _totais_componentes(df_sub, categorias)
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return p, r, f1


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

    # --- Resumo por modelo (ex: Gemini x modelo local via LM Studio) ---
    # Só existe quando a rodada avaliou mais de um modelo (ver
    # config.USAR_MODELO_LMSTUDIO) — com um modelo só, a comparação não diz
    # nada, a tabela nem é gerada.
    if "modelo" in df.columns and df["modelo"].nunique() > 1:
        resumo_modelo = df.groupby("modelo")[metricas_prompt].mean().round(3)
        if "textual_terra_virtual_acerto" in df.columns:
            resumo_modelo["terra_virtual_taxa_acerto_percentual"] = (
                df.groupby("modelo")["textual_terra_virtual_acerto"]
                .apply(lambda s: round((s == "True").sum() / s.notna().sum() * 100, 1) if s.notna().sum() else None)
            )
        tabelas["resumo_por_modelo"] = resumo_modelo

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
        p, r, f1 = _prf_agregado(df, [cat])
        linha["precision"], linha["recall"], linha["f1_score"] = (
            round(v, 4) if v is not None else None for v in (p, r, f1))
        col = f"componentes_{cat}_valor_taxa_acerto_percentual"
        if col in df.columns:
            valido = df[col].dropna()
            linha["valor_taxa_acerto_percentual"] = round(valido.mean(), 4) if len(valido) else None
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
                p, r, f1 = _prf_agregado(df_idioma, [cat])
                linha["precision"], linha["recall"], linha["f1_score"] = (
                    round(v, 4) if v is not None else None for v in (p, r, f1))
                col = f"componentes_{cat}_valor_taxa_acerto_percentual"
                if col in df_idioma.columns:
                    valido = df_idioma[col].dropna()
                    linha["valor_taxa_acerto_percentual"] = round(valido.mean(), 4) if len(valido) else None
                for metrica in ["TP", "FP", "FN"]:
                    col = f"componentes_{cat}_{metrica}"
                    linha[f"{metrica}_total"] = int(df_idioma[col].fillna(0).sum()) if col in df_idioma.columns else 0
                linhas_comp_idioma.append(linha)
        tabelas["componentes_por_idioma"] = pd.DataFrame(linhas_comp_idioma).set_index(["idioma", "categoria"])

    # --- Componentes por categoria, separado por modelo ---
    if "modelo" in df.columns and df["modelo"].nunique() > 1:
        linhas_comp_modelo = []
        for modelo, df_modelo in df.groupby("modelo"):
            for cat in CATEGORIAS_COMPONENTES:
                linha = {"modelo": modelo, "categoria": cat}
                p, r, f1 = _prf_agregado(df_modelo, [cat])
                linha["precision"], linha["recall"], linha["f1_score"] = (
                    round(v, 4) if v is not None else None for v in (p, r, f1))
                col = f"componentes_{cat}_valor_taxa_acerto_percentual"
                if col in df_modelo.columns:
                    valido = df_modelo[col].dropna()
                    linha["valor_taxa_acerto_percentual"] = round(valido.mean(), 4) if len(valido) else None
                for metrica in ["TP", "FP", "FN"]:
                    col = f"componentes_{cat}_{metrica}"
                    linha[f"{metrica}_total"] = int(df_modelo[col].fillna(0).sum()) if col in df_modelo.columns else 0
                linhas_comp_modelo.append(linha)
        tabelas["componentes_por_modelo"] = pd.DataFrame(linhas_comp_modelo).set_index(["modelo", "categoria"])

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

    # --- Status geral e por categoria, separado por modelo ---
    if "modelo" in dfd.columns and dfd["modelo"].nunique() > 1:
        tabelas["status_geral_por_modelo"] = (
            dfd.groupby("modelo")["status"].value_counts().unstack(fill_value=0)
        )
        tabelas["status_por_categoria_por_modelo"] = (
            dfd.groupby(["modelo", "categoria"])["status"].value_counts().unstack(fill_value=0)
        )

    if _APENAS_COMPARATIVO:
        tabelas = {k: v for k, v in tabelas.items() if k.endswith("_por_modelo")}

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
    ax.set_xticklabels([_quebrar_rotulo(r) for r in df_plot.index.astype(str)])
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
        _salvar_grafico(fig, "taxa_acerto_por_prompt.png")

        # --- Gráfico 2: taxa de acerto numérica por família ---
        resumo_familia = df.groupby("familia")[[c for c, _ in categorias_num]].mean()
        fig, ax = plt.subplots(figsize=(9, 5))
        _barras_agrupadas(ax, resumo_familia, categorias_num, CORES_PROMPT,
                           "Taxa de acerto numérico por família de circuito", "Taxa de acerto (%)", "{:.1f}%")
        ax.set_xlabel("Família")
        fig.tight_layout()
        _salvar_grafico(fig, "taxa_acerto_por_familia.png")

        # --- Gráfico 2b: taxa de acerto numérica por idioma (PT x EN) ---
        if "idioma" in df.columns and df["idioma"].nunique() > 1:
            resumo_idioma = df.groupby("idioma")[[c for c, _ in categorias_num]].mean()
            fig, ax = plt.subplots(figsize=(7, 5))
            _barras_agrupadas(ax, resumo_idioma, categorias_num, CORES_PROMPT,
                               "Taxa de acerto numérico: português x inglês", "Taxa de acerto (%)", "{:.1f}%")
            ax.set_xlabel("Idioma do prompt")
            fig.tight_layout()
            _salvar_grafico(fig, "taxa_acerto_por_idioma.png")

        # --- Gráfico 2c: taxa de acerto numérica por modelo ---
        if "modelo" in df.columns and df["modelo"].nunique() > 1:
            resumo_modelo = df.groupby("modelo")[[c for c, _ in categorias_num]].mean()
            fig, ax = plt.subplots(figsize=(8, 5))
            _barras_agrupadas(ax, resumo_modelo, categorias_num, CORES_PROMPT,
                               "Taxa de acerto numérico por modelo", "Taxa de acerto (%)", "{:.1f}%")
            ax.set_xlabel("Modelo")
            fig.tight_layout()
            _salvar_grafico(fig, "taxa_acerto_por_modelo.png")

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
        _salvar_grafico(fig, "regiao_operacao_por_prompt.png")

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
            _salvar_grafico(fig, "regiao_operacao_por_idioma.png")

        # --- Gráfico 3b2: região de operação por modelo ---
        if "modelo" in df.columns and df["modelo"].nunique() > 1:
            serie = df.groupby("modelo")["textual_regiao_operacao_taxa_acerto_percentual"].mean()
            fig, ax = plt.subplots(figsize=(7, 5))
            barras = ax.bar(serie.index.astype(str), serie.values, color=CORES_PROMPT[:len(serie)])
            ax.bar_label(barras, labels=[f"{v:.1f}%" if pd.notna(v) else "n/d" for v in serie.values], fontweight="bold")
            ax.set_title("Acerto de região de operação por modelo", fontsize=13, fontweight="bold")
            ax.set_xlabel("Modelo")
            ax.set_ylabel("Taxa de acerto (%)")
            ax.set_ylim(0, 110)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()
            _salvar_grafico(fig, "regiao_operacao_por_modelo.png")

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
        _salvar_grafico(fig, "terra_virtual_por_idioma.png")

    # --- Gráfico 3d: acerto de terra virtual por modelo ---
    if "textual_terra_virtual_acerto" in df.columns and "modelo" in df.columns and df["modelo"].nunique() > 1:
        serie = (
            df.groupby("modelo")["textual_terra_virtual_acerto"]
            .apply(lambda s: round((s == "True").sum() / s.notna().sum() * 100, 1) if s.notna().sum() else None)
        )
        fig, ax = plt.subplots(figsize=(7, 5))
        barras = ax.bar(serie.index.astype(str), serie.values, color=CORES_PROMPT[:len(serie)])
        ax.bar_label(barras, labels=[f"{v:.1f}%" if pd.notna(v) else "n/d" for v in serie.values], fontweight="bold")
        ax.set_title("Acerto de terra virtual por modelo", fontsize=13, fontweight="bold")
        ax.set_xlabel("Modelo")
        ax.set_ylabel("Taxa de acerto (%)")
        ax.set_ylim(0, 110)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        _salvar_grafico(fig, "terra_virtual_por_modelo.png")

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
        _salvar_grafico(fig, "status_geral.png")

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
        _salvar_grafico(fig, "status_por_categoria.png")

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
            _salvar_grafico(fig, "status_geral_por_idioma.png")

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
            _salvar_grafico(fig, "status_por_categoria_por_idioma.png")

        # --- Gráfico 6b: status geral, separado por modelo ---
        if "modelo" in dfd.columns and dfd["modelo"].nunique() > 1:
            tabela_modelo = dfd.groupby("modelo")["status"].value_counts().unstack(fill_value=0)
            status_presentes_modelo = [s for s in ["correto", "incorreto", "faltante", "alucinado", "nao_identificado"] if s in tabela_modelo.columns]
            fig, ax = plt.subplots(figsize=(9, 0.6 * len(tabela_modelo) + 1.5))
            esquerda = pd.Series(0, index=tabela_modelo.index)
            for status in status_presentes_modelo:
                valores = tabela_modelo[status]
                ax.barh(tabela_modelo.index, valores, left=esquerda, color=cores_status.get(status, "#888888"), label=status)
                esquerda = esquerda + valores
            ax.set_title("Status geral das comparações por modelo", fontsize=13, fontweight="bold")
            ax.set_ylabel("Modelo")
            ax.legend(loc="upper right", fontsize=8, ncol=len(status_presentes_modelo))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()
            _salvar_grafico(fig, "status_geral_por_modelo.png")

            # --- Gráfico 7b: status por categoria, separado por modelo ---
            modelos_presentes = sorted(dfd["modelo"].dropna().unique())
            fig, eixos = plt.subplots(1, len(modelos_presentes), figsize=(9 * len(modelos_presentes), 0.5 * dfd["categoria"].nunique() + 1.5), sharey=True)
            if len(modelos_presentes) == 1:
                eixos = [eixos]
            for ax_i, modelo in zip(eixos, modelos_presentes):
                tabela_cat_modelo = (
                    dfd[dfd["modelo"] == modelo].groupby("categoria")["status"].value_counts().unstack(fill_value=0)
                )
                status_presentes_cat = [s for s in ["correto", "incorreto", "faltante", "alucinado", "nao_identificado"] if s in tabela_cat_modelo.columns]
                esquerda = pd.Series(0, index=tabela_cat_modelo.index)
                for status in status_presentes_cat:
                    valores = tabela_cat_modelo[status]
                    ax_i.barh(tabela_cat_modelo.index, valores, left=esquerda, color=cores_status.get(status, "#888888"), label=status)
                    esquerda = esquerda + valores
                ax_i.set_title(f"Modelo: {modelo}", fontsize=12, fontweight="bold")
                ax_i.spines["top"].set_visible(False)
                ax_i.spines["right"].set_visible(False)
            eixos[-1].legend(loc="upper right", fontsize=8)
            fig.suptitle("Status das comparações por categoria, por modelo", fontsize=13, fontweight="bold")
            fig.tight_layout()
            _salvar_grafico(fig, "status_por_categoria_por_modelo.png")

    # --- Gráfico 8: identificação de componentes (precision/recall/f1/valor) por idioma ---
    if "idioma" in df.columns and df["idioma"].nunique() > 1:
        # precision/recall/F1 (agregados sobre todas as categorias, a partir dos
        # totais de TP/FP/FN) vêm em escala 0-1 e valor_taxa_acerto_percentual
        # em 0-100 — convertidas todas para % aqui (só para o gráfico, a
        # tabela 'componentes_por_idioma' mantém a escala original de cada
        # métrica) pra caberem no mesmo eixo sem achatar as três primeiras.
        metricas_comp = ["precision", "recall", "f1_score", "valor_taxa_acerto_percentual"]
        linhas = []
        for idioma, df_idioma in df.groupby("idioma"):
            linha = {"idioma": idioma}
            p, r, f1 = _prf_agregado(df_idioma, CATEGORIAS_COMPONENTES)
            for metrica, v in (("precision", p), ("recall", r), ("f1_score", f1)):
                linha[metrica] = v * 100 if v is not None else None
            cols = [f"componentes_{cat}_valor_taxa_acerto_percentual" for cat in CATEGORIAS_COMPONENTES if f"componentes_{cat}_valor_taxa_acerto_percentual" in df_idioma.columns]
            if cols:
                valido = df_idioma[cols].mean(axis=1).dropna()
                linha["valor_taxa_acerto_percentual"] = valido.mean() if len(valido) else None
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
            _salvar_grafico(fig, "componentes_por_idioma.png")

    # --- Gráfico 8b: identificação de componentes (precision/recall/f1/valor) por modelo ---
    if "modelo" in df.columns and df["modelo"].nunique() > 1:
        metricas_comp = ["precision", "recall", "f1_score", "valor_taxa_acerto_percentual"]
        linhas = []
        for modelo, df_modelo in df.groupby("modelo"):
            linha = {"modelo": modelo}
            p, r, f1 = _prf_agregado(df_modelo, CATEGORIAS_COMPONENTES)
            for metrica, v in (("precision", p), ("recall", r), ("f1_score", f1)):
                linha[metrica] = v * 100 if v is not None else None
            cols = [f"componentes_{cat}_valor_taxa_acerto_percentual" for cat in CATEGORIAS_COMPONENTES if f"componentes_{cat}_valor_taxa_acerto_percentual" in df_modelo.columns]
            if cols:
                valido = df_modelo[cols].mean(axis=1).dropna()
                linha["valor_taxa_acerto_percentual"] = valido.mean() if len(valido) else None
            linhas.append(linha)
        resumo_comp_modelo = pd.DataFrame(linhas).set_index("modelo")
        rotulos_comp = {
            "precision": "Precision (%)", "recall": "Recall (%)",
            "f1_score": "F1-score (%)", "valor_taxa_acerto_percentual": "Valor correto (%)",
        }
        categorias_comp_plot = [(m, rotulos_comp[m]) for m in metricas_comp if m in resumo_comp_modelo.columns]
        if categorias_comp_plot:
            cores_4 = CORES_PROMPT + ["#8172B2"]
            fig, ax = plt.subplots(figsize=(9, 5))
            _barras_agrupadas(ax, resumo_comp_modelo, categorias_comp_plot, cores_4,
                               "Identificação de componentes por modelo", "Percentual (%)", "{:.1f}%")
            ax.set_xlabel("Modelo")
            ax.set_ylim(0, 110)
            fig.tight_layout()
            _salvar_grafico(fig, "componentes_por_modelo.png")


def main():
    df, dfd = carregar_dados()
    print(f"\nCarregado: {len(df)} execuções (resumo), {len(dfd)} comparações (detalhe)\n")

    if "modelo" not in df.columns or "modelo" not in dfd.columns:
        _definir_saida(PASTA_ANALISE)
        gerar_tabelas(df, dfd)
        gerar_graficos(df, dfd)
        print(f"\nAnálise completa em: {PASTA_ANALISE}\n")
        return

    modelos = sorted(df["modelo"].dropna().unique())

    # Uma análise completa por modelo — nada misturado entre modelos.
    for modelo in modelos:
        print(f"\n===== Modelo: {modelo} =====")
        _definir_saida(PASTA_ANALISE / _nome_pasta(modelo), rotulo=f"Modelo: {modelo}")
        gerar_tabelas(df[df["modelo"] == modelo].copy(), dfd[dfd["modelo"] == modelo].copy())
        gerar_graficos(df[df["modelo"] == modelo].copy(), dfd[dfd["modelo"] == modelo].copy())

    # Comparação entre os modelos (só as saídas '_por_modelo').
    if len(modelos) > 1:
        print("\n===== Comparativo entre modelos =====")
        _definir_saida(PASTA_ANALISE / "comparativo", rotulo="Comparativo entre modelos", apenas_comparativo=True)
        gerar_tabelas(df, dfd)
        gerar_graficos(df, dfd)

    print(f"\nAnálise completa em: {PASTA_ANALISE}\n")


if __name__ == "__main__":
    main()

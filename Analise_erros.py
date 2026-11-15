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
'_por_modelo' compara os modelos avaliados na rodada (ex: Gemini x
Qwen3-VL 32B) — só gerado com mais de um modelo nos dados. O detalhe (dfd) não tem colunas próprias de
'idioma'/'modelo' — carregar_dados() as traz do resumo via id_execucao,
para permitir separar por elas também as métricas de 'status' item a item e
de identificação de componentes.
"""

import re
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import (CSV_RODADA_OFICIAL, PREFIXO_CSV_RESULTADOS,
                    PREFIXO_CSV_DETALHES, RESULTADOS_DIR,
                    caminho_detalhe as _detalhe_de)

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


def _pct(v, casas=1):
    """Percentual com virgula decimal, como no resto do texto."""
    if v is None or pd.isna(v):
        return "n/d"
    return f"{v:.{casas}f}%".replace(".", ",")


def _barras_modelo(ax, serie):
    """Barras de uma serie so, com UMA cor para todas.

    A cor aqui nao carregaria informacao que o eixo ja nao de, e pintar cada
    barra de um tom diferente gasta o unico canal livre com o que o rotulo ja
    diz. Pior: com cinco modelos e uma paleta de quatro, o quinto reciclava a
    cor do primeiro, sugerindo parentesco entre modelos distintos.

    Os rotulos do eixo sao quebrados em varias linhas porque
    "Gemini sem raciocínio" nao cabe em uma linha e se sobrepunha ao
    rotulo vizinho.
    """
    barras = ax.bar([_quebrar_rotulo(r) for r in serie.index.astype(str)],
                    serie.values, color=PALETA[0])
    ax.bar_label(barras, labels=[_pct(v) for v in serie.values],
                 fontweight="bold")
    return barras


def _salvar_grafico(fig, nome):
    """Grava o gráfico marcando o modelo no canto superior esquerdo, para que a
    imagem sozinha diga de qual modelo é. No comparativo só saem os gráficos
    '_por_modelo'."""
    if _APENAS_COMPARATIVO and not nome.endswith("_por_modelo.png"):
        plt.close(fig)
        return
    if _APENAS_COMPARATIVO:
        # Os gráficos do comparativo vão para a monografia, onde a legenda do
        # LaTeX já faz o papel de título: o título interno sai para não
        # duplicá-la. Títulos de painéis (figuras com vários eixos) ficam,
        # porque identificam cada painel.
        if fig._suptitle is not None:
            fig._suptitle.set_text("")
        if len(fig.axes) == 1:
            fig.axes[0].set_title("")
        fig.tight_layout()
    if _ROTULO_MODELO:
        fig.text(0.01, 0.995, _ROTULO_MODELO, ha="left", va="top", fontsize=9,
                 fontweight="bold", color="#444444")
    caminho = PASTA_GRAFICOS / nome
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"Gráfico salvo: {caminho}")

# Paleta categorica em ordem fixa, conferida no validador de paletas (checa
# banda de luminosidade, piso de croma, separacao sob daltonismo e piso de
# visao normal entre pares adjacentes).
#
# A paleta anterior reprovava em tres checagens: o azul #4C72B0 e o roxo
# #8172B2 ficavam a dE 1,9 sob protanopia e 7,2 em visao normal, abaixo do
# piso de 15 — ou seja, Gemini e Gemma eram dificeis de separar mesmo com
# visao de cores normal, e o roxo ainda caia abaixo do piso de croma.
#
# Tres destes tons ficam abaixo de 3:1 de contraste com o fundo branco, o
# que e aceitavel porque toda barra leva rotulo direto com o valor.
PALETA = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#6250d6", "#e34948"]
CORES_PROMPT = PALETA[:3]
CORES_MODELO = PALETA[:5]

# Nomes de exibicao das categorias, que no CSV vem em snake_case.
NOME_CATEGORIA = {
    "amplificadores_operacionais": "Amp. operacionais",
    "regiao_operacao": "Região de operação", "resistores": "Resistores",
    "capacitores": "Capacitores", "indutores": "Indutores",
    "diodos": "Diodos", "fontes": "Fontes", "tensoes": "Tensões",
    "correntes": "Correntes", "potencias": "Potências",
}
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
    """Carimbo da rodada mais recente em resultados/, ou None se não houver
    nenhuma — num clone limpo a pasta nem existe, e aí a análise cai na rodada
    oficial versionada (ver _caminhos)."""
    padrao = re.compile(rf"^{re.escape(prefixo)}_(\d{{8}}_\d{{6}})\.csv$")
    timestamps = [m.group(1) for m in (padrao.match(p.name) for p in RESULTADOS_DIR.glob("*.csv")) if m]
    if not timestamps:
        return None
    return max(timestamps)  # formato AAAAMMDD_HHMMSS ordena cronologicamente como string


def _caminhos(timestamp):
    """(CSV de resumo, CSV de detalhe, rótulo) da rodada a analisar.

    'oficial' é a pasta versionada, que acompanha o repositório e sustenta os
    números do texto; é também o recurso quando resultados/ está vazia.
    Qualquer outro valor é um carimbo AAAAMMDD_HHMMSS em resultados/."""
    if timestamp is None:
        timestamp = _timestamp_mais_recente(PREFIXO_CSV_RESULTADOS) or "oficial"

    if timestamp == "oficial":
        if not CSV_RODADA_OFICIAL.exists():
            raise FileNotFoundError(
                f"Nenhuma rodada em {RESULTADOS_DIR} e {CSV_RODADA_OFICIAL} não existe "
                "— rode main.py, ou restaure a pasta rodada_oficial/ do repositório."
            )
        return CSV_RODADA_OFICIAL, _detalhe_de(CSV_RODADA_OFICIAL), "oficial (versionada)"

    return (RESULTADOS_DIR / f"{PREFIXO_CSV_RESULTADOS}_{timestamp}.csv",
            RESULTADOS_DIR / f"{PREFIXO_CSV_DETALHES}_{timestamp}.csv",
            timestamp)


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
    """Carrega resumo e detalhe da rodada mais recente (ou de 'timestamp':
    um carimbo AAAAMMDD_HHMMSS, ou 'oficial' para a rodada versionada), já
    com as colunas numéricas convertidas e a coluna 'familia' derivada do
    nome da imagem (prefixo antes do número)."""
    caminho_resumo, caminho_detalhe, rotulo = _caminhos(timestamp)
    print(f"Analisando a rodada: {rotulo}")

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

    # --- Resumo por modelo (ex: Gemini x Qwen3-VL 32B) ---
    # Só existe quando a rodada avaliou mais de um modelo: com um modelo só,
    # a comparação não diz nada e a tabela nem é gerada.
    if "modelo" in df.columns and df["modelo"].nunique() > 1:
        resumo_modelo = df.groupby("modelo")[metricas_prompt].mean().round(3)
        if "textual_terra_virtual_acerto" in df.columns:
            resumo_modelo["terra_virtual_taxa_acerto_percentual"] = (
                df.groupby("modelo")["textual_terra_virtual_acerto"]
                .apply(lambda s: round((s == "True").sum() / s.notna().sum() * 100, 1) if s.notna().sum() else None)
            )
        tabelas["resumo_por_modelo"] = resumo_modelo

    # --- Magnitude do erro: ERM, MAE e RMSE por grandeza ---
    # A taxa de acerto e binaria: cai dentro da tolerancia ou nao. Estas tres
    # metricas medem QUANTO se erra, separando quem erra por pouco de quem
    # erra por muito. O avaliador ja as calcula por execucao; aqui elas sao
    # apenas agregadas, o que faltava para chegarem ao texto.
    # O ERM e reportado tambem pela mediana: a referencia proxima de zero faz
    # o erro relativo explodir em poucos itens e a media deixa de descrever o
    # caso tipico (ha execucao com ERM acima de 180.000%). MAE e RMSE, por
    # serem absolutos, nao sofrem esse efeito.
    GRANDEZAS = [("tensoes", "Tensões"), ("correntes", "Correntes"),
                 ("potencias", "Potências")]
    METRICAS = [("ERM_percentual", "ERM médio (%)", "mean"),
                ("ERM_percentual", "ERM mediano (%)", "median"),
                ("MAE", "MAE", "mean"),
                ("RMSE", "RMSE", "mean")]
    por_eixo = "modelo" if ("modelo" in df.columns and df["modelo"].nunique() > 1) else None
    linhas_mag = []
    grupos = df.groupby(por_eixo) if por_eixo else [("todos", df)]
    for chave, sub in grupos:
        for g, rot_g in GRANDEZAS:
            linha = {por_eixo or "conjunto": chave, "grandeza": rot_g}
            for suf, rot_m, como in METRICAS:
                col = f"numerico_{g}_{suf}"
                if col in sub.columns:
                    v = pd.to_numeric(sub[col].astype(str).str.replace(",", "."),
                                      errors="coerce")
                    if v.notna().any():
                        linha[rot_m] = round(v.median() if como == "median" else v.mean(), 4)
            linhas_mag.append(linha)
    if linhas_mag:
        # O sufixo '_por_modelo' e o que faz a tabela sobreviver ao filtro da
        # pasta comparativo (ver _APENAS_COMPARATIVO).
        nome_mag = "magnitude_erro_por_modelo" if por_eixo else "magnitude_erro"
        tabelas[nome_mag] = (
            pd.DataFrame(linhas_mag).set_index([por_eixo or "conjunto", "grandeza"]))

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
        tabela.to_csv(caminho, sep=";", decimal=",", encoding="utf-8-sig", lineterminator="\n")
        print(f"Tabela salva: {caminho}")

    return tabelas


# =========================================================
# GRÁFICOS
# =========================================================
# Nomes curtos dos modelos, os mesmos usados no texto e nas tabelas do TCC.
# Os identificadores gravados no CSV ("gemma-3-27b-bf16 (openrouter)") passam
# de 28 caracteres e se sobrepoem no eixo x dos graficos comparativos.
ROTULO_CURTO = {
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.5-flash (sem raciocinio)": "Gemini sem raciocínio",
    "qwen3-vl-32b (openrouter)": "Qwen3-VL 32B",
    "gemma-3-27b-bf16 (openrouter)": "Gemma 3 27B",
    "qwen3-vl-8b-bf16 (openrouter)": "Qwen3-VL 8B",
}
# Um modelo ausente da rodada nao vira linha: _modelos_no_eixo filtra esta
# lista pelo que existe no indice, de modo que acrescentar nomes aqui e seguro.
ORDEM_MODELOS = ["Gemini 2.5 Flash", "Gemini sem raciocínio",
                 "Qwen3-VL 32B", "Gemma 3 27B", "Qwen3-VL 8B"]


def curto(modelo):
    """Nome curto do modelo; devolve o proprio identificador se for outro."""
    return ROTULO_CURTO.get(str(modelo), str(modelo))


def _modelos_no_eixo(obj, inverter=False):
    """Encurta os nomes de modelo no indice e impoe a ordem usada no texto.

    A ordem alfabetica do groupby nao e a das tabelas do capitulo de
    resultados; sem isto, figura e tabela apresentam os mesmos modelos em
    sequencias diferentes. 'inverter' serve para barras horizontais, em que
    o primeiro item e desenhado embaixo.
    """
    obj = obj.rename(index=curto)
    ordenados = [m for m in ORDEM_MODELOS if m in obj.index]
    ordenados += [m for m in obj.index if m not in ordenados]
    if inverter:
        ordenados = ordenados[::-1]
    return obj.reindex(ordenados)


def _barras_agrupadas(ax, df_plot, categorias, cores, titulo, ylabel, fmt="{:.1f}"):
    n_grupos = len(df_plot)
    n_series = len(categorias)
    largura = 0.8 / n_series
    x = range(n_grupos)

    for i, (col, label) in enumerate(categorias):
        valores = df_plot[col].values
        posicoes = [xi + i * largura - 0.4 + largura / 2 for xi in x]
        barras = ax.bar(posicoes, valores, width=largura, label=label, color=cores[i % len(cores)])
        ax.bar_label(barras, fontsize=8, padding=2,
                     labels=[fmt.format(v).replace(".", ",") if pd.notna(v) else ""
                             for v in valores])

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
            resumo_modelo = _modelos_no_eixo(
                df.groupby("modelo")[[c for c, _ in categorias_num]].mean())
            fig, ax = plt.subplots(figsize=(9, 5))
            _barras_agrupadas(ax, resumo_modelo, categorias_num, CORES_PROMPT,
                               "Taxa de acerto numérico por modelo", "Taxa de acerto (%)", "{:.1f}%")
            ax.set_xlabel("Configuração")
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
            serie = _modelos_no_eixo(
                df.groupby("modelo")["textual_regiao_operacao_taxa_acerto_percentual"].mean())
            fig, ax = plt.subplots(figsize=(8, 5))
            _barras_modelo(ax, serie)
            ax.set_title("Acerto de região de operação por modelo", fontsize=13, fontweight="bold")
            ax.set_xlabel("Configuração")
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
        serie = _modelos_no_eixo(
            df.groupby("modelo")["textual_terra_virtual_acerto"]
            .apply(lambda s: round((s == "True").sum() / s.notna().sum() * 100, 1) if s.notna().sum() else None)
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        _barras_modelo(ax, serie)
        ax.set_title("Acerto de terra virtual por modelo", fontsize=13, fontweight="bold")
        ax.set_xlabel("Configuração")
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
            tabela_modelo = _modelos_no_eixo(
                dfd.groupby("modelo")["status"].value_counts().unstack(fill_value=0), inverter=True)
            status_presentes_modelo = [s for s in ["correto", "incorreto", "faltante", "alucinado", "nao_identificado"] if s in tabela_modelo.columns]
            # A figura entra na monografia a 0,8 da largura do texto: com 9
            # polegadas e fonte 8 a legenda ficava ilegivel. Com a fonte maior,
            # a legenda sobe para fora do eixo, onde nao cobre a barra mais longa.
            # Largura 9: com 7 polegadas a legenda de quatro colunas em fonte 12
            # estourava a borda e "alucinado" saia cortado como "alucina".
            fig, ax = plt.subplots(figsize=(9, 0.6 * len(tabela_modelo) + 1.9))
            esquerda = pd.Series(0, index=tabela_modelo.index)
            for status in status_presentes_modelo:
                valores = tabela_modelo[status]
                ax.barh(tabela_modelo.index, valores, left=esquerda, color=cores_status.get(status, "#888888"), label=status)
                esquerda = esquerda + valores
            ax.set_title("Status geral das comparações por modelo", fontsize=13, fontweight="bold")
            ax.set_xlabel("Itens", fontsize=13)
            ax.set_ylabel("Configuração", fontsize=13)
            ax.tick_params(labelsize=12)
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=11,
                      ncol=len(status_presentes_modelo), frameon=False,
                      columnspacing=1.4, handlelength=1.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()
            _salvar_grafico(fig, "status_geral_por_modelo.png")

            # --- Gráfico 7b: status por categoria, separado por modelo ---
            modelos_presentes = sorted(
                dfd["modelo"].dropna().unique(),
                key=lambda m: (ORDEM_MODELOS.index(curto(m)) if curto(m) in ORDEM_MODELOS
                               else len(ORDEM_MODELOS), str(m)))
            # 9 polegadas POR modelo davam 45 de largura com cinco modelos: o
            # texto virava ilegivel. sharex alinha a escala dos paineis, sem o
            # que cada um tinha um maximo diferente e a comparacao visual entre
            # modelos, que e a razao de ser do painel, ficava invalida.
            fig, eixos = plt.subplots(
                1, len(modelos_presentes), sharey=True, sharex=True,
                figsize=(2.6 * len(modelos_presentes) + 1.6,
                         0.42 * dfd["categoria"].nunique() + 2.4))
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
                ax_i.set_title(_quebrar_rotulo(curto(modelo), 16),
                               fontsize=10, fontweight="bold")
                ax_i.tick_params(labelsize=9)
                ax_i.set_xlabel("Itens", fontsize=9)
                ax_i.spines["top"].set_visible(False)
                ax_i.spines["right"].set_visible(False)
            eixos[0].set_yticklabels(
                [NOME_CATEGORIA.get(t.get_text(), t.get_text())
                 for t in eixos[0].get_yticklabels()])
            eixos[0].set_ylabel("Categoria", fontsize=9)
            # Legenda unica para o painel inteiro, abaixo dos eixos: dentro do
            # ultimo painel ela ficava do tamanho de um carimbo.
            manipuladores, nomes = eixos[0].get_legend_handles_labels()
            fig.legend(manipuladores, nomes, loc="lower center", ncol=len(nomes),
                       fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02))
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
        resumo_comp_modelo = _modelos_no_eixo(pd.DataFrame(linhas).set_index("modelo"))
        rotulos_comp = {
            "precision": "Precision (%)", "recall": "Recall (%)",
            "f1_score": "F1-score (%)", "valor_taxa_acerto_percentual": "Valor correto (%)",
        }
        categorias_comp_plot = [(m, rotulos_comp[m]) for m in metricas_comp if m in resumo_comp_modelo.columns]
        if categorias_comp_plot:
            # Pontos, e nao barras: todos os valores ficam entre 92% e 100%, de
            # modo que barras a partir de zero gastariam quase toda a area com
            # espaco vazio e empilhariam vinte rotulos de "100,0%" uns sobre os
            # outros. Encurtar o eixo resolveria o espaco, mas barra cortada
            # mente sobre a proporcao; ponto nao tem base, entao o eixo pode
            # comecar perto do piso dos dados sem enganar o leitor.
            colunas_m = [c for c, _ in categorias_comp_plot]
            fig, ax = plt.subplots(
                figsize=(9.5, 0.95 * len(resumo_comp_modelo) + 2.2))
            y = list(range(len(resumo_comp_modelo)))
            # Cada serie ganha a sua propria sub-linha dentro da faixa do
            # modelo. Sem esse deslocamento, precision, recall e F1 valem
            # todos 100% em tres dos modelos e os tres pontos ficam um sobre
            # o outro: so o ultimo desenhado aparece, e o leitor conclui que
            # as outras metricas nao foram medidas.
            n_series = len(categorias_comp_plot)
            passo = 0.19
            for i, (col, label) in enumerate(categorias_comp_plot):
                yi = [v + (i - (n_series - 1) / 2) * passo for v in y]
                ax.scatter(resumo_comp_modelo[col].values, yi, s=80, zorder=3,
                           color=PALETA[i], label=label,
                           edgecolors="white", linewidths=1.2)
                for xv, yv in zip(resumo_comp_modelo[col].values, yi):
                    if pd.notna(xv):
                        ax.annotate(_pct(xv), (xv, yv), fontsize=8,
                                    textcoords="offset points", xytext=(9, 0),
                                    va="center", color="#333333")
            ax.set_yticks(y)
            ax.set_yticklabels(list(resumo_comp_modelo.index.astype(str)))
            ax.invert_yaxis()
            ax.set_xlabel("Percentual (%)")
            ax.set_ylabel("Configuração")
            piso = float(resumo_comp_modelo[colunas_m].min().min())
            # Folga a direita para os rotulos dos pontos em 100%.
            ax.set_xlim(max(0.0, piso - 2.0), 103.2)
            ax.grid(axis="x", color="#DDDDDD", linewidth=0.8)
            ax.set_axisbelow(True)
            ax.legend(fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                      ncol=len(categorias_comp_plot), frameon=False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()
            _salvar_grafico(fig, "componentes_por_modelo.png")


def main(timestamp=None):
    df, dfd = carregar_dados(timestamp)
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
        _definir_saida(PASTA_ANALISE / "comparativo", apenas_comparativo=True)
        gerar_tabelas(df, dfd)
        gerar_graficos(df, dfd)

    gerar_figuras_tfc(df)

    print(f"\nAnálise completa em: {PASTA_ANALISE}\n")


# =========================================================
# Figuras do Capitulo 4
#
# Gravadas direto em TFC/figuras/comparativo porque entram no texto. As
# duas de painel usam uma coluna por configuracao, com sharey e sharex: sem
# isso cada painel ganha a sua propria escala e a comparacao visual entre
# configuracoes, que e a razao de ser do painel, fica invalida.
# =========================================================
PASTA_TFC = RESULTADOS_DIR.parent / "TFC" / "figuras" / "comparativo"

NOME_FAMILIA = {"Amp_op": "Amp. operacional", "Diodo": "Diodo", "RLC": "RLC"}
NOME_NIVEL = {"1": "Direto", "2": "Guiado", "3": "Estruturado"}
NOME_IDIOMA = {"pt": "Português", "en": "Inglês"}

CATEGORIAS_NUM = [
    ("numerico_tensoes_taxa_acerto_percentual", "Tensões"),
    ("numerico_correntes_taxa_acerto_percentual", "Correntes"),
    ("numerico_potencias_taxa_acerto_percentual", "Potências"),
]


def _salvar_tfc(fig, nome):
    PASTA_TFC.mkdir(parents=True, exist_ok=True)
    caminho = PASTA_TFC / nome
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura do TFC: {caminho}")


def _painel_por_fator(df, coluna, rotulos, xlabel, nome):
    """Grafico de pontos, um painel por grandeza.

    O painel de barras anterior punha 45 valores em 16 cm: a figura
    nascia com 39,9 cm e entrava no texto em 16, de modo que os rotulos
    de 7 pt chegavam ao papel com 2,8. Qualquer arranjo de barras daria o
    mesmo espaco por valor, entao a figura troca largura por altura: cada
    par (configuracao, categoria) ganha uma linha.

    As tres grandezas ficam em paineis separados, e nao deslocadas dentro
    da mesma linha, porque ali os rotulos de valor se cobriam sempre que
    dois pontos caiam perto. Com um ponto por linha em cada painel, a
    distancia vertical entre rotulos e a da propria linha. O sharey
    mantem a escala comum, que e o que permite comparar os paineis.
    """
    cats = [(c, l) for c, l in CATEGORIAS_NUM if c in df.columns]
    if not cats or coluna not in df.columns:
        return

    presentes = {curto(m) for m in df["modelo"].dropna().unique()}
    modelos = [m for m in ORDEM_MODELOS if m in presentes]
    chaves = [k for k in rotulos if k in set(df[coluna].astype(str))]
    if not modelos or not chaves:
        # sem aviso, um mapa de rotulos desalinhado com os dados
        # produzia silencio, e a figura simplesmente nao saia
        print(f"  [aviso] {nome} nao gerada: nenhum valor de "
              f"'{coluna}' casa com os rotulos {list(rotulos)}")
        return

    # uma linha por par; o respiro entre grupos abriga o nome da configuracao
    PASSO, RESPIRO = 1.0, 1.15
    linhas, grupos, y = [], [], 0.0
    for modelo in modelos:
        inicio = y
        for k in chaves:
            linhas.append((y, modelo, k))
            y += PASSO
        grupos.append((modelo, inicio))
        y += RESPIRO
    altura = y - RESPIRO

    fig, eixos = plt.subplots(1, len(cats), sharey=True,
                              figsize=(7.4, 0.215 * altura + 1.05))
    if len(cats) == 1:
        eixos = [eixos]

    for ax, (col, rotulo) in zip(eixos, cats):
        for i, (_, inicio) in enumerate(grupos):
            if i % 2:
                ax.axhspan(inicio - 0.55, inicio + len(chaves) - 0.45,
                           color="#F2F4F7", zorder=0)
        xs, ys = [], []
        for pos, modelo, k in linhas:
            sub = df[(df["modelo"].map(curto) == modelo)
                     & (df[coluna].astype(str) == k)]
            v = sub[col].mean()
            if pd.notna(v):
                xs.append(v)
                ys.append(pos)
        ax.plot(xs, ys, "o", color=PALETA[0], markersize=5,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        for x, yy in zip(xs, ys):
            ax.annotate(_pct(x), (x, yy), fontsize=7.5, color="#333333",
                        textcoords="offset points", xytext=(7, 0),
                        va="center", zorder=4)
        ax.set_title(rotulo, fontsize=10, fontweight="bold")
        ax.set_xlim(0, 128)
        ax.set_xticks([0, 50, 100])
        ax.set_xlabel("Acerto (%)", fontsize=9)
        ax.tick_params(labelsize=9)
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.8)
        ax.set_axisbelow(True)
        for lado in ("top", "right", "left"):
            ax.spines[lado].set_visible(False)

    ax0 = eixos[0]
    ax0.set_yticks([p for p, _, _ in linhas])
    ax0.set_yticklabels([rotulos[k] for _, _, k in linhas], fontsize=9)
    # o nome da configuracao vai no respiro ACIMA do grupo: a esquerda ele
    # disputava a coluna com o rotulo da categoria e os dois se encostavam
    for modelo, inicio in grupos:
        ax0.text(-0.46, inicio - 0.78, modelo,
                 transform=ax0.get_yaxis_transform(), ha="left", va="center",
                 fontsize=9, fontweight="bold")
    ax0.set_ylim(altura + 0.5, -1.25)
    fig.subplots_adjust(left=0.235, right=0.99, top=0.9, bottom=0.085,
                        wspace=0.12)
    _salvar_tfc(fig, nome)


def _figura_terra_virtual(df):
    """Taxa de acerto em terra virtual, com a linha do classificador trivial.

    Sem essa linha a figura diz o contrario do que a secao conclui: quatro
    das cinco barras passam de 58% e parecem bons resultados, quando so uma
    supera o que se obteria respondendo sempre 'nao ha terra virtual'. A
    linha vem do proprio gabarito, e nao de um numero escrito a mao.
    """
    col, esp = "textual_terra_virtual_acerto", "textual_terra_virtual_esperado"
    if col not in df.columns or esp not in df.columns:
        return

    ok = df[col].astype(str).str.strip().str.lower().eq("true")
    serie = _modelos_no_eixo(ok.groupby(df["modelo"]).mean() * 100)

    positivos = df[esp].astype(str).str.strip().str.lower().eq("true").mean()
    base = max(positivos, 1 - positivos) * 100

    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    _barras_modelo(ax, serie)
    ax.axhline(base, color="#e34948", linestyle="--", linewidth=1.6, zorder=1)
    for texto in ax.texts:  # o rotulo de 85,4% cai em cima da linha de 87,5%
        texto.set_zorder(5)
        texto.set_bbox(dict(facecolor="white", edgecolor="none", pad=0.6))
    ax.annotate(f"classificador trivial ({_pct(base)})",
                xy=(len(serie) - 0.45, base), xytext=(0, 5),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=9, color="#e34948", fontweight="bold")
    ax.set_ylim(0, 108)
    ax.set_ylabel("Execuções classificadas corretamente (%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _salvar_tfc(fig, "terra_virtual_baseline.png")


def gerar_figuras_tfc(df):
    """Grava em TFC/figuras/comparativo tudo o que o Capitulo 4 usa.

    O status geral e desenhado pelo comparativo acima, que o grava em
    resultados/analise. Aqui ele e copiado, para que a cadeia "rodar os
    scripts, compilar o TFC" nao tenha nenhum passo manual.
    """
    print("\n===== Figuras do Capítulo 4 =====")
    # Toda comparativa vai junto, e nao so a que o texto usa hoje: uma
    # figura que fica para tras numa pasta do TFC e pior do que uma que
    # nao existe, porque parece atual.
    pasta = PASTA_ANALISE / "comparativo" / "graficos"
    if pasta.exists():
        PASTA_TFC.mkdir(parents=True, exist_ok=True)
        for origem in sorted(pasta.glob("*_por_modelo.png")):
            shutil.copyfile(origem, PASTA_TFC / origem.name)
            print(f"Figura do TFC: {PASTA_TFC / origem.name}")
    _painel_por_fator(df, "familia", NOME_FAMILIA, "Família de circuito",
                      "familia_por_modelo.png")
    _painel_por_fator(df, "nivel_prompt", NOME_NIVEL, "Nível de prompt",
                      "prompt_por_modelo.png")
    _painel_por_fator(df, "idioma", NOME_IDIOMA, "Idioma do prompt",
                      "idioma_por_modelo.png")
    _figura_terra_virtual(df)


if __name__ == "__main__":
    main(*sys.argv[1:])

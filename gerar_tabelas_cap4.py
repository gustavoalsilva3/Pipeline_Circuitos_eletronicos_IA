"""Gera as tabelas LaTeX do capitulo de resultados a partir do CSV consolidado.

Escreve um arquivo .tex por tabela em TFC/tabelas_cap4/, de modo que o texto
da dissertacao apenas as inclua. Assim, refazer a rodada e regerar as tabelas
nao exige reescrever o capitulo, e nenhum numero e digitado a mao.
"""
import json
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
# Rodada analisada: a pasta versionada, que acompanha o repositorio, para que
# as tabelas do texto possam ser regeradas num clone limpo (resultados/ e
# ignorada pelo git). CARIMBO registra de qual rodada esses dados sairam.
CARIMBO = "20261010_144312"
CSV = os.path.join(BASE, "rodada_oficial", "resultados.csv")
DET = os.path.join(BASE, "rodada_oficial", "resultados_detalhe.csv")
SAIDA = os.path.join(BASE, "TFC", "tabelas_cap4")
os.makedirs(SAIDA, exist_ok=True)

ROT = {"gemini-2.5-flash": "Gemini 2.5 Flash",
       "qwen3-vl-32b (openrouter)": "Qwen3-VL 32B",
       "gemma-3-27b-bf16 (openrouter)": "Gemma 3 27B",
       "qwen3-vl-8b-bf16 (openrouter)": "Qwen3-VL 8B"}
ORDEM = ["Gemini 2.5 Flash", "Qwen3-VL 32B", "Gemma 3 27B", "Qwen3-VL 8B"]
FAM = {"Amp": "Amplificador operacional", "Dio": "Diodo", "RLC": "RLC"}
CATS = ["amplificadores_operacionais", "resistores", "fontes", "diodos", "capacitores", "indutores"]
NOME_CAT = {"amplificadores_operacionais": "Amplificadores operacionais", "resistores": "Resistores",
            "fontes": "Fontes", "diodos": "Diodos", "capacitores": "Capacitores", "indutores": "Indutores"}

d = pd.read_csv(CSV, sep=";", engine="python", on_bad_lines="skip", dtype=str)
dd = pd.read_csv(DET, sep=";", engine="python", on_bad_lines="skip", dtype=str)
d["m"] = d.modelo.map(ROT)
d["fam"] = d["imagem"].str.extract(r"(Amp|Dio|RLC)", expand=False).map(FAM)
d["circ"] = d["imagem"].str.replace(r"\.\w+$", "", regex=True)
d["idi"] = d["idioma"].map({"pt": "Português", "en": "Inglês"})
NUM = {"tens": "numerico_tensoes_taxa_acerto_percentual",
       "corr": "numerico_correntes_taxa_acerto_percentual",
       "pot": "numerico_potencias_taxa_acerto_percentual",
       "reg": "textual_regiao_operacao_taxa_acerto_percentual",
       "errVout": "numerico_Vout_erro_percentual", "tempo": "tempo_execucao_s",
       "tin": "input_tokens", "tout": "output_tokens", "custo": "custo_usd"}
for c in NUM.values():
    d[c] = pd.to_numeric(d[c].astype(str).str.replace(",", "."), errors="coerce")
d["tv"] = d.textual_terra_virtual_acerto.astype(str).str.lower().isin(["true", "1"]) * 100.0
dd = dd.merge(d[["id_execucao", "m", "fam", "circ"]], on="id_execucao", how="left")


def v(x, casas=1):
    return f"{x:.{casas}f}".replace(".", ",") if pd.notna(x) else "---"


def mil(x):
    return f"{x:,.0f}".replace(",", ".")


def comp(sub, cat=None, suf="TP"):
    cs = [f"componentes_{c}_{suf}" for c in ([cat] if cat else CATS)]
    cs = [c for c in cs if c in sub.columns]
    return pd.to_numeric(sub[cs].stack().astype(str).str.replace(",", "."), errors="coerce").fillna(0).sum()


def escreve(nome, corpo):
    open(os.path.join(SAIDA, nome), "w", encoding="utf-8").write(corpo)
    print(f"  {nome}")


def tabela(caption, label, colunas, linhas, tam="footnotesize", tabcolsep=None):
    sep = f"\t\\setlength{{\\tabcolsep}}{{{tabcolsep}}}\n" if tabcolsep else ""
    return (f"\\begin{{table}}[H]\n\t\\centering\n\t\\caption{{{caption}}}\n\t\\{tam}\n{sep}"
            f"\t\\begin{{tabular}}{{{colunas['spec']}}}\n\t\t\\toprule\n\t\t{colunas['cab']} \\\\\n\t\t\\midrule\n"
            + "\n".join("\t\t" + l for l in linhas)
            + f"\n\t\t\\bottomrule\n\t\\end{{tabular}}\n\t\\label{{{label}}}\n\\end{{table}}\n")


print("gerando tabelas em TFC/tabelas_cap4/:")

# ---------------------------------------------------------------- status geral
# Modelos nas LINHAS e status nas colunas: a forma transposta cabe na largura
# do texto, enquanto 4 modelos x (contagem + percentual) nas colunas nao cabe.
ordem_st = [("correto", "Correto"), ("incorreto", "Incorreto"), ("faltante", "Faltante"),
            ("alucinado", "Alucinado"), ("nao_identificado", "Não ident.")]
# Um status que nao ocorre em nenhum modelo viraria uma coluna inteira de "---".
ordem_st = [(st, rot) for st, rot in ordem_st if (dd.status == st).any()]
linhas = []
for m in ORDEM:
    s = dd[dd.m == m]
    cel = []
    for st, _ in ordem_st:
        k = (s.status == st).sum()
        cel.append(f"{mil(k)} ({v(100*k/len(s))}\\%)" if k else "---")
    linhas.append(f"{m} & " + " & ".join(cel) + f" & {mil(len(s))}" + r" \\")
escreve("status_geral.tex", tabela(
    "Distribuição geral de status dos itens comparados, por modelo. Cada célula traz a contagem de itens e, entre parênteses, sua participação no total do modelo.",
    "tab:statusGeral",
    {"spec": "l" + " c" * (len(ordem_st) + 1),
     "cab": "Modelo & " + " & ".join(r for _, r in ordem_st) + " & Total"},
    linhas, tam="scriptsize", tabcolsep="4pt"))

# ---------------------------------------------------------------- resumo por modelo
linhas = []
for rot, k in [("Tensões nodais (\\%)", "tens"), ("Correntes (\\%)", "corr"),
               ("Potências (\\%)", "pot"), ("Região de operação (\\%)", "reg"),
               ("Terra virtual (\\%)", "tv")]:
    col = NUM.get(k, k)
    linhas.append(f"{rot} & " + " & ".join(v(d[d.m == m][col].mean()) for m in ORDEM) + r" \\")
linhas.append(r"\midrule")
for st, rot in [("correto", "corretas"), ("incorreto", "incorretas"), ("faltante", "sem valor informado")]:
    linhas.append(f"Tensão de saída: execuções {rot} & " +
                  " & ".join(mil((d[d.m == m].numerico_Vout_status == st).sum()) for m in ORDEM) + r" \\")
linhas.append("Erro percentual mediano na tensão de saída (\\%) & " +
              " & ".join(v(d[d.m == m][NUM['errVout']].median()) for m in ORDEM) + r" \\")
escreve("resumo_modelos.tex", tabela(
    "Resumo do desempenho por modelo (taxa média de acerto por execução).", "tab:resumoModelos",
    {"spec": "p{5.6cm} c c c c", "cab": "Métrica & " + " & ".join(ORDEM)},
    linhas, tam="scriptsize", tabcolsep="4pt"))

# ---------------------------------------------------------------- por prompt / idioma
for nome, chave, grupos, cap, lab in [
        ("por_prompt.tex", "nivel_prompt", [("1", "1 --- Direto"), ("2", "2 --- Guiado"), ("3", "3 --- Estruturado")],
         "Taxa média de acerto (\\%) por nível de \\textit{prompt}, para cada modelo.", "tab:porPrompt"),
        ("por_idioma.tex", "idi", [("Português", "Português"), ("Inglês", "Inglês")],
         "Taxa média de acerto (\\%) por idioma do \\textit{prompt}, para cada modelo.", "tab:porIdioma")]:
    linhas = []
    for i, m in enumerate(ORDEM):
        if i:
            linhas.append(r"\midrule")
        for j, (val, rot) in enumerate(grupos):
            s = d[(d.m == m) & (d[chave].astype(str) == val)]
            nome_m = m if j == 0 else ""
            linhas.append(f"{nome_m} & {rot} & " + " & ".join(
                v(s[NUM[k]].mean()) if k != "tv" else v(s["tv"].mean())
                for k in ["tens", "corr", "pot", "reg", "tv"]) + r" \\")
    escreve(nome, tabela(cap, lab, {"spec": "l l c c c c c",
            "cab": "Modelo & " + ("Nível de \\textit{prompt}" if chave == "nivel_prompt" else "Idioma")
            + " & Tensões & Correntes & Potências & Região & Terra virtual"}, linhas, tabcolsep="4pt"))

# ---------------------------------------------------------------- por familia
linhas = []
for i, m in enumerate(ORDEM):
    if i:
        linhas.append(r"\midrule")
    for j, f in enumerate(["Amplificador operacional", "Diodo", "RLC"]):
        s = d[(d.m == m) & (d.fam == f)]
        reg = s[NUM["reg"]].mean()
        linhas.append(f"{m if j==0 else ''} & {f} & " + " & ".join(v(s[NUM[k]].mean()) for k in ["tens", "corr", "pot"])
                      + f" & {'---' if f=='RLC' else v(reg)} & {v(s[NUM['errVout']].mean())}" + r" \\")
escreve("por_familia.tex", tabela(
    "Taxa média de acerto (\\%) e erro médio na tensão de saída por família de circuito, para cada modelo.",
    "tab:porFamilia", {"spec": "l l c c c c c",
    "cab": "Modelo & Família & Tensões & Correntes & Potências & Região & Erro em Vout (\\%)"}, linhas, tabcolsep="4pt"))

# ---------------------------------------------------------------- componentes
# Duas tabelas em vez de uma: 4 modelos x 4 metricas dariam 17 colunas, que
# nao cabem na largura do texto. A separacao tambem espelha a leitura: uma
# tabela responde "identificou o componente?" e a outra, "leu o valor certo?".
for suf, cap, lab, calc in [
        ("f1", r"Identificação de componentes por categoria e por modelo ($F_1$-\textit{score}).",
         "tab:identComponentes", "f1"),
        ("valor", r"Acerto do valor informado dos componentes (\%), por categoria e por modelo.",
         "tab:valorComponentes", "valor")]:
    linhas = []
    for cat in CATS:
        cel = []
        for m in ORDEM:
            sub = d[d.m == m]
            if calc == "f1":
                tp, fp, fn = comp(sub, cat, "TP"), comp(sub, cat, "FP"), comp(sub, cat, "FN")
                cel.append(v(2*tp/(2*tp+fp+fn), 3) if tp else "---")
            else:
                vcol = f"componentes_{cat}_valor_taxa_acerto_percentual"
                val = pd.to_numeric(sub[vcol].astype(str).str.replace(",", "."), errors="coerce").dropna() if vcol in sub.columns else pd.Series(dtype=float)
                cel.append(v(val.mean()) if len(val) else "---")
        linhas.append(f"{NOME_CAT[cat]} & " + " & ".join(cel) + r" \\")
    if calc == "f1":
        linhas.append(r"\midrule")
        cel = []
        for m in ORDEM:
            sub = d[d.m == m]
            tp, fp, fn = comp(sub, None, "TP"), comp(sub, None, "FP"), comp(sub, None, "FN")
            cel.append(v(2*tp/(2*tp+fp+fn), 3))
        linhas.append("Todas as categorias & " + " & ".join(cel) + r" \\")
    escreve(f"componentes_{suf}.tex", tabela(cap, lab,
            {"spec": "l c c c c", "cab": "Categoria & " + " & ".join(ORDEM)}, linhas, tabcolsep="4pt"))

# ---------------------------------------------------------------- regiao de operacao
r = dd[dd.categoria == "regiao_operacao"]
linhas = []
for i, m in enumerate(ORDEM):
    if i:
        linhas.append(r"\midrule")
    for j, f in enumerate(["Amplificador operacional", "Diodo"]):
        s = r[(r.m == m) & (r.fam == f)]
        ok = (s.status == "correto").sum()
        linhas.append(f"{m if j==0 else ''} & {f} & {ok} & {len(s)} & {v(100*ok/len(s)) if len(s) else '---'}" + r" \\")
escreve("regiao_familia.tex", tabela(
    "Região de operação: componentes avaliados e classificados corretamente, por família e por modelo.",
    "tab:regiaoFamilia",
    {"spec": "l l c c c",
     "cab": "Modelo & Família & Corretos & Avaliados & Acerto (\\%)"}, linhas, tabcolsep="5pt"))

# ---------------------------------------------------------------- piores Vout
linhas = []
for i, m in enumerate(ORDEM):
    if i:
        linhas.append(r"\midrule")
    s = d[d.m == m].dropna(subset=[NUM["errVout"]]).nlargest(3, NUM["errVout"])
    for j, (_, x) in enumerate(s.iterrows()):
        ia = pd.to_numeric(str(x.numerico_Vout_IA).replace(",", "."), errors="coerce")
        re_ = pd.to_numeric(str(x.numerico_Vout_REAL).replace(",", "."), errors="coerce")
        circ = x.circ.replace("_", r"\_")
        linhas.append(f"{m if j==0 else ''} & \\textit{{{circ}}} & {x.idioma.upper()} & {x.nivel_prompt} & "
                      f"${v(ia,2)}$ V & ${v(re_,2)}$ V & {v(x[NUM['errVout']])}" + r" \\")
escreve("piores_vout.tex", tabela(
    "Três execuções com maior erro percentual na tensão de saída, por modelo.", "tab:pioresVout",
    {"spec": "l l c c c c c",
     "cab": "Modelo & Circuito & Idioma & \\textit{Prompt} & Vout (IA) & Vout (gabarito) & Erro (\\%)"},
    linhas, tabcolsep="4pt"))

# ---------------------------------------------------------------- custo e tempo
linhas = []
for rot, f in [("Tempo médio por execução (s)", lambda s: v(s[NUM["tempo"]].mean())),
               ("Tempo total da rodada (min)", lambda s: v(s[NUM["tempo"]].sum()/60)),
               ("\\textit{Tokens} de entrada (média)", lambda s: mil(s[NUM["tin"]].mean())),
               ("\\textit{Tokens} de saída (média)", lambda s: mil(s[NUM["tout"]].mean())),
               ("Custo total (US\\$)", lambda s: v(s[NUM["custo"]].sum(), 4))]:
    linhas.append(f"{rot} & " + " & ".join(f(d[d.m == m]) for m in ORDEM) + r" \\")
escreve("custo_tempo.tex", tabela(
    "Consumo de \\textit{tokens}, tempo e custo por modelo (144 execuções cada).", "tab:custoTempo",
    {"spec": "l c c c c", "cab": " & " + " & ".join(ORDEM)}, linhas, tabcolsep="4pt"))

# ------------------------------------------------- natureza do item alucinado
# O status 'alucinado' reune dois casos de natureza muito diferente: informar
# uma grandeza de um componente que EXISTE no circuito e que o gabarito nao
# cobra, e citar um item sem correspondencia alguma no esquematico. A separacao
# usa os designadores declarados no proprio gabarito, sem lista digitada a mao.
FAM_GAB = {"Amp_op": "Amp", "Diodo": "Dio", "RLC": "RLC"}
PASTA_GAB = os.path.join(BASE, "gabaritos")
_ids_cache = {}


def ids_circuito(circ):
    """Designadores de componente e rotulos de no declarados no gabarito."""
    if circ not in _ids_cache:
        familia, n = circ.rsplit("_", 1)
        alvo = ("resposta_" + FAM_GAB[familia] + n + ".json").lower()
        nome = next(f for f in os.listdir(PASTA_GAB) if f.lower() == alvo)
        g = json.load(open(os.path.join(PASTA_GAB, nome), encoding="utf-8"))
        comps = {i["id"] for grupo in g["componentes"].values() for i in grupo}
        nos = set(g["analise_numerica"].get("tensoes_V") or {})
        _ids_cache[circ] = (comps, nos)
    return _ids_cache[circ]


al = dd[dd.status == "alucinado"]
linhas = []
for m in ORDEM:
    s = al[al.m == m]
    exc = inv = 0
    for _, x in s.iterrows():
        comps, nos = ids_circuito(x["circ"])
        item = str(x["item"])
        if item in comps or (x["categoria"] == "tensoes" and item in nos):
            exc += 1
        else:
            inv += 1
    tot = exc + inv
    linhas.append(f"{m} & {mil(exc)} & {mil(inv)} & {mil(tot)} & "
                  + (rf"{v(100*inv/tot)}\%" if tot else "---") + r" \\")
escreve("alucinado_natureza.tex", tabela(
    "Natureza dos itens classificados como alucinados, por modelo. A última coluna traz "
    "a participação da invenção no total de itens alucinados do modelo.",
    "tab:alucinadoNatureza",
    {"spec": "l c c c c",
     "cab": "Modelo & Excesso & Invenção & Total & Invenção (\\%)"},
    linhas, tabcolsep="6pt"))

# ------------------------------------------------- magnitude do erro numerico
# A taxa de acerto e binaria: cai dentro da tolerancia ou nao. Estas metricas
# dizem QUANTO se erra. O ERM vai pela mediana porque a referencia proxima de
# zero faz o erro relativo explodir em poucos itens; MAE e RMSE sao absolutos
# e vao pela media.
GRANDEZAS = [("tensoes", "Tensões", "V"), ("correntes", "Correntes", "A"),
             ("potencias", "Potências", "W")]
# Estas colunas nao constam de NUM e chegam como texto, com virgula decimal.
for _g, _, _ in GRANDEZAS:
    for _suf in ("ERM_percentual", "MAE", "RMSE"):
        _c = f"numerico_{_g}_{_suf}"
        d[_c] = pd.to_numeric(d[_c].astype(str).str.replace(",", "."), errors="coerce")
linhas = []
for g, rot, un in GRANDEZAS:
    cel = []
    for m in ORDEM:
        sub = d[d.m == m]
        erm = sub[f"numerico_{g}_ERM_percentual"].median()
        mae = sub[f"numerico_{g}_MAE"].mean()
        rms = sub[f"numerico_{g}_RMSE"].mean()
        cel += [v(erm), v(mae, 2), v(rms, 2)]
    linhas.append(f"{rot} ({un}) & " + " & ".join(cel) + r" \\")
escreve("magnitude_erro.tex", tabela(
    "Magnitude do erro numérico por modelo: erro relativo mediano (ERM), erro absoluto "
    "médio (MAE) e raiz do erro quadrático médio (RMSE).",
    "tab:magnitudeErro",
    {"spec": "l" + " c c c" * len(ORDEM),
     "cab": " & " + " & ".join("\\multicolumn{3}{c}{" + m + "}" for m in ORDEM)
            + r" \\ & " + " & ".join(["ERM & MAE & RMSE"] * len(ORDEM))},
    linhas, tam="scriptsize", tabcolsep="3pt"))

# ------------------------------------------- sensibilidade a tolerancia
# A tolerancia de 5% e uma escolha do autor. Aqui as taxas sao recalculadas
# com 1% e 10% para mostrar o quanto dependem dela. A regra reproduz a do
# avaliador: referencia nula exige resposta nula; nos demais, acerta quem nao
# excede a tolerancia. Execucoes sem nenhum item de referencia na categoria
# ficam de fora da media, como no avaliador.
ZERO_ABS = 1e-9
TOLS = [1, 5, 10]
dd_t = dd.copy()
dd_t["ia_n"] = pd.to_numeric(dd_t.valor_ia.astype(str).str.replace(",", "."), errors="coerce")
dd_t["re_n"] = pd.to_numeric(dd_t.valor_real.astype(str).str.replace(",", "."), errors="coerce")
linhas = []
for g, rot, in [("tensoes", "Tensões"), ("correntes", "Correntes"), ("potencias", "Potências")]:
    sub = dd_t[dd_t.categoria == g]
    tem_ref = sub.groupby("id_execucao")["re_n"].apply(lambda v: v.notna().any())
    sub = sub[sub.id_execucao.map(tem_ref)]
    ambos = sub.ia_n.notna() & sub.re_n.notna()
    ref_zero = sub.re_n.abs() < ZERO_ABS
    erro = (sub.ia_n - sub.re_n).abs() / sub.re_n.abs().where(~ref_zero) * 100
    cel = []
    for m in ORDEM:
        for tol in TOLS:
            ok = ambos & ((ref_zero & (sub.ia_n.abs() < ZERO_ABS)) | (~ref_zero & (erro <= tol)))
            # Taxa de cada execucao arredondada a uma casa, como o avaliador
            # grava no resumo; sem isso a coluna de 5% diverge da Tabela de
            # resumo quando a media cai num empate de arredondamento.
            taxa = (ok[sub.m == m].groupby(sub.id_execucao[sub.m == m]).mean() * 100).round(1).mean()
            cel.append(v(taxa))
    linhas.append(f"{rot} & " + " & ".join(cel) + r" \\")
escreve("tolerancia.tex", tabela(
    "Sensibilidade da taxa de acerto numérico à tolerância adotada.",
    "tab:tolerancia",
    {"spec": "l" + " c c c" * len(ORDEM),
     "cab": " & " + " & ".join("\\multicolumn{3}{c}{" + m + "}" for m in ORDEM)
            + r" \\ & " + " & ".join([r"1\% & 5\% & 10\%"] * len(ORDEM))},
    linhas, tam="scriptsize", tabcolsep="3pt"))

print("\nresumo dos numeros para o texto:")
for m in ORDEM:
    s = d[d.m == m]
    sd = dd[dd.m == m]
    tp, fp, fn = comp(s, None, "TP"), comp(s, None, "FP"), comp(s, None, "FN")
    print(f"  {m:<18} itens={len(sd)} corretos={100*(sd.status=='correto').mean():.1f}% "
          f"alucinados={(sd.status=='alucinado').sum()} F1comp={2*tp/(2*tp+fp+fn):.3f} "
          f"custo=US$ {s[NUM['custo']].sum():.4f}")

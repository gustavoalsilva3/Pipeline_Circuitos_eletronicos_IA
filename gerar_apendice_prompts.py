"""Gera o Apendice A (texto integral dos prompts) a partir de prompts.py.

O apendice e reproduzido a partir do modulo que a pipeline realmente usa, e
nao transcrito a mao, para que nao haja divergencia entre o que foi enviado
aos modelos e o que a dissertacao documenta. Basta reexecutar este script
sempre que os prompts mudarem.

Escreve TFC/apendice_prompts.tex, incluido pelo Main_TFC.tex.
"""
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import prompts as P  # noqa: E402

SAIDA = os.path.join(BASE, "TFC", "apendice_prompts.tex")
MARCADOR = "[... esquema de saída em JSON, idêntico ao do Prompt Nível 1 ...]"
MARC_IDEAL = "[... bloco de idealidade dos componentes, idêntico ao do Prompt Nível 1 ...]"
MARC_REGRAS = "[... regras finais de preenchimento, idênticas às do Prompt Nível 1 ...]"


def partes(txt):
    """Separa o texto antes e depois do bloco JSON, que e identico nos tres
    niveis e por isso so e reproduzido por completo no primeiro."""
    txt = txt.strip("\n")
    i = txt.index('{\n    "tipo_circuito"')
    j = txt.index("\n}\n", i) + 3
    return txt[:i].rstrip("\n"), txt[j:].strip("\n")


def bloco_idealidade(txt):
    """O paragrafo de idealidade, do titulo ate a linha em branco seguinte."""
    i = txt.index("Considere todos os componentes IDEAIS:")
    j = txt.index("\n\n", i)
    return txt[i:j]


def elidir_idealidade(pre, ideal):
    """Troca o bloco de idealidade por um marcador, quando presente."""
    return pre.replace(ideal, MARC_IDEAL) if ideal in pre else pre


def elidir_regras(pos, pos_ref):
    """Mostra apenas as regras que diferem das do Nivel 1.

    As regras finais sao uma lista de itens iniciados por '-'. As que ja
    constam do Nivel 1 sao substituidas por um unico marcador, preservando a
    ordem das que forem exclusivas daquele nivel.
    """
    ref = set(l.strip() for l in pos_ref.split("\n") if l.strip())
    saida, pendente = [], False
    for l in pos.split("\n"):
        if not l.strip():
            continue
        if l.strip() in ref:
            pendente = True
            continue
        if pendente:
            saida.append(MARC_REGRAS)
            pendente = False
        saida.append(l)
    if pendente:
        saida.append(MARC_REGRAS)
    return "\n".join(saida)


def bloco(t):
    return ("\\begin{Verbatim}[fontsize=\\footnotesize,breaklines=true,"
            "breakautoindent=false]\n" + t + "\n\\end{Verbatim}\n")


p1 = P.PROMPT_ANALISE_CIRCUITO_1.strip("\n")
_, pos1 = partes(P.PROMPT_ANALISE_CIRCUITO_1)
IDEAL = bloco_idealidade(p1)

pre2, pos2 = partes(P.PROMPT_ANALISE_CIRCUITO_2)
pre3, pos3 = partes(P.PROMPT_ANALISE_CIRCUITO_3)
pre2, pre3 = elidir_idealidade(pre2, IDEAL), elidir_idealidade(pre3, IDEAL)
pos2, pos3 = elidir_regras(pos2, pos1), elidir_regras(pos3, pos1)

corpo = r"""\chapter*{APÊNDICE A --- PROMPTS UTILIZADOS NOS EXPERIMENTOS}
\label{ch:apendice}
\addcontentsline{toc}{chapter}{APÊNDICE A --- PROMPTS UTILIZADOS NOS EXPERIMENTOS}

Este apêndice apresenta o texto integral dos três níveis de \textit{prompt} em língua portuguesa, descritos na Seção~\ref{sec:prompts}. As versões em língua inglesa têm a mesma estrutura, com o texto das instruções e os marcadores traduzidos e as chaves do JSON de saída mantidas em português, de modo que o restante da \textit{pipeline} as processe sem distinção de idioma.

Os três níveis compartilham a maior parte do texto: diferem sobretudo nas instruções que antecedem o esquema de saída. O Nível~1 é, por isso, reproduzido por completo; nos Níveis~2 e~3, os trechos idênticos a ele --- o bloco de idealidade dos componentes, o esquema de saída em JSON e as regras finais de preenchimento --- são indicados por marcadores entre colchetes, e o que os distingue aparece na íntegra. O texto integral dos seis \textit{prompts} está no módulo que a \textit{pipeline} executa. Conforme descrito na Seção~\ref{sec:prompts}, todas as posições de valor do esquema são ocupadas por marcadores entre sinais de menor e maior, e não por valores preenchidos, a fim de evitar que os modelos os copiem em vez de calcular. O bloco de abertura declara a hipótese de idealidade dos componentes sob a qual as respostas são conferidas. Todos os \textit{prompts} foram enviados acompanhados da imagem do circuito.

\section*{A.1 \quad Nível 1 --- Direto}
""" + bloco(p1) + r"""
\section*{A.2 \quad Nível 2 --- Guiado}
""" + bloco(pre2 + "\n\n" + MARCADOR + "\n\n" + pos2) + r"""
\section*{A.3 \quad Nível 3 --- Estruturado}
""" + bloco(pre3 + "\n\n" + MARCADOR + "\n\n" + pos3)

open(SAIDA, "w", encoding="utf-8").write(corpo)
print(f"gerado: TFC/apendice_prompts.tex ({len(corpo)} caracteres)")
for rotulo, txt in [("nivel 1", p1), ("nivel 2", pre2 + pos2), ("nivel 3", pre3 + pos3)]:
    marc = txt.count("<numero a preencher>") + txt.count("<designador na imagem>")
    print(f"   {rotulo}: {marc} marcadores, valores de exemplo antigos: "
          f"{sum(txt.count(x) for x in ['4.7 kΩ', '-6.13', '3.21'])}")

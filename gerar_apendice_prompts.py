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


def partes(txt):
    """Separa o texto antes e depois do bloco JSON, que e identico nos tres
    niveis e por isso so e reproduzido por completo no primeiro."""
    txt = txt.strip("\n")
    i = txt.index('{\n    "tipo_circuito"')
    j = txt.index("\n}\n", i) + 3
    return txt[:i].rstrip("\n"), txt[j:].strip("\n")


def bloco(t):
    return ("\\begin{Verbatim}[fontsize=\\footnotesize,breaklines=true,"
            "breakautoindent=false]\n" + t + "\n\\end{Verbatim}\n")


p1 = P.PROMPT_ANALISE_CIRCUITO_1.strip("\n")
pre2, pos2 = partes(P.PROMPT_ANALISE_CIRCUITO_2)
pre3, pos3 = partes(P.PROMPT_ANALISE_CIRCUITO_3)

corpo = r"""\chapter*{APÊNDICE A --- PROMPTS UTILIZADOS NOS EXPERIMENTOS}
\label{ch:apendice}
\addcontentsline{toc}{chapter}{APÊNDICE A --- PROMPTS UTILIZADOS NOS EXPERIMENTOS}

Este apêndice apresenta o texto integral dos três níveis de \textit{prompt} em língua portuguesa, descritos na Seção~\ref{sec:prompts}. As versões em língua inglesa têm a mesma estrutura, com o texto das instruções e os marcadores traduzidos e as chaves do JSON de saída mantidas em português, de modo que o restante da \textit{pipeline} as processe sem distinção de idioma.

O esquema de saída é idêntico nos três níveis e, por isso, é reproduzido por completo apenas no Nível~1; nos Níveis~2 e~3, sua posição é indicada por um marcador entre colchetes. Conforme descrito na Seção~\ref{subsec:estudoPreliminar}, todas as posições de valor do esquema são ocupadas por marcadores entre sinais de menor e maior, e não por valores preenchidos, a fim de evitar que os modelos os copiem em vez de calcular. Todos os \textit{prompts} foram enviados acompanhados da imagem do circuito.

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

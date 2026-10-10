"""Reúne várias rodadas parciais em um único par de CSVs, que é o que o
Analise_erros.py consome (ele lê o par mais recente de resultados/).

Útil sobretudo para quem avalia aos poucos: dá para rodar um modelo hoje e
outro amanhã, ou um punhado de circuitos por vez, e só então juntar tudo numa
rodada só. Sem argumentos, o script mostra as rodadas que encontrou e pergunta
quais juntar.

Necessário porque cada chamada de main.py numera suas execuções a partir de
1: juntar os arquivos sem tratamento faria ids de rodadas diferentes
colidirem, e o CSV de detalhe — que referencia id_execucao — passaria a
apontar para a linha errada. Aqui os ids são renumerados de forma contínua e
o id antigo fica guardado em 'id_execucao_original', junto da rodada de
origem.

Acrescenta também duas colunas de reprodutibilidade que main.py não grava:

  provedor    — quem executou o modelo (Google, Alibaba, Parasail, ...)
  quantizacao — precisão em que o modelo foi servido

Elas importam porque o mesmo modelo servido em precisões diferentes não dá os
mesmos números; sem registrar isso no CSV, o dado não é auditável a partir da
planilha. Modelos que não estiverem em ORIGEM entram como 'nao informado' —
acrescente uma entrada ali para registrar a procedência dos seus.

Uso:
    python consolidar_rodadas.py                              # menu de escolha
    python consolidar_rodadas.py 20260924_082751 20260924_091447
"""

import sys
from datetime import datetime

import pandas as pd

from config import (PREFIXO_CSV_DETALHES, PREFIXO_CSV_RESULTADOS,
                    RESULTADOS_DIR, caminho_detalhe)

# provedor/quantização por modelo, para as colunas de reprodutibilidade.
# Modelos ausentes daqui são gravados com ORIGEM_PADRAO, com aviso.
ORIGEM = {
    "gemini-2.5-flash": ("Google (API)", "nao divulgada"),
    "gemma-3-27b-bf16 (openrouter)": ("Novita (OpenRouter)", "bf16"),
    "qwen3-vl-32b (openrouter)": ("Alibaba (OpenRouter)", "nao declarada"),
    "qwen3-vl-8b-bf16 (openrouter)": ("Parasail (OpenRouter)", "bf16"),
}
ORIGEM_PADRAO = ("nao informado", "nao informado")


def _ler_resumo(carimbo):
    return pd.read_csv(RESULTADOS_DIR / f"{PREFIXO_CSV_RESULTADOS}_{carimbo}.csv",
                       sep=";", engine="python", on_bad_lines="skip", dtype=str)


def _rodadas_disponiveis():
    """(carimbo, nº de execuções, modelos) de cada rodada completa em resultados/.

    Completa = tem resumo E detalhe; sem o detalhe não há o que renumerar.
    """
    rodadas = []
    for p in sorted(RESULTADOS_DIR.glob(f"{PREFIXO_CSV_RESULTADOS}_*.csv")):
        carimbo = p.name[len(PREFIXO_CSV_RESULTADOS) + 1:-len(".csv")]
        if carimbo.startswith("detalhe") or not caminho_detalhe(p).exists():
            continue
        d = _ler_resumo(carimbo)
        modelos = ", ".join(sorted(d["modelo"].dropna().unique())) if "modelo" in d else "?"
        rodadas.append((carimbo, len(d), modelos))
    return rodadas


def _escolher(rodadas):
    """Menu no terminal; devolve os carimbos escolhidos, na ordem digitada.

    A ordem importa: é nela que as execuções são renumeradas.
    """
    print(f"\nRodadas disponiveis em {RESULTADOS_DIR.name}/:\n")
    for i, (carimbo, n, modelos) in enumerate(rodadas, 1):
        print(f"  {i:>2}) {carimbo}   {n:>4} execucoes   [{modelos}]")
    print("\nQuais juntar? Numeros separados por espaco, na ordem em que serao")
    print("renumeradas.  't' = todas,  Enter = cancelar.")

    while True:
        try:
            resp = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("\ncancelado.")

        if not resp:
            raise SystemExit("cancelado.")
        if resp.lower() in ("t", "todas"):
            return [c for c, _, _ in rodadas]

        try:
            indices = [int(x) for x in resp.replace(",", " ").split()]
        except ValueError:
            print("  so numeros, separados por espaco (ou 't' para todas).")
            continue

        fora = [i for i in indices if not 1 <= i <= len(rodadas)]
        if fora:
            print(f"  fora da lista: {fora}  (escolha de 1 a {len(rodadas)}).")
            continue

        repetidos = sorted({i for i in indices if indices.count(i) > 1})
        if repetidos:
            print(f"  repetido: {repetidos}  (cada rodada entra uma vez so).")
            continue

        escolhidos = [rodadas[i - 1][0] for i in indices]
        print(f"\njuntando {len(escolhidos)}: {', '.join(escolhidos)}\n")
        return escolhidos


def main(*carimbos):
    if not carimbos:
        rodadas = _rodadas_disponiveis()
        if not rodadas:
            raise SystemExit(f"Nenhuma rodada completa (resumo + detalhe) em {RESULTADOS_DIR}.")
        carimbos = _escolher(rodadas)

    resumos, detalhes = [], []
    proximo_id = 1

    for carimbo in carimbos:
        p_resumo = RESULTADOS_DIR / f"{PREFIXO_CSV_RESULTADOS}_{carimbo}.csv"
        p_detalhe = RESULTADOS_DIR / f"{PREFIXO_CSV_DETALHES}_{carimbo}.csv"
        for p in (p_resumo, p_detalhe):
            if not p.exists():
                raise SystemExit(f"nao encontrado: {p}")

        r = _ler_resumo(carimbo)
        d = pd.read_csv(p_detalhe, sep=";", engine="python", on_bad_lines="skip", dtype=str)

        # renumera de forma contínua e leva o mapa para o detalhe
        r = r.reset_index(drop=True)
        novos = {velho: str(proximo_id + i)
                 for i, velho in enumerate(r["id_execucao"].tolist())}
        proximo_id += len(r)

        if "id_execucao_original" not in r.columns:
            r["id_execucao_original"] = r["id_execucao"]
        if "rodada_origem" not in r.columns:
            r["rodada_origem"] = carimbo
        r["rodada_origem"] = r["rodada_origem"].fillna(carimbo)
        if "fonte" not in r.columns:
            r["fonte"] = "json_reavaliado"

        r["id_execucao"] = r["id_execucao"].map(novos)
        d["id_execucao"] = d["id_execucao"].map(novos)
        faltando = d["id_execucao"].isna().sum()
        if faltando:
            raise SystemExit(f"{faltando} linhas de detalhe de {p_detalhe.name} sem id correspondente.")

        resumos.append(r)
        detalhes.append(d)

    resumo = pd.concat(resumos, ignore_index=True)
    detalhe = pd.concat(detalhes, ignore_index=True)

    # Um modelo fora de ORIGEM não impede a consolidação: as colunas são de
    # procedência, não de avaliação. Avisa para que não passe em branco.
    desconhecidos = sorted(set(resumo["modelo"].dropna()) - set(ORIGEM))
    if desconhecidos:
        print(f"aviso: {len(desconhecidos)} modelo(s) fora de ORIGEM, gravados como "
              f"'{ORIGEM_PADRAO[0]}':")
        for m in desconhecidos:
            print(f"   {m}")
        print("   (acrescente uma entrada em ORIGEM para registrar a procedencia)\n")

    origem_de = lambda m: ORIGEM.get(m, ORIGEM_PADRAO)
    resumo["provedor"] = resumo["modelo"].map(lambda m: origem_de(m)[0])
    resumo["quantizacao"] = resumo["modelo"].map(lambda m: origem_de(m)[1])

    novo = datetime.now().strftime("%Y%m%d_%H%M%S")
    p_res = RESULTADOS_DIR / f"{PREFIXO_CSV_RESULTADOS}_{novo}.csv"
    p_det = caminho_detalhe(p_res)
    p_res.parent.mkdir(parents=True, exist_ok=True)
    resumo.to_csv(p_res, sep=";", index=False, encoding="utf-8-sig")
    detalhe.to_csv(p_det, sep=";", index=False, encoding="utf-8-sig")

    print(f"resumo : {p_res.name}  ({len(resumo)} execucoes)")
    print(f"detalhe: {p_det.name}  ({len(detalhe)} itens)")
    print(f"\nids: {resumo.id_execucao.astype(int).min()} a {resumo.id_execucao.astype(int).max()}"
          f" | unicos: {resumo.id_execucao.nunique()}")
    print("\npor modelo:")
    for m, n in resumo["modelo"].value_counts().items():
        print(f"   {m:<32} {n:>3}  [{origem_de(m)[0]}, {origem_de(m)[1]}]")
    orfas = set(detalhe.id_execucao) - set(resumo.id_execucao)
    print(f"\nlinhas de detalhe orfas: {len(orfas)}")


if __name__ == "__main__":
    main(*sys.argv[1:])

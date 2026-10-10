"""Reexecuta execuções que falharam e, se a nova resposta for válida,
substitui o JSON salvo e as linhas correspondentes nos dois CSVs da rodada.

Roda nas MESMAS condições da rodada original (temperatura 0, mesmo provedor,
mesmo prompt e idioma). Isso é proposital: se a falha for comportamento
determinístico do modelo, ela se repete e nada é substituído — o que já é a
resposta para saber se vale reexecutar. Falhas de rede, ao contrário, tendem
a não se repetir.

Nada é sobrescrito sem backup: os dois CSVs são copiados para
<nome>.bak_<timestamp> antes de qualquer alteração, e o JSON antigo vira
<nome>_falha_original.json.

Uso:
    python refazer_falhas.py --dry-run     # só mostra o que faria
    python refazer_falhas.py
    python refazer_falhas.py --carimbo 20260927_122524
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import IMAGENS_DIR, JSON_DIR, MODELOS_OPENROUTER, RESULTADOS_DIR
from main import avaliar_resposta_completa, carregar_gabarito
from openrouter_client import OpenRouterCircuitAnalyzer
from prompts import (PROMPT_ANALISE_CIRCUITO_1, PROMPT_ANALISE_CIRCUITO_1_EN,
                     PROMPT_ANALISE_CIRCUITO_2, PROMPT_ANALISE_CIRCUITO_2_EN,
                     PROMPT_ANALISE_CIRCUITO_3, PROMPT_ANALISE_CIRCUITO_3_EN)

PROMPTS = {
    ("pt", 1): PROMPT_ANALISE_CIRCUITO_1, ("pt", 2): PROMPT_ANALISE_CIRCUITO_2,
    ("pt", 3): PROMPT_ANALISE_CIRCUITO_3, ("en", 1): PROMPT_ANALISE_CIRCUITO_1_EN,
    ("en", 2): PROMPT_ANALISE_CIRCUITO_2_EN, ("en", 3): PROMPT_ANALISE_CIRCUITO_3_EN,
}
# Carimbo da rodada a corrigir. Fica como argumento, e não cravado no código,
# porque um valor fixo aqui deixa de existir assim que a rodada é arquivada.
CARIMBO_PADRAO = "20260927_122524"
CHAVES = ("tipo_circuito", "componentes", "analise_numerica", "analise_textual")


def cfg_por_rotulo(rotulo):
    for c in MODELOS_OPENROUTER:
        if (c.get("rotulo") or c["id"]) == rotulo:
            return c
    raise KeyError(rotulo)


def valida(resp):
    return bool(resp) and not resp.get("erro_json") and all(k in resp for k in CHAVES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--carimbo", default=CARIMBO_PADRAO,
                    help="rodada a corrigir, no formato AAAAMMDD_HHMMSS")
    args = ap.parse_args()

    csv_resumo = RESULTADOS_DIR / f"resultados_{args.carimbo}.csv"
    csv_detalhe = RESULTADOS_DIR / f"resultados_detalhe_{args.carimbo}.csv"
    for p in (csv_resumo, csv_detalhe):
        if not p.exists():
            raise SystemExit(f"CSV da rodada nao encontrado: {p}")
    print(f"rodada: {args.carimbo}\n")

    resumo = pd.read_csv(csv_resumo, sep=";", engine="python", on_bad_lines="skip", dtype=str)
    detalhe = pd.read_csv(csv_detalhe, sep=";", engine="python", on_bad_lines="skip", dtype=str)

    # descobre as execuções falhas lendo os JSONs salvos da rodada
    falhas = []
    for f in sorted(JSON_DIR.glob("*openrouter*.json")):
        # Os backups que este script grava numa execução anterior guardam a
        # resposta falha e voltariam a casar com o filtro de erro_json logo
        # abaixo, fazendo a rodada seguinte tentar reexecutá-los.
        if f.stem.endswith("_falha_original"):
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not d.get("erro_json"):
            continue
        nome = f.stem  # Circuito_idioma_promptN_rotulo
        circ, idioma, pn, rotulo = nome.split("_", 1)[0], None, None, None
        partes = nome.split("_")
        for i, p in enumerate(partes):
            if p in ("pt", "en"):
                idioma, circ = p, "_".join(partes[:i])
                pn = int(partes[i + 1].replace("prompt", ""))
                rotulo = "_".join(partes[i + 2:])
                break
        falhas.append({"arquivo": f, "circuito": circ, "idioma": idioma,
                       "nivel": pn, "rotulo": rotulo,
                       "motivo": "falha_requisicao" if d.get("falha_requisicao") else "json_invalido"})

    if not falhas:
        print("Nenhuma execução com falha encontrada.")
        return

    print(f"{len(falhas)} execução(ões) com falha:\n")
    for x in falhas:
        print(f"  {x['circuito']:<10} {x['idioma']}  prompt{x['nivel']}  {x['rotulo']:<32} [{x['motivo']}]")
    if args.dry_run:
        print("\n--dry-run: nada foi executado.")
        return

    agora = datetime.now().strftime("%Y%m%d_%H%M%S")
    for p in (csv_resumo, csv_detalhe):
        shutil.copy(p, p.with_suffix(f".csv.bak_{agora}"))
    print(f"\nBackup dos CSVs criado (.bak_{agora}).\n")

    trocados = 0
    for x in falhas:
        cfg = cfg_por_rotulo(x["rotulo"])
        an = OpenRouterCircuitAnalyzer(cfg["id"], provider=cfg.get("provider"))
        img = next(Path(IMAGENS_DIR).glob(f"{x['circuito']}.*"))
        gabarito, _ = carregar_gabarito(img)

        print(f"-> reexecutando {x['circuito']} {x['idioma']} prompt{x['nivel']} ({cfg['id']}) ...")
        resp, i_tok, o_tok = an.analisar_imagem(img, PROMPTS[(x["idioma"], x["nivel"])])

        if not valida(resp):
            print(f"   FALHOU DE NOVO ({'sem JSON valido'}). Mantido como falha, nada substituido.")
            continue

        # localiza a linha no resumo
        mask = ((resumo.imagem == img.name) & (resumo.modelo == x["rotulo"]) &
                (resumo.idioma == x["idioma"]) & (resumo.nivel_prompt.astype(str) == str(x["nivel"])))
        if mask.sum() != 1:
            print(f"   ATENCAO: {mask.sum()} linhas casaram no resumo; pulando para nao corromper o CSV.")
            continue
        idx = resumo.index[mask][0]
        id_exec = resumo.at[idx, "id_execucao"]

        av, det = avaliar_resposta_completa(resp, gabarito, int(id_exec))

        # resumo: atualiza só as colunas que a avaliação produz, mais tokens
        for k, v in av.items():
            if k in resumo.columns:
                resumo.at[idx, k] = ("" if v is None else str(v).replace(".", ","))
        for k, v in (("input_tokens", i_tok), ("output_tokens", o_tok)):
            if k in resumo.columns:
                resumo.at[idx, k] = str(v)
        if "reexecucoes" in resumo.columns:
            resumo.at[idx, "reexecucoes"] = "1,0"

        # detalhe: remove as linhas antigas dessa execução e insere as novas
        detalhe = detalhe[detalhe.id_execucao.astype(str) != str(id_exec)]
        novas = pd.DataFrame(det)
        for c in detalhe.columns:
            if c not in novas.columns:
                novas[c] = None
        detalhe = pd.concat([detalhe, novas[detalhe.columns]], ignore_index=True)

        # JSON: guarda o antigo e grava o novo
        x["arquivo"].rename(x["arquivo"].with_name(x["arquivo"].stem + "_falha_original.json"))
        x["arquivo"].write_text(json.dumps(resp, ensure_ascii=False, indent=2), encoding="utf-8")
        trocados += 1
        print(f"   OK: JSON valido, linha {id_exec} substituida no resumo e no detalhe.")

    if trocados:
        detalhe = detalhe.sort_values("id_execucao", key=lambda s: s.astype(int)).reset_index(drop=True)
        resumo.to_csv(csv_resumo, sep=";", index=False, encoding="utf-8-sig")
        detalhe.to_csv(csv_detalhe, sep=";", index=False, encoding="utf-8-sig")
    print(f"\n{trocados} de {len(falhas)} execução(ões) substituída(s).")


if __name__ == "__main__":
    main()

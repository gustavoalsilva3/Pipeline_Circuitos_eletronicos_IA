"""Teste rápido do OpenRouter antes da rodada completa.

Roda poucos circuitos em um modelo do OpenRouter, mostrando o provedor que
respondeu, a quantização declarada, os tokens consumidos e o custo real
extrapolado para as 144 execuções da rodada final. Serve para conferir, com
gasto de centavos, se o endpoint aceita imagem, se o JSON volta no formato
esperado e quanto custaria a rodada.

Uso:
    python teste_openrouter.py                      # 2 circuitos, 32B
    python teste_openrouter.py --modelo qwen/qwen3-vl-8b-instruct --provider Parasail
    python teste_openrouter.py --circuitos Amp_op_1 Diodo_5 RLC_7 --nivel 2
"""

import argparse
import json
import sys
from pathlib import Path

import requests

from config import IMAGENS_DIR
from openrouter_client import BASE_URL, OpenRouterCircuitAnalyzer, _chave
from prompts import (PROMPT_ANALISE_CIRCUITO_1, PROMPT_ANALISE_CIRCUITO_2,
                     PROMPT_ANALISE_CIRCUITO_3)

PROMPTS = {1: PROMPT_ANALISE_CIRCUITO_1, 2: PROMPT_ANALISE_CIRCUITO_2, 3: PROMPT_ANALISE_CIRCUITO_3}
CHAVES_ESPERADAS = ("tipo_circuito", "componentes", "analise_numerica", "analise_textual")


def precos(modelo):
    """Preço por token (USD) do modelo, lido do catálogo."""
    try:
        r = requests.get(f"{BASE_URL}/models", timeout=30)
        for m in r.json().get("data", []):
            if m.get("id") == modelo:
                p = m.get("pricing") or {}
                return float(p.get("prompt") or 0), float(p.get("completion") or 0)
    except Exception as e:
        print(f"[aviso] nao foi possivel ler os precos: {e}")
    return 0.0, 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="qwen/qwen3-vl-32b-instruct")
    ap.add_argument("--provider", default="Alibaba", help="provedor fixo; use '' para deixar livre")
    ap.add_argument("--circuitos", nargs="+", default=["Amp_op_1", "RLC_1"])
    ap.add_argument("--nivel", type=int, default=1, choices=[1, 2, 3])
    args = ap.parse_args()

    try:
        _chave()
    except RuntimeError as e:
        sys.exit(str(e))

    p_in, p_out = precos(args.modelo)
    an = OpenRouterCircuitAnalyzer(args.modelo, provider=args.provider or None)

    print(f"\nModelo   : {args.modelo}")
    print(f"Provedor : {args.provider or '(livre)'}")
    print(f"Preco    : US$ {p_in * 1e6:.4f}/M entrada, US$ {p_out * 1e6:.4f}/M saida")
    print(f"Prompt   : nivel {args.nivel}\n" + "-" * 66)

    tot_in = tot_out = 0
    ok = 0
    for nome in args.circuitos:
        img = next((p for p in Path(IMAGENS_DIR).glob(f"{nome}.*")), None)
        if img is None:
            print(f"{nome:<12} imagem nao encontrada em {IMAGENS_DIR}")
            continue

        resposta, i_tok, o_tok = an.analisar_imagem(img, PROMPTS[args.nivel])
        tot_in += i_tok
        tot_out += o_tok
        meta = an.ultima_chamada
        faltando = [c for c in CHAVES_ESPERADAS if c not in resposta]
        valido = not resposta.get("erro_json") and not faltando
        ok += valido

        print(f"{nome:<12} {'JSON OK' if valido else 'FALHOU':<8} "
              f"in={i_tok:<6} out={o_tok:<6} "
              f"provedor={meta.get('provedor')} quant={meta.get('quantizacao')} "
              f"finish={meta.get('finish_reason')}")
        if faltando:
            print(f"             chaves ausentes: {faltando}")
        if resposta.get("erro_json"):
            print(f"             bruto: {str(resposta.get('resposta_bruta'))[:200]}")
        else:
            comp = resposta.get("componentes") or {}
            n = sum(len(v) for v in comp.values() if isinstance(v, list))
            num = resposta.get("analise_numerica") or {}
            print(f"             tipo={str(resposta.get('tipo_circuito'))[:44]!r} "
                  f"componentes={n} Vout={num.get('Vout_real_V')}")

    n = len(args.circuitos)
    if n and tot_in:
        custo = tot_in * p_in + tot_out * p_out
        print("-" * 66)
        print(f"{ok}/{n} respostas validas | media in={tot_in // n} out={tot_out // n}")
        print(f"Custo do teste     : US$ {custo:.6f}")
        print(f"Estimativa 144 exec: US$ {custo / n * 144:.4f}  "
              f"(R$ {custo / n * 144 * 5.5:.2f} a R$ 5,50/US$)")
        print(f"\nLog das chamadas em resultados/openrouter_chamadas.jsonl")


if __name__ == "__main__":
    main()

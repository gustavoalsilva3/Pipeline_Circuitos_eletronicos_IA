"""Piloto avaliado do OpenRouter: roda alguns circuitos num modelo e PONTUA
as respostas com os mesmos avaliadores da rodada oficial.

Serve para sondar um modelo novo — ou comparar dois — sem pagar a rodada
completa: são 6 circuitos dos mais difíceis, no prompt de nível 1, e as
métricas impressas são as mesmas do capítulo de resultados, então os números
são diretamente comparáveis aos das tabelas. As respostas ficam em
resultados/piloto_openrouter/.

Uso:
    python piloto_openrouter.py
    python piloto_openrouter.py --modelo qwen/qwen3-vl-8b-instruct --provider Parasail
    python piloto_openrouter.py --circuitos Amp_op_2 RLC_8

O --provider fixa o provedor, pelo mesmo motivo da rodada oficial (ver
config.MODELOS_OPENROUTER): sem isso o OpenRouter pode rotear para backends
com precisões diferentes, e duas execuções deixam de ser comparáveis.
"""

import argparse
import json
from pathlib import Path

from config import IMAGENS_DIR
from main import avaliar_resposta_completa, carregar_gabarito
from openrouter_client import OpenRouterCircuitAnalyzer
from prompts import PROMPT_ANALISE_CIRCUITO_1

DIFICEIS = ["Amp_op_2", "Amp_op_4", "Amp_op_6", "Diodo_7", "Diodo_8", "RLC_8"]
METRICAS = [
    ("numerico_tensoes_taxa_acerto_percentual", "tensoes"),
    ("numerico_correntes_taxa_acerto_percentual", "correntes"),
    ("numerico_potencias_taxa_acerto_percentual", "potencias"),
    ("textual_regiao_operacao_taxa_acerto_percentual", "regiao"),
]


def main():
    ap = argparse.ArgumentParser(
        description="Sonda um modelo do OpenRouter em alguns circuitos e pontua "
                    "as respostas com os avaliadores da rodada oficial.")
    ap.add_argument("--modelo", default="qwen/qwen3-vl-32b-instruct",
                    help="id do modelo no OpenRouter (padrao: %(default)s)")
    ap.add_argument("--provider", default="Alibaba",
                    help="provedor a fixar, para que duas execucoes sejam "
                         "comparaveis; vazio deixa o roteamento livre "
                         "(padrao: %(default)s)")
    ap.add_argument("--circuitos", nargs="+", default=DIFICEIS,
                    help="nomes das imagens, sem extensao (padrao: os 6 mais "
                         "dificeis da rodada oficial)")
    args = ap.parse_args()

    an = OpenRouterCircuitAnalyzer(args.modelo, provider=args.provider or None)
    saida = Path("resultados") / "piloto_openrouter"
    saida.mkdir(parents=True, exist_ok=True)

    print(f"\nModelo: {args.modelo}  |  provedor: {args.provider or '(livre)'}  |  prompt nivel 1")
    print("-" * 88)
    print(f"{'circuito':<11}{'tensoes':>9}{'corr':>7}{'pot':>7}{'regiao':>8}"
          f"{'  Vout(IA)':>12}{'Vout(gab)':>11}{'  status':>10}  comp")
    print("-" * 88)

    acum = {k: [] for k, _ in METRICAS}
    tv_ok = tv_n = 0
    for nome in args.circuitos:
        img = next((p for p in Path(IMAGENS_DIR).glob(f"{nome}.*")), None)
        if img is None:
            print(f"{nome:<11} imagem nao encontrada")
            continue
        gabarito, _ = carregar_gabarito(img)
        if not gabarito:
            print(f"{nome:<11} gabarito nao encontrado")
            continue

        resposta, _, _ = an.analisar_imagem(img, PROMPT_ANALISE_CIRCUITO_1)
        (saida / f"{nome}_{args.modelo.replace('/', '-')}.json").write_text(
            json.dumps(resposta, ensure_ascii=False, indent=2), encoding="utf-8")

        av, _ = avaliar_resposta_completa(resposta, gabarito, 0)

        def pct(chave):
            v = av.get(chave)
            return f"{v:.0f}" if isinstance(v, (int, float)) else "-"

        for chave, _rot in METRICAS:
            v = av.get(chave)
            if isinstance(v, (int, float)):
                acum[chave].append(v)

        esp = av.get("textual_terra_virtual_esperado")
        obt = av.get("textual_terra_virtual_obtido")
        if esp is not None:
            tv_n += 1
            tv_ok += bool(av.get("textual_terra_virtual_acerto"))

        # componentes: soma TP / (TP+FN) de todas as categorias
        tp = sum(v for k, v in av.items() if k.endswith("_TP") and isinstance(v, (int, float)))
        fn = sum(v for k, v in av.items() if k.endswith("_FN") and isinstance(v, (int, float)))
        fp = sum(v for k, v in av.items() if k.endswith("_FP") and isinstance(v, (int, float)))

        print(f"{nome:<11}{pct(METRICAS[0][0]):>9}{pct(METRICAS[1][0]):>7}"
              f"{pct(METRICAS[2][0]):>7}{pct(METRICAS[3][0]):>8}"
              f"{str(av.get('numerico_Vout_IA')):>12}{str(av.get('numerico_Vout_REAL')):>11}"
              f"{str(av.get('numerico_Vout_status')):>10}  TP={tp:.0f} FP={fp:.0f} FN={fn:.0f}")

    print("-" * 88)
    for chave, rot in METRICAS:
        v = acum[chave]
        print(f"  media {rot:<10} {sum(v)/len(v):.1f}%  (n={len(v)})" if v else f"  media {rot:<10} -")
    if tv_n:
        print(f"  terra virtual   {100*tv_ok/tv_n:.1f}%  ({tv_ok}/{tv_n})")
    print(f"\nRespostas salvas em {saida}")


if __name__ == "__main__":
    main()

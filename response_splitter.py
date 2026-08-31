"""Divide a resposta da IA entre dados numéricos e dados textuais.

Esse módulo serve como camada de pré-processamento para que o avaliador
possa lidar de forma separada com valores quantitativos e atributos descritivos.
"""

from __future__ import annotations

from typing import Any


def _buscar_valor(dados: dict | None, caminhos: list[tuple[str, ...]]):
    """Busca um valor em diferentes caminhos aninhados dentro de um dicionário."""
    if not isinstance(dados, dict):
        return None

    for caminho in caminhos:
        atual = dados
        valido = True

        for parte in caminho:
            if not isinstance(atual, dict) or parte not in atual:
                valido = False
                break
            atual = atual[parte]

        if valido:
            return atual

    return None


def separar_resposta(resposta: dict | None) -> dict[str, dict[str, Any]]:
    """Separa a resposta em duas estruturas: numérica e textual, para facilitar a avaliação."""
    if not isinstance(resposta, dict):
        return {"numerico": {}, "textual": {}}

    numerico: dict[str, Any] = {}
    textual: dict[str, Any] = {}

    analise_numerica = resposta.get("analise_numerica")
    if not isinstance(analise_numerica, dict):
        analise_numerica = resposta.get("analise_eletrica")

    if isinstance(analise_numerica, dict):
        numerico["analise_numerica"] = analise_numerica

    for chave, caminhos in {
        "tipo_circuito": [("tipo_circuito",), ("analise_textual", "tipo_circuito")],
        "funcao": [("funcao",), ("analise_textual", "funcao")],
        "regiao_operacao": [("regiao_operacao",), ("analise_textual", "regiao_operacao")],
        "componentes": [("componentes",), ("analise_textual", "componentes")],
        "topologia": [("topologia",), ("analise_textual", "topologia")],
        "explicacao": [("explicacao",), ("analise_textual", "explicacao")],
        "confianca_modelo": [("confianca_modelo",), ("analise_textual", "confianca_modelo")],
    }.items():
        valor = _buscar_valor(resposta, list(caminhos))
        if valor is not None:
            textual[chave] = valor

    if "analise_textual" in resposta and isinstance(resposta["analise_textual"], dict):
        for chave in ["funcao", "regiao_operacao", "topologia", "explicacao", "confianca_modelo"]:
            valor = resposta["analise_textual"].get(chave)
            if valor is not None and chave not in textual:
                textual[chave] = valor

    return {"numerico": numerico, "textual": textual}
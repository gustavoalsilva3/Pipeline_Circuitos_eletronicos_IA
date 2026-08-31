#Prompt nível 1
PROMPT_ANALISE_CIRCUITO_1 = """
Analise o circuito eletrônico presente na imagem.

Retorne APENAS um JSON válido no seguinte formato:

{
    "tipo_circuito": "Amplificador Operacional Inversor",

    "componentes": {
        "amplificadores_operacionais": [
            {"id": "U1", "modelo": "LT1057"}
        ],
        "resistores": [
            {"id": "R1", "valor": "10 kΩ"},
            {"id": "R2", "valor": "10 kΩ"},
            {"id": "R3", "valor": "50 kΩ"}
        ],
        "fontes": [
            {"id": "V1", "valor": "+12 V"},
            {"id": "V2", "valor": "-12 V"},
            {"id": "V3", "valor": "+10 V"}
        ],
        "capacitores": [],
        "diodos": []
    },

    "analise_numerica": {
        "ganho_teorico": -5.0,
        "Vout_teorico_V": -50.0,
        "Vout_real_V": -12.0,
        "tensoes_V": {
            "entrada_inversora": 0.0,
            "entrada_nao_inversora": 0.0,
            "saida_ampop": -12.0
        },
        "correntes_A": {
            "R1": 0.001,
            "R2": 0.0,
            "R3": 0.00024
        },
        "potencias_W": {
            "R1": 0.01,
            "R2": 0.0,
            "R3": 0.00288
        }
    },

    "analise_textual": {
        "regiao_operacao": "Saturação negativa",
        "terra_virtual": true
    }
}

Regras:
- Todos os valores numéricos devem ser números JSON válidos.
- Exemplo:
    "Vout_V": -12.0
    "correntes_A": {
        "R1": 0.0
    }
    "potencias_W": {
        "R1": 0.0
    }
- Caso não seja possível calcular, use "não determinado" ou null para valores individuais.
- Não escreva explicações fora do JSON.
"""

#Prompt nível 2
PROMPT_ANALISE_CIRCUITO_2 = """
Analise o circuito eletrônico.

1. Identifique os componentes presentes.

2. Classifique o tipo do circuito.

3. Determine a região de operação.

4. Resolva eletricamente o circuito.

5. Calcule ganho, tensões, correntes e potências quando possível.

Retorne APENAS um JSON válido no seguinte formato:

{
    "tipo_circuito": "Amplificador Operacional Inversor",

    "componentes": {
        "amplificadores_operacionais": [
            {"id": "U1", "modelo": "LT1057"}
        ],
        "resistores": [
            {"id": "R1", "valor": "10 kΩ"},
            {"id": "R2", "valor": "10 kΩ"},
            {"id": "R3", "valor": "50 kΩ"}
        ],
        "fontes": [
            {"id": "V1", "valor": "+12 V"},
            {"id": "V2", "valor": "-12 V"},
            {"id": "V3", "valor": "+10 V"}
        ],
        "capacitores": [],
        "diodos": []
    },

    "analise_numerica": {
        "ganho_teorico": -5.0,
        "Vout_teorico_V": -50.0,
        "Vout_real_V": -12.0,
        "tensoes_V": {
            "entrada_inversora": 0.0,
            "entrada_nao_inversora": 0.0,
            "saida_ampop": -12.0
        },
        "correntes_A": {
            "R1": 0.001,
            "R2": 0.0,
            "R3": 0.00024
        },
        "potencias_W": {
            "R1": 0.01,
            "R2": 0.0,
            "R3": 0.00288
        }
    },

    "analise_textual": {
        "regiao_operacao": "Saturação negativa",
        "terra_virtual": true
    }
}

Regras:
- Todos os valores numéricos devem ser números JSON válidos.
- Exemplo:
    "Vout_V": -12.0
    "correntes_A": {
        "R1": 0.0
    }
    "potencias_W": {
        "R1": 0.0
    }
- Caso não seja possível calcular, use "não determinado" ou null para valores individuais.
- Não escreva explicações fora do JSON.
"""

#Prompt nível 3
PROMPT_ANALISE_CIRCUITO_3 = """
OBJETIVO

Analise detalhadamente o circuito eletrônico apresentado.

PROCEDIMENTO

1. Identifique todos os componentes presentes.

2. Classifique a topologia do circuito.

3. Identifique o componente não linear principal.

4. Determine a região de operação de cada componente não linear.

5. Verifique hipóteses como:
   - terra virtual;
   - saturação;
   - condução ou bloqueio de diodos;
   - realimentação.

6. Resolva o circuito aplicando as relações elétricas apropriadas.

7. Calcule:
   - ganho;
   - tensões;
   - correntes;
   - potências.

8. Verifique se os resultados obtidos são fisicamente coerentes.

9. Caso algum valor não possa ser determinado apenas pela imagem, utilize null ou "não determinado" e justifique.

10. Preencha o JSON exatamente no formato solicitado:

{
    "tipo_circuito": "Amplificador Operacional Inversor",

    "componentes": {
        "amplificadores_operacionais": [
            {"id": "U1", "modelo": "LT1057"}
        ],
        "resistores": [
            {"id": "R1", "valor": "10 kΩ"},
            {"id": "R2", "valor": "10 kΩ"},
            {"id": "R3", "valor": "50 kΩ"}
        ],
        "fontes": [
            {"id": "V1", "valor": "+12 V"},
            {"id": "V2", "valor": "-12 V"},
            {"id": "V3", "valor": "+10 V"}
        ],
        "capacitores": [],
        "diodos": []
    },

    "analise_numerica": {
        "ganho_teorico": -5.0,
        "Vout_teorico_V": -50.0,
        "Vout_real_V": -12.0,
        "tensoes_V": {
            "entrada_inversora": 0.0,
            "entrada_nao_inversora": 0.0,
            "saida_ampop": -12.0
        },
        "correntes_A": {
            "R1": 0.001,
            "R2": 0.0,
            "R3": 0.00024
        },
        "potencias_W": {
            "R1": 0.01,
            "R2": 0.0,
            "R3": 0.00288
        }
    },

    "analise_textual": {
        "regiao_operacao": "Saturação negativa",
        "terra_virtual": true
    }
}

Regras:
- Todos os valores numéricos devem ser números JSON válidos.
- Exemplo:
    "Vout_V": -12.0
    "correntes_A": {
        "R1": 0.0
    }
    "potencias_W": {
        "R1": 0.0
    }
"""
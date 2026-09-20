"""Prompts enviados ao modelo multimodal, em três níveis de estruturação
(direto, guiado por passos, procedimento detalhado) para comparar se um
prompt mais estruturado melhora a qualidade da resposta.

VERSÃO DOS PROMPTS: esta é a versão usada na rodada do Qwen3-VL 8B Instruct
salva em resultados/resultados_20260919_151822.csv (24 circuitos, PT e EN) e
na rodada final do Gemini 2.5 Flash, para os dois modelos serem comparados
com prompts idênticos. O JSON de exemplo usa números fictícios (não é o
gabarito de nenhum circuito do dataset); sem o campo 'raciocinio' e sem a
regra de unidades.

Os três pedem o mesmo formato de JSON de saída. O ponto de atenção é a
convenção de nomes em 'tensoes_V': os nós de tensão de interesse são
marcados na IMAGEM com letras (A, B, C...) — ver os arquivos em imagens/ —
e o prompt instrui o modelo a usar exatamente essas letras como chave, em
vez de inventar nomes descritivos (ex: 'entrada_inversora'). Isso existe
porque nomes descritivos variam de circuito para circuito e explodiam o
número de colunas do CSV de resultados; com letras fixas por imagem, os
gabaritos em gabaritos/*.json e as respostas da IA usam o mesmo vocabulário
pequeno (A/B/C), permitindo comparação direta. 'correntes_A'/'potencias_W'
continuam usando o designador do componente exibido na imagem (R1, D1...),
que já é naturalmente estável.

'regiao_operacao' (dentro de 'analise_textual') é um objeto {designador:
estado}, uma entrada por diodo/amp-op do circuito — não uma frase única
para o circuito inteiro. Isso existe porque um circuito pode ter vários
componentes não-lineares em estados diferentes ao mesmo tempo (ex: D1 em
condução, D2 em corte), que uma frase só não descreve de forma comparável
com o gabarito (ver evaluator_textual.py, que compara por componente).

Os três níveis também existem em inglês (sufixo _EN), para o braço
exploratório que testa se o idioma do prompt influencia o desempenho. As
CHAVES do JSON continuam em português mesmo na versão em inglês (o código
— response_splitter.py, evaluator_numerico.py — procura essas chaves
literalmente: 'tensoes_V', 'correntes_A', etc.); só o texto de instrução e
os valores de exemplo (estado de 'regiao_operacao', 'tipo_circuito') mudam
de idioma. ESTADOS_CANONICOS em evaluator_textual.py já reconhece os
termos em inglês usados nos exemplos abaixo (saturation/conduction/cutoff).
"""

#Prompt nível 1
PROMPT_ANALISE_CIRCUITO_1 = """
Analise o circuito eletrônico presente na imagem.

Retorne APENAS um JSON válido no seguinte formato:

{
    "tipo_circuito": "Amplificador Operacional Inversor",

    "componentes": {
        "amplificadores_operacionais": [
            {"id": "U1"}
        ],
        "resistores": [
            {"id": "R1", "valor": "4.7 kΩ"},
            {"id": "R2", "valor": "22 kΩ"},
            {"id": "R3", "valor": "680 Ω"}
        ],
        "fontes": [
            {"id": "V1", "valor": "+7.5 V"},
            {"id": "V2", "valor": "-9.1 V"},
            {"id": "V3", "valor": "+3.3 V"}
        ],
        "capacitores": [],
        "diodos": [],
        "indutores": []
    },

    "analise_numerica": {
        "ganho_teorico": -3.7,
        "Vout_teorico_V": -8.42,
        "Vout_real_V": -6.13,
        "tensoes_V": {
            "A": 3.21,
            "B": -1.47,
            "C": 9.88
        },
        "correntes_A": {
            "R1": 0.00317,
            "R2": 0.00842,
            "R3": 0.00059
        },
        "potencias_W": {
            "R1": 0.0231,
            "R2": 0.0517,
            "R3": 0.00174
        }
    },

    "analise_textual": {
        "regiao_operacao": {
            "U1": "Saturação negativa"
        },
        "terra_virtual": true
    }
}

Regras:
- A imagem contém rótulos de nó (letras como A, B, C...) marcando os pontos onde a tensão deve ser calculada. Use EXATAMENTE esses rótulos como chaves em "tensoes_V" — não invente nomes descritivos (ex: não use "entrada_inversora"; use a letra mostrada na imagem, ex: "A").
- Para "correntes_A" e "potencias_W", use o designador do componente exibido na imagem (ex: "R1", "D1").
- "regiao_operacao" é um objeto (dict), não uma frase única: inclua uma entrada para CADA diodo e CADA amplificador operacional do circuito, usando o designador exibido na imagem como chave (ex: "D1", "U1") e o estado desse componente como valor (ex: "Condução Direta", "Corte", "Saturação Positiva", "Saturação Negativa", "Linear"). Resistores, capacitores e indutores não entram nesse objeto.
- Todos os valores numéricos devem ser números JSON válidos.
- Exemplo:
    "Vout_V": 2.75
    "correntes_A": {
        "R1": 0.00415
    }
    "potencias_W": {
        "R1": 0.0293
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
            {"id": "U1"}
        ],
        "resistores": [
            {"id": "R1", "valor": "4.7 kΩ"},
            {"id": "R2", "valor": "22 kΩ"},
            {"id": "R3", "valor": "680 Ω"}
        ],
        "fontes": [
            {"id": "V1", "valor": "+7.5 V"},
            {"id": "V2", "valor": "-9.1 V"},
            {"id": "V3", "valor": "+3.3 V"}
        ],
        "capacitores": [],
        "diodos": [],
        "indutores": []
    },

    "analise_numerica": {
        "ganho_teorico": -3.7,
        "Vout_teorico_V": -8.42,
        "Vout_real_V": -6.13,
        "tensoes_V": {
            "A": 3.21,
            "B": -1.47,
            "C": 9.88
        },
        "correntes_A": {
            "R1": 0.00317,
            "R2": 0.00842,
            "R3": 0.00059
        },
        "potencias_W": {
            "R1": 0.0231,
            "R2": 0.0517,
            "R3": 0.00174
        }
    },

    "analise_textual": {
        "regiao_operacao": {
            "U1": "Saturação negativa"
        },
        "terra_virtual": true
    }
}
- A imagem contém rótulos de nó (letras como A, B, C...) marcando os pontos onde a tensão deve ser calculada. Use EXATAMENTE esses rótulos como chaves em "tensoes_V" — não invente nomes descritivos. Para "correntes_A" e "potencias_W", use o designador do componente exibido na imagem (ex: "R1", "D1").
- "regiao_operacao" é um objeto (dict), não uma frase única: inclua uma entrada para CADA diodo e CADA amplificador operacional do circuito, usando o designador exibido na imagem como chave (ex: "D1", "U1") e o estado desse componente como valor (ex: "Condução Direta", "Corte", "Saturação Positiva", "Saturação Negativa", "Linear"). Resistores, capacitores e indutores não entram nesse objeto.
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
            {"id": "U1"}
        ],
        "resistores": [
            {"id": "R1", "valor": "4.7 kΩ"},
            {"id": "R2", "valor": "22 kΩ"},
            {"id": "R3", "valor": "680 Ω"}
        ],
        "fontes": [
            {"id": "V1", "valor": "+7.5 V"},
            {"id": "V2", "valor": "-9.1 V"},
            {"id": "V3", "valor": "+3.3 V"}
        ],
        "capacitores": [],
        "diodos": [],
        "indutores": []
    },

    "analise_numerica": {
        "ganho_teorico": -3.7,
        "Vout_teorico_V": -8.42,
        "Vout_real_V": -6.13,
        "tensoes_V": {
            "A": 3.21,
            "B": -1.47,
            "C": 9.88
        },
        "correntes_A": {
            "R1": 0.00317,
            "R2": 0.00842,
            "R3": 0.00059
        },
        "potencias_W": {
            "R1": 0.0231,
            "R2": 0.0517,
            "R3": 0.00174
        }
    },

    "analise_textual": {
        "regiao_operacao": {
            "U1": "Saturação negativa"
        },
        "terra_virtual": true
    }
}
- A imagem contém rótulos de nó (letras como A, B, C...) marcando os pontos onde a tensão deve ser calculada. Use EXATAMENTE esses rótulos como chaves em "tensoes_V" — não invente nomes descritivos. Para "correntes_A" e "potencias_W", use o designador do componente exibido na imagem (ex: "R1", "D1").
- "regiao_operacao" é um objeto (dict), não uma frase única: inclua uma entrada para CADA diodo e CADA amplificador operacional do circuito, usando o designador exibido na imagem como chave (ex: "D1", "U1") e o estado desse componente como valor (ex: "Condução Direta", "Corte", "Saturação Positiva", "Saturação Negativa", "Linear"). Resistores, capacitores e indutores não entram nesse objeto.
- Caso não seja possível calcular, use "não determinado" ou null para valores individuais.
- Não escreva explicações fora do JSON.
"""

# =========================================================
# Versões em inglês — mesmo formato de JSON (chaves em português mantidas
# de propósito, ver docstring do módulo), só o texto de instrução e os
# valores de exemplo mudam de idioma.
# =========================================================

#Prompt level 1 (English)
PROMPT_ANALISE_CIRCUITO_1_EN = """
Analyze the electronic circuit shown in the image.

Return ONLY a valid JSON in the following format:

{
    "tipo_circuito": "Inverting Operational Amplifier",

    "componentes": {
        "amplificadores_operacionais": [
            {"id": "U1"}
        ],
        "resistores": [
            {"id": "R1", "valor": "4.7 kΩ"},
            {"id": "R2", "valor": "22 kΩ"},
            {"id": "R3", "valor": "680 Ω"}
        ],
        "fontes": [
            {"id": "V1", "valor": "+7.5 V"},
            {"id": "V2", "valor": "-9.1 V"},
            {"id": "V3", "valor": "+3.3 V"}
        ],
        "capacitores": [],
        "diodos": [],
        "indutores": []
    },

    "analise_numerica": {
        "ganho_teorico": -3.7,
        "Vout_teorico_V": -8.42,
        "Vout_real_V": -6.13,
        "tensoes_V": {
            "A": 3.21,
            "B": -1.47,
            "C": 9.88
        },
        "correntes_A": {
            "R1": 0.00317,
            "R2": 0.00842,
            "R3": 0.00059
        },
        "potencias_W": {
            "R1": 0.0231,
            "R2": 0.0517,
            "R3": 0.00174
        }
    },

    "analise_textual": {
        "regiao_operacao": {
            "U1": "Negative Saturation"
        },
        "terra_virtual": true
    }
}

Rules:
- The image contains node labels (letters like A, B, C...) marking the points where the voltage must be calculated. Use EXACTLY those labels as keys in "tensoes_V" — do not invent descriptive names (e.g. do not use "inverting_input"; use the letter shown in the image, e.g. "A").
- For "correntes_A" and "potencias_W", use the component designator shown in the image (e.g. "R1", "D1").
- "regiao_operacao" is an object (dict), not a single sentence: include one entry for EACH diode and EACH operational amplifier in the circuit, using the designator shown in the image as the key (e.g. "D1", "U1") and that component's state as the value (e.g. "Forward Conduction", "Cutoff", "Positive Saturation", "Negative Saturation", "Linear"). Resistors, capacitors and inductors do not belong in this object.
- All numeric values must be valid JSON numbers.
- Example:
    "Vout_V": 2.75
    "correntes_A": {
        "R1": 0.00415
    }
    "potencias_W": {
        "R1": 0.0293
    }
- If a value cannot be calculated, use "undetermined" or null for individual values.
- Do not write explanations outside the JSON.
"""

#Prompt level 2 (English)
PROMPT_ANALISE_CIRCUITO_2_EN = """
Analyze the electronic circuit.

1. Identify the components present.

2. Classify the type of circuit.

3. Determine the region of operation.

4. Solve the circuit electrically.

5. Calculate gain, voltages, currents and powers when possible.

Return ONLY a valid JSON in the following format:

{
    "tipo_circuito": "Inverting Operational Amplifier",

    "componentes": {
        "amplificadores_operacionais": [
            {"id": "U1"}
        ],
        "resistores": [
            {"id": "R1", "valor": "4.7 kΩ"},
            {"id": "R2", "valor": "22 kΩ"},
            {"id": "R3", "valor": "680 Ω"}
        ],
        "fontes": [
            {"id": "V1", "valor": "+7.5 V"},
            {"id": "V2", "valor": "-9.1 V"},
            {"id": "V3", "valor": "+3.3 V"}
        ],
        "capacitores": [],
        "diodos": [],
        "indutores": []
    },

    "analise_numerica": {
        "ganho_teorico": -3.7,
        "Vout_teorico_V": -8.42,
        "Vout_real_V": -6.13,
        "tensoes_V": {
            "A": 3.21,
            "B": -1.47,
            "C": 9.88
        },
        "correntes_A": {
            "R1": 0.00317,
            "R2": 0.00842,
            "R3": 0.00059
        },
        "potencias_W": {
            "R1": 0.0231,
            "R2": 0.0517,
            "R3": 0.00174
        }
    },

    "analise_textual": {
        "regiao_operacao": {
            "U1": "Negative Saturation"
        },
        "terra_virtual": true
    }
}
- The image contains node labels (letters like A, B, C...) marking the points where the voltage must be calculated. Use EXACTLY those labels as keys in "tensoes_V" — do not invent descriptive names. For "correntes_A" and "potencias_W", use the component designator shown in the image (e.g. "R1", "D1").
- "regiao_operacao" is an object (dict), not a single sentence: include one entry for EACH diode and EACH operational amplifier in the circuit, using the designator shown in the image as the key (e.g. "D1", "U1") and that component's state as the value (e.g. "Forward Conduction", "Cutoff", "Positive Saturation", "Negative Saturation", "Linear"). Resistors, capacitors and inductors do not belong in this object.
- If a value cannot be calculated, use "undetermined" or null for individual values.
- Do not write explanations outside the JSON.
"""

#Prompt level 3 (English)
PROMPT_ANALISE_CIRCUITO_3_EN = """
OBJECTIVE

Analyze the electronic circuit shown in detail.

PROCEDURE

1. Identify all components present.

2. Classify the circuit topology.

3. Identify the main non-linear component.

4. Determine the region of operation of each non-linear component.

5. Check hypotheses such as:
   - virtual ground;
   - saturation;
   - diode conduction or blocking;
   - feedback.

6. Solve the circuit by applying the appropriate electrical relations.

7. Calculate:
   - gain;
   - voltages;
   - currents;
   - powers.

8. Check whether the results obtained are physically consistent.

9. If a value cannot be determined from the image alone, use null or "undetermined" and justify it.

10. Fill in the JSON exactly in the requested format:

{
    "tipo_circuito": "Inverting Operational Amplifier",

    "componentes": {
        "amplificadores_operacionais": [
            {"id": "U1"}
        ],
        "resistores": [
            {"id": "R1", "valor": "4.7 kΩ"},
            {"id": "R2", "valor": "22 kΩ"},
            {"id": "R3", "valor": "680 Ω"}
        ],
        "fontes": [
            {"id": "V1", "valor": "+7.5 V"},
            {"id": "V2", "valor": "-9.1 V"},
            {"id": "V3", "valor": "+3.3 V"}
        ],
        "capacitores": [],
        "diodos": [],
        "indutores": []
    },

    "analise_numerica": {
        "ganho_teorico": -3.7,
        "Vout_teorico_V": -8.42,
        "Vout_real_V": -6.13,
        "tensoes_V": {
            "A": 3.21,
            "B": -1.47,
            "C": 9.88
        },
        "correntes_A": {
            "R1": 0.00317,
            "R2": 0.00842,
            "R3": 0.00059
        },
        "potencias_W": {
            "R1": 0.0231,
            "R2": 0.0517,
            "R3": 0.00174
        }
    },

    "analise_textual": {
        "regiao_operacao": {
            "U1": "Negative Saturation"
        },
        "terra_virtual": true
    }
}
- The image contains node labels (letters like A, B, C...) marking the points where the voltage must be calculated. Use EXACTLY those labels as keys in "tensoes_V" — do not invent descriptive names. For "correntes_A" and "potencias_W", use the component designator shown in the image (e.g. "R1", "D1").
- "regiao_operacao" is an object (dict), not a single sentence: include one entry for EACH diode and EACH operational amplifier in the circuit, using the designator shown in the image as the key (e.g. "D1", "U1") and that component's state as the value (e.g. "Forward Conduction", "Cutoff", "Positive Saturation", "Negative Saturation", "Linear"). Resistors, capacitors and inductors do not belong in this object.
- If a value cannot be calculated, use "undetermined" or null for individual values.
- Do not write explanations outside the JSON.
"""

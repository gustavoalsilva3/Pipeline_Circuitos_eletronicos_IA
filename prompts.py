"""Prompts enviados ao modelo multimodal, em três níveis de estruturação
(direto, guiado por passos, procedimento detalhado) para comparar se um
prompt mais estruturado melhora a qualidade da resposta.

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

Os exemplos de valor no JSON dos prompts são marcadores não numéricos
(<...>) de propósito: em testes, os modelos copiavam números de exemplo (ex: valores de
componentes e tensões) em vez de ler a imagem, e o "kΩ" do exemplo induzia
erro de escala (as imagens mostram valores sem unidade, em Ω).

Um campo de texto livre 'raciocinio' antes dos números foi testado e
retirado: em modelos pequenos abriu espaço para loops de repetição, sem
ganho visível de acerto.

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
            {"id": "R1", "valor": "<valor lido na imagem, com unidade>"},
            {"id": "R2", "valor": "<valor lido na imagem, com unidade>"},
            {"id": "R3", "valor": "<valor lido na imagem, com unidade>"}
        ],
        "fontes": [
            {"id": "V1", "valor": "<valor lido na imagem, com unidade>"},
            {"id": "V2", "valor": "<valor lido na imagem, com unidade>"},
            {"id": "V3", "valor": "<valor lido na imagem, com unidade>"}
        ],
        "capacitores": [],
        "diodos": [],
        "indutores": []
    },

    "analise_numerica": {
        "ganho_teorico": "<número ou null>",
        "Vout_teorico_V": "<número em V>",
        "Vout_real_V": "<número em V>",
        "tensoes_V": {
            "A": "<número em V>",
            "B": "<número em V>",
            "C": "<número em V>"
        },
        "correntes_A": {
            "R1": "<número em A>",
            "R2": "<número em A>",
            "R3": "<número em A>"
        },
        "potencias_W": {
            "R1": "<número em W>",
            "R2": "<número em W>",
            "R3": "<número em W>"
        }
    },

    "analise_textual": {
        "regiao_operacao": {
            "U1": "<estado do componente>"
        },
        "terra_virtual": "<true ou false>"
    }
}

Regras:
- A imagem contém rótulos de nó (letras como A, B, C...) marcando os pontos onde a tensão deve ser calculada. Use EXATAMENTE esses rótulos como chaves em "tensoes_V" — não invente nomes descritivos (ex: não use "entrada_inversora"; use a letra mostrada na imagem, ex: "A").
- Para "correntes_A" e "potencias_W", use o designador do componente exibido na imagem (ex: "R1", "D1").
- "regiao_operacao" é um objeto (dict), não uma frase única: inclua uma entrada para CADA diodo e CADA amplificador operacional do circuito, usando o designador exibido na imagem como chave (ex: "D1", "U1") e o estado desse componente como valor (ex: "Condução Direta", "Corte", "Saturação Positiva", "Saturação Negativa", "Linear"). Resistores, capacitores e indutores não entram nesse objeto.
- Todos os valores numéricos devem ser números JSON válidos.
- Valores de componentes: escreva sempre com unidade. Se a imagem mostrar o número sem unidade, use a unidade padrão do simulador (resistores em Ω, fontes em V, capacitores em F, indutores em H) e NÃO acrescente prefixos (k, m, M...) que não estejam escritos na imagem (ex: se a imagem mostra 20, escreva "20 Ω", nunca "20 kΩ"). Em "tensoes_V", "correntes_A" e "potencias_W" use números puros em V, A e W.
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
            {"id": "R1", "valor": "<valor lido na imagem, com unidade>"},
            {"id": "R2", "valor": "<valor lido na imagem, com unidade>"},
            {"id": "R3", "valor": "<valor lido na imagem, com unidade>"}
        ],
        "fontes": [
            {"id": "V1", "valor": "<valor lido na imagem, com unidade>"},
            {"id": "V2", "valor": "<valor lido na imagem, com unidade>"},
            {"id": "V3", "valor": "<valor lido na imagem, com unidade>"}
        ],
        "capacitores": [],
        "diodos": [],
        "indutores": []
    },

    "analise_numerica": {
        "ganho_teorico": "<número ou null>",
        "Vout_teorico_V": "<número em V>",
        "Vout_real_V": "<número em V>",
        "tensoes_V": {
            "A": "<número em V>",
            "B": "<número em V>",
            "C": "<número em V>"
        },
        "correntes_A": {
            "R1": "<número em A>",
            "R2": "<número em A>",
            "R3": "<número em A>"
        },
        "potencias_W": {
            "R1": "<número em W>",
            "R2": "<número em W>",
            "R3": "<número em W>"
        }
    },

    "analise_textual": {
        "regiao_operacao": {
            "U1": "<estado do componente>"
        },
        "terra_virtual": "<true ou false>"
    }
}
- A imagem contém rótulos de nó (letras como A, B, C...) marcando os pontos onde a tensão deve ser calculada. Use EXATAMENTE esses rótulos como chaves em "tensoes_V" — não invente nomes descritivos. Para "correntes_A" e "potencias_W", use o designador do componente exibido na imagem (ex: "R1", "D1").
- "regiao_operacao" é um objeto (dict), não uma frase única: inclua uma entrada para CADA diodo e CADA amplificador operacional do circuito, usando o designador exibido na imagem como chave (ex: "D1", "U1") e o estado desse componente como valor (ex: "Condução Direta", "Corte", "Saturação Positiva", "Saturação Negativa", "Linear"). Resistores, capacitores e indutores não entram nesse objeto.
- Valores de componentes: escreva sempre com unidade. Se a imagem mostrar o número sem unidade, use a unidade padrão do simulador (resistores em Ω, fontes em V, capacitores em F, indutores em H) e NÃO acrescente prefixos (k, m, M...) que não estejam escritos na imagem (ex: se a imagem mostra 20, escreva "20 Ω", nunca "20 kΩ"). Em "tensoes_V", "correntes_A" e "potencias_W" use números puros em V, A e W.
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
            {"id": "R1", "valor": "<valor lido na imagem, com unidade>"},
            {"id": "R2", "valor": "<valor lido na imagem, com unidade>"},
            {"id": "R3", "valor": "<valor lido na imagem, com unidade>"}
        ],
        "fontes": [
            {"id": "V1", "valor": "<valor lido na imagem, com unidade>"},
            {"id": "V2", "valor": "<valor lido na imagem, com unidade>"},
            {"id": "V3", "valor": "<valor lido na imagem, com unidade>"}
        ],
        "capacitores": [],
        "diodos": [],
        "indutores": []
    },

    "analise_numerica": {
        "ganho_teorico": "<número ou null>",
        "Vout_teorico_V": "<número em V>",
        "Vout_real_V": "<número em V>",
        "tensoes_V": {
            "A": "<número em V>",
            "B": "<número em V>",
            "C": "<número em V>"
        },
        "correntes_A": {
            "R1": "<número em A>",
            "R2": "<número em A>",
            "R3": "<número em A>"
        },
        "potencias_W": {
            "R1": "<número em W>",
            "R2": "<número em W>",
            "R3": "<número em W>"
        }
    },

    "analise_textual": {
        "regiao_operacao": {
            "U1": "<estado do componente>"
        },
        "terra_virtual": "<true ou false>"
    }
}
- A imagem contém rótulos de nó (letras como A, B, C...) marcando os pontos onde a tensão deve ser calculada. Use EXATAMENTE esses rótulos como chaves em "tensoes_V" — não invente nomes descritivos. Para "correntes_A" e "potencias_W", use o designador do componente exibido na imagem (ex: "R1", "D1").
- "regiao_operacao" é um objeto (dict), não uma frase única: inclua uma entrada para CADA diodo e CADA amplificador operacional do circuito, usando o designador exibido na imagem como chave (ex: "D1", "U1") e o estado desse componente como valor (ex: "Condução Direta", "Corte", "Saturação Positiva", "Saturação Negativa", "Linear"). Resistores, capacitores e indutores não entram nesse objeto.
- Valores de componentes: escreva sempre com unidade. Se a imagem mostrar o número sem unidade, use a unidade padrão do simulador (resistores em Ω, fontes em V, capacitores em F, indutores em H) e NÃO acrescente prefixos (k, m, M...) que não estejam escritos na imagem (ex: se a imagem mostra 20, escreva "20 Ω", nunca "20 kΩ"). Em "tensoes_V", "correntes_A" e "potencias_W" use números puros em V, A e W.
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
            {"id": "R1", "valor": "<value read from the image, with unit>"},
            {"id": "R2", "valor": "<value read from the image, with unit>"},
            {"id": "R3", "valor": "<value read from the image, with unit>"}
        ],
        "fontes": [
            {"id": "V1", "valor": "<value read from the image, with unit>"},
            {"id": "V2", "valor": "<value read from the image, with unit>"},
            {"id": "V3", "valor": "<value read from the image, with unit>"}
        ],
        "capacitores": [],
        "diodos": [],
        "indutores": []
    },

    "analise_numerica": {
        "ganho_teorico": "<number or null>",
        "Vout_teorico_V": "<number in V>",
        "Vout_real_V": "<number in V>",
        "tensoes_V": {
            "A": "<number in V>",
            "B": "<number in V>",
            "C": "<number in V>"
        },
        "correntes_A": {
            "R1": "<number in A>",
            "R2": "<number in A>",
            "R3": "<number in A>"
        },
        "potencias_W": {
            "R1": "<number in W>",
            "R2": "<number in W>",
            "R3": "<number in W>"
        }
    },

    "analise_textual": {
        "regiao_operacao": {
            "U1": "<component state>"
        },
        "terra_virtual": "<true or false>"
    }
}

Rules:
- The image contains node labels (letters like A, B, C...) marking the points where the voltage must be calculated. Use EXACTLY those labels as keys in "tensoes_V" — do not invent descriptive names (e.g. do not use "inverting_input"; use the letter shown in the image, e.g. "A").
- For "correntes_A" and "potencias_W", use the component designator shown in the image (e.g. "R1", "D1").
- "regiao_operacao" is an object (dict), not a single sentence: include one entry for EACH diode and EACH operational amplifier in the circuit, using the designator shown in the image as the key (e.g. "D1", "U1") and that component's state as the value (e.g. "Forward Conduction", "Cutoff", "Positive Saturation", "Negative Saturation", "Linear"). Resistors, capacitors and inductors do not belong in this object.
- All numeric values must be valid JSON numbers.
- Component values: always write them with a unit. If the image shows the number without a unit, use the simulator default unit (resistors in Ω, sources in V, capacitors in F, inductors in H) and do NOT add prefixes (k, m, M...) that are not written in the image (e.g. if the image shows 20, write "20 Ω", never "20 kΩ"). In "tensoes_V", "correntes_A" and "potencias_W" use plain numbers in V, A and W.
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
            {"id": "R1", "valor": "<value read from the image, with unit>"},
            {"id": "R2", "valor": "<value read from the image, with unit>"},
            {"id": "R3", "valor": "<value read from the image, with unit>"}
        ],
        "fontes": [
            {"id": "V1", "valor": "<value read from the image, with unit>"},
            {"id": "V2", "valor": "<value read from the image, with unit>"},
            {"id": "V3", "valor": "<value read from the image, with unit>"}
        ],
        "capacitores": [],
        "diodos": [],
        "indutores": []
    },

    "analise_numerica": {
        "ganho_teorico": "<number or null>",
        "Vout_teorico_V": "<number in V>",
        "Vout_real_V": "<number in V>",
        "tensoes_V": {
            "A": "<number in V>",
            "B": "<number in V>",
            "C": "<number in V>"
        },
        "correntes_A": {
            "R1": "<number in A>",
            "R2": "<number in A>",
            "R3": "<number in A>"
        },
        "potencias_W": {
            "R1": "<number in W>",
            "R2": "<number in W>",
            "R3": "<number in W>"
        }
    },

    "analise_textual": {
        "regiao_operacao": {
            "U1": "<component state>"
        },
        "terra_virtual": "<true or false>"
    }
}
- The image contains node labels (letters like A, B, C...) marking the points where the voltage must be calculated. Use EXACTLY those labels as keys in "tensoes_V" — do not invent descriptive names. For "correntes_A" and "potencias_W", use the component designator shown in the image (e.g. "R1", "D1").
- "regiao_operacao" is an object (dict), not a single sentence: include one entry for EACH diode and EACH operational amplifier in the circuit, using the designator shown in the image as the key (e.g. "D1", "U1") and that component's state as the value (e.g. "Forward Conduction", "Cutoff", "Positive Saturation", "Negative Saturation", "Linear"). Resistors, capacitors and inductors do not belong in this object.
- Component values: always write them with a unit. If the image shows the number without a unit, use the simulator default unit (resistors in Ω, sources in V, capacitors in F, inductors in H) and do NOT add prefixes (k, m, M...) that are not written in the image (e.g. if the image shows 20, write "20 Ω", never "20 kΩ"). In "tensoes_V", "correntes_A" and "potencias_W" use plain numbers in V, A and W.
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
            {"id": "R1", "valor": "<value read from the image, with unit>"},
            {"id": "R2", "valor": "<value read from the image, with unit>"},
            {"id": "R3", "valor": "<value read from the image, with unit>"}
        ],
        "fontes": [
            {"id": "V1", "valor": "<value read from the image, with unit>"},
            {"id": "V2", "valor": "<value read from the image, with unit>"},
            {"id": "V3", "valor": "<value read from the image, with unit>"}
        ],
        "capacitores": [],
        "diodos": [],
        "indutores": []
    },

    "analise_numerica": {
        "ganho_teorico": "<number or null>",
        "Vout_teorico_V": "<number in V>",
        "Vout_real_V": "<number in V>",
        "tensoes_V": {
            "A": "<number in V>",
            "B": "<number in V>",
            "C": "<number in V>"
        },
        "correntes_A": {
            "R1": "<number in A>",
            "R2": "<number in A>",
            "R3": "<number in A>"
        },
        "potencias_W": {
            "R1": "<number in W>",
            "R2": "<number in W>",
            "R3": "<number in W>"
        }
    },

    "analise_textual": {
        "regiao_operacao": {
            "U1": "<component state>"
        },
        "terra_virtual": "<true or false>"
    }
}
- The image contains node labels (letters like A, B, C...) marking the points where the voltage must be calculated. Use EXACTLY those labels as keys in "tensoes_V" — do not invent descriptive names. For "correntes_A" and "potencias_W", use the component designator shown in the image (e.g. "R1", "D1").
- "regiao_operacao" is an object (dict), not a single sentence: include one entry for EACH diode and EACH operational amplifier in the circuit, using the designator shown in the image as the key (e.g. "D1", "U1") and that component's state as the value (e.g. "Forward Conduction", "Cutoff", "Positive Saturation", "Negative Saturation", "Linear"). Resistors, capacitors and inductors do not belong in this object.
- Component values: always write them with a unit. If the image shows the number without a unit, use the simulator default unit (resistors in Ω, sources in V, capacitors in F, inductors in H) and do NOT add prefixes (k, m, M...) that are not written in the image (e.g. if the image shows 20, write "20 Ω", never "20 kΩ"). In "tensoes_V", "correntes_A" and "potencias_W" use plain numbers in V, A and W.
- If a value cannot be calculated, use "undetermined" or null for individual values.
- Do not write explanations outside the JSON.
"""


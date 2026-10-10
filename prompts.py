"""Prompts enviados ao modelo multimodal, em três níveis de estruturação
(direto, guiado por passos, procedimento detalhado) para comparar se um
prompt mais estruturado melhora a qualidade da resposta.

VERSÃO DOS PROMPTS: é exatamente o texto enviado aos quatro modelos na rodada
oficial (rodada_oficial/), executada em 27/09/2026. Os seis prompts declaram
a hipótese de idealidade dos componentes sob a qual o gabarito foi calculado
e trazem o esquema de saída em JSON apenas com marcadores entre < > (ex:
<numero a preencher>), sem valores de exemplo preenchidos, para que o modelo
não os copie em vez de calcular. Alterar este texto muda o experimento: o
Apêndice A do TFC é gerado daqui (gerar_apendice_prompts.py).

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
o dos marcadores mudam de idioma. ESTADOS_CANONICOS em evaluator_textual.py
reconhece os termos de estado em português e em inglês
(saturation/conduction/cutoff).
"""

#Prompt nível 1
PROMPT_ANALISE_CIRCUITO_1 = """
Analise o circuito eletrônico presente na imagem.

Considere todos os componentes IDEAIS:
- diodo: queda de tensão nula em condução direta, corrente nula em corte;
- amplificador operacional: ganho infinito, corrente de saída ilimitada e saturação exatamente nas tensões de alimentação;
- em regime permanente de corrente contínua, indutor como curto-circuito e capacitor como circuito aberto.

Retorne APENAS um JSON válido no seguinte formato:

{
    "tipo_circuito": "<tipo do circuito identificado>",

    "componentes": {
        "amplificadores_operacionais": [
            {"id": "<designador na imagem>"}
        ],
        "resistores": [
            {"id": "<designador na imagem>", "valor": "<valor e unidade exatamente como na imagem>"},
            {"id": "<designador na imagem>", "valor": "<valor e unidade exatamente como na imagem>"}
        ],
        "fontes": [
            {"id": "<designador na imagem>", "valor": "<valor e unidade exatamente como na imagem>"},
            {"id": "<designador na imagem>", "valor": "<valor e unidade exatamente como na imagem>"}
        ],
        "capacitores": [
            {"id": "<designador na imagem>", "valor": "<valor e unidade exatamente como na imagem>"}
        ],
        "diodos": [
            {"id": "<designador na imagem>"}
        ],
        "indutores": [
            {"id": "<designador na imagem>", "valor": "<valor e unidade exatamente como na imagem>"}
        ]
    },

    "analise_numerica": {
        "ganho_teorico": <numero a preencher>,
        "Vout_teorico_V": <tensao de saida obtida pela analise do circuito, sem aplicar o limite da alimentacao>,
        "Vout_real_V": <tensao de saida final; igual a teorica quando nao ha saturacao>,
        "tensoes_V": {
            "<rotulo de no da imagem>": <numero a preencher>,
            "<rotulo de no da imagem>": <numero a preencher>
        },
        "correntes_A": {
            "<designador na imagem>": <numero a preencher>,
            "<designador na imagem>": <numero a preencher>
        },
        "potencias_W": {
            "<designador na imagem>": <numero a preencher>,
            "<designador na imagem>": <numero a preencher>
        }
    },

    "analise_textual": {
        "regiao_operacao": {
            "<designador na imagem>": "<estado desse componente>"
        },
        "terra_virtual": <true ou false>
    }
}

Regras:
- A imagem contém rótulos de nó (letras como A, B, C...) marcando os pontos onde a tensão deve ser calculada. Use EXATAMENTE esses rótulos como chaves em "tensoes_V" — não invente nomes descritivos (ex: não use "entrada_inversora"; use a letra mostrada na imagem, ex: "A").
- Para "correntes_A" e "potencias_W", use o designador do componente exibido na imagem (ex: "R1", "D1").
- "regiao_operacao" é um objeto (dict), não uma frase única: inclua uma entrada para CADA diodo e CADA amplificador operacional do circuito, usando o designador exibido na imagem como chave (ex: "D1", "U1") e o estado desse componente como valor (ex: "Condução Direta", "Corte", "Saturação Positiva", "Saturação Negativa", "Linear"). Resistores, capacitores e indutores não entram nesse objeto.
- O bloco acima é um MODELO, não um exemplo resolvido: cada marcador entre < > indica onde inserir o valor obtido a partir DESTA imagem. Nunca copie um marcador para a resposta.
- Todos os valores numéricos devem ser números JSON válidos, sem aspas e sem unidade (a unidade já está no nome da chave).
- Em "componentes", o campo "valor" é texto e deve reproduzir a unidade exatamente como aparece na imagem; se a imagem mostra apenas o número, não acrescente prefixo (não escreva "10 kΩ" onde a imagem mostra "10").
- O número de itens em cada lista e em cada dicionário depende do circuito da imagem, não do modelo acima.
- Caso não seja possível calcular, use "não determinado" ou null para valores individuais.
- Não escreva explicações fora do JSON.
"""

#Prompt nível 2
PROMPT_ANALISE_CIRCUITO_2 = """
Analise o circuito eletrônico.

Considere todos os componentes IDEAIS:
- diodo: queda de tensão nula em condução direta, corrente nula em corte;
- amplificador operacional: ganho infinito, corrente de saída ilimitada e saturação exatamente nas tensões de alimentação;
- em regime permanente de corrente contínua, indutor como curto-circuito e capacitor como circuito aberto.

1. Identifique os componentes presentes.

2. Classifique o tipo do circuito.

3. Determine a região de operação.

4. Resolva eletricamente o circuito.

5. Calcule ganho, tensões, correntes e potências quando possível.

Retorne APENAS um JSON válido no seguinte formato:

{
    "tipo_circuito": "<tipo do circuito identificado>",

    "componentes": {
        "amplificadores_operacionais": [
            {"id": "<designador na imagem>"}
        ],
        "resistores": [
            {"id": "<designador na imagem>", "valor": "<valor e unidade exatamente como na imagem>"},
            {"id": "<designador na imagem>", "valor": "<valor e unidade exatamente como na imagem>"}
        ],
        "fontes": [
            {"id": "<designador na imagem>", "valor": "<valor e unidade exatamente como na imagem>"},
            {"id": "<designador na imagem>", "valor": "<valor e unidade exatamente como na imagem>"}
        ],
        "capacitores": [
            {"id": "<designador na imagem>", "valor": "<valor e unidade exatamente como na imagem>"}
        ],
        "diodos": [
            {"id": "<designador na imagem>"}
        ],
        "indutores": [
            {"id": "<designador na imagem>", "valor": "<valor e unidade exatamente como na imagem>"}
        ]
    },

    "analise_numerica": {
        "ganho_teorico": <numero a preencher>,
        "Vout_teorico_V": <tensao de saida obtida pela analise do circuito, sem aplicar o limite da alimentacao>,
        "Vout_real_V": <tensao de saida final; igual a teorica quando nao ha saturacao>,
        "tensoes_V": {
            "<rotulo de no da imagem>": <numero a preencher>,
            "<rotulo de no da imagem>": <numero a preencher>
        },
        "correntes_A": {
            "<designador na imagem>": <numero a preencher>,
            "<designador na imagem>": <numero a preencher>
        },
        "potencias_W": {
            "<designador na imagem>": <numero a preencher>,
            "<designador na imagem>": <numero a preencher>
        }
    },

    "analise_textual": {
        "regiao_operacao": {
            "<designador na imagem>": "<estado desse componente>"
        },
        "terra_virtual": <true ou false>
    }
}
- A imagem contém rótulos de nó (letras como A, B, C...) marcando os pontos onde a tensão deve ser calculada. Use EXATAMENTE esses rótulos como chaves em "tensoes_V" — não invente nomes descritivos. Para "correntes_A" e "potencias_W", use o designador do componente exibido na imagem (ex: "R1", "D1").
- "regiao_operacao" é um objeto (dict), não uma frase única: inclua uma entrada para CADA diodo e CADA amplificador operacional do circuito, usando o designador exibido na imagem como chave (ex: "D1", "U1") e o estado desse componente como valor (ex: "Condução Direta", "Corte", "Saturação Positiva", "Saturação Negativa", "Linear"). Resistores, capacitores e indutores não entram nesse objeto.
- O bloco acima é um MODELO, não um exemplo resolvido: cada marcador entre < > indica onde inserir o valor obtido a partir DESTA imagem. Nunca copie um marcador para a resposta.
- Todos os valores numéricos devem ser números JSON válidos, sem aspas e sem unidade (a unidade já está no nome da chave).
- Em "componentes", o campo "valor" é texto e deve reproduzir a unidade exatamente como aparece na imagem; se a imagem mostra apenas o número, não acrescente prefixo (não escreva "10 kΩ" onde a imagem mostra "10").
- O número de itens em cada lista e em cada dicionário depende do circuito da imagem, não do modelo acima.
- Caso não seja possível calcular, use "não determinado" ou null para valores individuais.
- Não escreva explicações fora do JSON.
"""

#Prompt nível 3
PROMPT_ANALISE_CIRCUITO_3 = """
OBJETIVO

Analise detalhadamente o circuito eletrônico apresentado.

Considere todos os componentes IDEAIS:
- diodo: queda de tensão nula em condução direta, corrente nula em corte;
- amplificador operacional: ganho infinito, corrente de saída ilimitada e saturação exatamente nas tensões de alimentação;
- em regime permanente de corrente contínua, indutor como curto-circuito e capacitor como circuito aberto.

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
    "tipo_circuito": "<tipo do circuito identificado>",

    "componentes": {
        "amplificadores_operacionais": [
            {"id": "<designador na imagem>"}
        ],
        "resistores": [
            {"id": "<designador na imagem>", "valor": "<valor e unidade exatamente como na imagem>"},
            {"id": "<designador na imagem>", "valor": "<valor e unidade exatamente como na imagem>"}
        ],
        "fontes": [
            {"id": "<designador na imagem>", "valor": "<valor e unidade exatamente como na imagem>"},
            {"id": "<designador na imagem>", "valor": "<valor e unidade exatamente como na imagem>"}
        ],
        "capacitores": [
            {"id": "<designador na imagem>", "valor": "<valor e unidade exatamente como na imagem>"}
        ],
        "diodos": [
            {"id": "<designador na imagem>"}
        ],
        "indutores": [
            {"id": "<designador na imagem>", "valor": "<valor e unidade exatamente como na imagem>"}
        ]
    },

    "analise_numerica": {
        "ganho_teorico": <numero a preencher>,
        "Vout_teorico_V": <tensao de saida obtida pela analise do circuito, sem aplicar o limite da alimentacao>,
        "Vout_real_V": <tensao de saida final; igual a teorica quando nao ha saturacao>,
        "tensoes_V": {
            "<rotulo de no da imagem>": <numero a preencher>,
            "<rotulo de no da imagem>": <numero a preencher>
        },
        "correntes_A": {
            "<designador na imagem>": <numero a preencher>,
            "<designador na imagem>": <numero a preencher>
        },
        "potencias_W": {
            "<designador na imagem>": <numero a preencher>,
            "<designador na imagem>": <numero a preencher>
        }
    },

    "analise_textual": {
        "regiao_operacao": {
            "<designador na imagem>": "<estado desse componente>"
        },
        "terra_virtual": <true ou false>
    }
}
- A imagem contém rótulos de nó (letras como A, B, C...) marcando os pontos onde a tensão deve ser calculada. Use EXATAMENTE esses rótulos como chaves em "tensoes_V" — não invente nomes descritivos. Para "correntes_A" e "potencias_W", use o designador do componente exibido na imagem (ex: "R1", "D1").
- "regiao_operacao" é um objeto (dict), não uma frase única: inclua uma entrada para CADA diodo e CADA amplificador operacional do circuito, usando o designador exibido na imagem como chave (ex: "D1", "U1") e o estado desse componente como valor (ex: "Condução Direta", "Corte", "Saturação Positiva", "Saturação Negativa", "Linear"). Resistores, capacitores e indutores não entram nesse objeto.
- O bloco acima é um MODELO, não um exemplo resolvido: cada marcador entre < > indica onde inserir o valor obtido a partir DESTA imagem. Nunca copie um marcador para a resposta.
- Todos os valores numéricos devem ser números JSON válidos, sem aspas e sem unidade (a unidade já está no nome da chave).
- Em "componentes", o campo "valor" é texto e deve reproduzir a unidade exatamente como aparece na imagem; se a imagem mostra apenas o número, não acrescente prefixo (não escreva "10 kΩ" onde a imagem mostra "10").
- O número de itens em cada lista e em cada dicionário depende do circuito da imagem, não do modelo acima.
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

Assume all components are IDEAL:
- diode: zero forward voltage drop when conducting, zero current when off;
- operational amplifier: infinite gain, unlimited output current, and saturation exactly at the supply voltages;
- in DC steady state, inductor as a short circuit and capacitor as an open circuit.

Return ONLY a valid JSON in the following format:

{
    "tipo_circuito": "<identified circuit type>",

    "componentes": {
        "amplificadores_operacionais": [
            {"id": "<designator in the image>"}
        ],
        "resistores": [
            {"id": "<designator in the image>", "valor": "<value and unit exactly as in the image>"},
            {"id": "<designator in the image>", "valor": "<value and unit exactly as in the image>"}
        ],
        "fontes": [
            {"id": "<designator in the image>", "valor": "<value and unit exactly as in the image>"},
            {"id": "<designator in the image>", "valor": "<value and unit exactly as in the image>"}
        ],
        "capacitores": [
            {"id": "<designator in the image>", "valor": "<value and unit exactly as in the image>"}
        ],
        "diodos": [
            {"id": "<designator in the image>"}
        ],
        "indutores": [
            {"id": "<designator in the image>", "valor": "<value and unit exactly as in the image>"}
        ]
    },

    "analise_numerica": {
        "ganho_teorico": <number to fill in>,
        "Vout_teorico_V": <output voltage from the circuit analysis, without applying the supply limit>,
        "Vout_real_V": <final output voltage; equals the theoretical one when there is no saturation>,
        "tensoes_V": {
            "<node label from the image>": <number to fill in>,
            "<node label from the image>": <number to fill in>
        },
        "correntes_A": {
            "<designator in the image>": <number to fill in>,
            "<designator in the image>": <number to fill in>
        },
        "potencias_W": {
            "<designator in the image>": <number to fill in>,
            "<designator in the image>": <number to fill in>
        }
    },

    "analise_textual": {
        "regiao_operacao": {
            "<designator in the image>": "<that component's state>"
        },
        "terra_virtual": <true or false>
    }
}

Rules:
- The image contains node labels (letters like A, B, C...) marking the points where the voltage must be calculated. Use EXACTLY those labels as keys in "tensoes_V" — do not invent descriptive names (e.g. do not use "inverting_input"; use the letter shown in the image, e.g. "A").
- For "correntes_A" and "potencias_W", use the component designator shown in the image (e.g. "R1", "D1").
- "regiao_operacao" is an object (dict), not a single sentence: include one entry for EACH diode and EACH operational amplifier in the circuit, using the designator shown in the image as the key (e.g. "D1", "U1") and that component's state as the value (e.g. "Forward Conduction", "Cutoff", "Positive Saturation", "Negative Saturation", "Linear"). Resistors, capacitors and inductors do not belong in this object.
- The block above is a TEMPLATE, not a solved example: each marker between < > indicates where to insert the value obtained from THIS image. Never copy a marker into your answer.
- All numeric values must be valid JSON numbers, without quotes and without units (the unit is already in the key name).
- In "componentes", the "valor" field is text and must reproduce the unit exactly as shown in the image; if the image shows only the number, do not add a prefix (do not write "10 kΩ" where the image shows "10").
- The number of items in each list and dictionary depends on the circuit in the image, not on the template above.
- If a value cannot be calculated, use "undetermined" or null for individual values.
- Do not write explanations outside the JSON.
"""

#Prompt level 2 (English)
PROMPT_ANALISE_CIRCUITO_2_EN = """
Analyze the electronic circuit.

Assume all components are IDEAL:
- diode: zero forward voltage drop when conducting, zero current when off;
- operational amplifier: infinite gain, unlimited output current, and saturation exactly at the supply voltages;
- in DC steady state, inductor as a short circuit and capacitor as an open circuit.

1. Identify the components present.

2. Classify the type of circuit.

3. Determine the region of operation.

4. Solve the circuit electrically.

5. Calculate gain, voltages, currents and powers when possible.

Return ONLY a valid JSON in the following format:

{
    "tipo_circuito": "<identified circuit type>",

    "componentes": {
        "amplificadores_operacionais": [
            {"id": "<designator in the image>"}
        ],
        "resistores": [
            {"id": "<designator in the image>", "valor": "<value and unit exactly as in the image>"},
            {"id": "<designator in the image>", "valor": "<value and unit exactly as in the image>"}
        ],
        "fontes": [
            {"id": "<designator in the image>", "valor": "<value and unit exactly as in the image>"},
            {"id": "<designator in the image>", "valor": "<value and unit exactly as in the image>"}
        ],
        "capacitores": [
            {"id": "<designator in the image>", "valor": "<value and unit exactly as in the image>"}
        ],
        "diodos": [
            {"id": "<designator in the image>"}
        ],
        "indutores": [
            {"id": "<designator in the image>", "valor": "<value and unit exactly as in the image>"}
        ]
    },

    "analise_numerica": {
        "ganho_teorico": <number to fill in>,
        "Vout_teorico_V": <output voltage from the circuit analysis, without applying the supply limit>,
        "Vout_real_V": <final output voltage; equals the theoretical one when there is no saturation>,
        "tensoes_V": {
            "<node label from the image>": <number to fill in>,
            "<node label from the image>": <number to fill in>
        },
        "correntes_A": {
            "<designator in the image>": <number to fill in>,
            "<designator in the image>": <number to fill in>
        },
        "potencias_W": {
            "<designator in the image>": <number to fill in>,
            "<designator in the image>": <number to fill in>
        }
    },

    "analise_textual": {
        "regiao_operacao": {
            "<designator in the image>": "<that component's state>"
        },
        "terra_virtual": <true or false>
    }
}
- The image contains node labels (letters like A, B, C...) marking the points where the voltage must be calculated. Use EXACTLY those labels as keys in "tensoes_V" — do not invent descriptive names. For "correntes_A" and "potencias_W", use the component designator shown in the image (e.g. "R1", "D1").
- "regiao_operacao" is an object (dict), not a single sentence: include one entry for EACH diode and EACH operational amplifier in the circuit, using the designator shown in the image as the key (e.g. "D1", "U1") and that component's state as the value (e.g. "Forward Conduction", "Cutoff", "Positive Saturation", "Negative Saturation", "Linear"). Resistors, capacitors and inductors do not belong in this object.
- The block above is a TEMPLATE, not a solved example: each marker between < > indicates where to insert the value obtained from THIS image. Never copy a marker into your answer.
- All numeric values must be valid JSON numbers, without quotes and without units (the unit is already in the key name).
- In "componentes", the "valor" field is text and must reproduce the unit exactly as shown in the image; if the image shows only the number, do not add a prefix (do not write "10 kΩ" where the image shows "10").
- The number of items in each list and dictionary depends on the circuit in the image, not on the template above.
- If a value cannot be calculated, use "undetermined" or null for individual values.
- Do not write explanations outside the JSON.
"""

#Prompt level 3 (English)
PROMPT_ANALISE_CIRCUITO_3_EN = """
OBJECTIVE

Analyze the electronic circuit shown in detail.

Assume all components are IDEAL:
- diode: zero forward voltage drop when conducting, zero current when off;
- operational amplifier: infinite gain, unlimited output current, and saturation exactly at the supply voltages;
- in DC steady state, inductor as a short circuit and capacitor as an open circuit.

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
    "tipo_circuito": "<identified circuit type>",

    "componentes": {
        "amplificadores_operacionais": [
            {"id": "<designator in the image>"}
        ],
        "resistores": [
            {"id": "<designator in the image>", "valor": "<value and unit exactly as in the image>"},
            {"id": "<designator in the image>", "valor": "<value and unit exactly as in the image>"}
        ],
        "fontes": [
            {"id": "<designator in the image>", "valor": "<value and unit exactly as in the image>"},
            {"id": "<designator in the image>", "valor": "<value and unit exactly as in the image>"}
        ],
        "capacitores": [
            {"id": "<designator in the image>", "valor": "<value and unit exactly as in the image>"}
        ],
        "diodos": [
            {"id": "<designator in the image>"}
        ],
        "indutores": [
            {"id": "<designator in the image>", "valor": "<value and unit exactly as in the image>"}
        ]
    },

    "analise_numerica": {
        "ganho_teorico": <number to fill in>,
        "Vout_teorico_V": <output voltage from the circuit analysis, without applying the supply limit>,
        "Vout_real_V": <final output voltage; equals the theoretical one when there is no saturation>,
        "tensoes_V": {
            "<node label from the image>": <number to fill in>,
            "<node label from the image>": <number to fill in>
        },
        "correntes_A": {
            "<designator in the image>": <number to fill in>,
            "<designator in the image>": <number to fill in>
        },
        "potencias_W": {
            "<designator in the image>": <number to fill in>,
            "<designator in the image>": <number to fill in>
        }
    },

    "analise_textual": {
        "regiao_operacao": {
            "<designator in the image>": "<that component's state>"
        },
        "terra_virtual": <true or false>
    }
}
- The image contains node labels (letters like A, B, C...) marking the points where the voltage must be calculated. Use EXACTLY those labels as keys in "tensoes_V" — do not invent descriptive names. For "correntes_A" and "potencias_W", use the component designator shown in the image (e.g. "R1", "D1").
- "regiao_operacao" is an object (dict), not a single sentence: include one entry for EACH diode and EACH operational amplifier in the circuit, using the designator shown in the image as the key (e.g. "D1", "U1") and that component's state as the value (e.g. "Forward Conduction", "Cutoff", "Positive Saturation", "Negative Saturation", "Linear"). Resistors, capacitors and inductors do not belong in this object.
- The block above is a TEMPLATE, not a solved example: each marker between < > indicates where to insert the value obtained from THIS image. Never copy a marker into your answer.
- All numeric values must be valid JSON numbers, without quotes and without units (the unit is already in the key name).
- In "componentes", the "valor" field is text and must reproduce the unit exactly as shown in the image; if the image shows only the number, do not add a prefix (do not write "10 kΩ" where the image shows "10").
- The number of items in each list and dictionary depends on the circuit in the image, not on the template above.
- If a value cannot be calculated, use "undetermined" or null for individual values.
- Do not write explanations outside the JSON.
"""


# Os seis prompts por idioma e nível (1 = Direto, 2 = Guiado, 3 = Estruturado).
# As chaves do JSON de saída pedidas são sempre em português; só o texto de
# instrução muda entre os dois idiomas.
PROMPTS = {
    "pt": {
        1: PROMPT_ANALISE_CIRCUITO_1,
        2: PROMPT_ANALISE_CIRCUITO_2,
        3: PROMPT_ANALISE_CIRCUITO_3,
    },
    "en": {
        1: PROMPT_ANALISE_CIRCUITO_1_EN,
        2: PROMPT_ANALISE_CIRCUITO_2_EN,
        3: PROMPT_ANALISE_CIRCUITO_3_EN,
    },
}

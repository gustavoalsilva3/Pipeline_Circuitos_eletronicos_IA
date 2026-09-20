# TCC_Gemini

Projeto de pesquisa para avaliação de modelos multimodais aplicados à análise e resolução de circuitos eletrônicos por imagem utilizando Large Language Models (LLMs).

## Objetivo

Este projeto tem como objetivo desenvolver uma estrutura automatizada capaz de:

* Ler imagens de circuitos eletrônicos (amplificador operacional, diodo, RLC);
* Enviar as imagens para modelos multimodais (hoje: Gemini 2.5 Flash; open source planejado);
* Solicitar a resolução do circuito via prompt, em 3 níveis de estruturação e 2 idiomas (PT/EN);
* Receber respostas estruturadas em JSON;
* Avaliar automaticamente a resposta contra um gabarito (validado via simulação LTspice);
* Consolidar métricas agregadas e gerar tabelas/gráficos comparativos.

O foco principal do trabalho é avaliar:

* identificação dos componentes do circuito (por categoria e por ID);
* reconhecimento da região de operação de diodos/amp-ops;
* precisão numérica (tensões, correntes, potências, ganho, Vout);
* efeito do nível de estruturação do prompt e do idioma no desempenho.

---

# Estrutura do Projeto

```text
Pipeline_TFC/
│
├── .venv/                    (não versionado)
├── .env                      (não versionado — GEMINI_API_KEY)
├── .gitignore
│
├── main.py                   # orquestra o pipeline completo
├── config.py                 # caminhos, modelo, idioma, tolerância, custos
├── prompts.py                # 3 níveis de prompt, em PT e EN
├── gemini_client.py          # comunicação com a API Gemini
├── response_splitter.py      # separa a resposta da IA em blocos (numérico/textual)
├── evaluator_numerico.py     # avalia tensões/correntes/potências/Vout/ganho
├── evaluator_textual.py      # avalia região de operação e terra virtual
├── evaluator_componentes.py  # avalia identificação e valor dos componentes
├── Analise_erros.py          # gera tabelas (CSV) e gráficos (PNG) a partir dos resultados
├── requirements.txt
├── Metricas.txt              # documentação detalhada de todas as métricas calculadas
│
├── imagens/                  # circuitos (Amp_op_1..8, Diodo_1..8, RLC_1..8)
├── gabaritos/                # um JSON de gabarito por circuito (Resposta_Amp1.json, ...)
│
└── resultados/                (não versionado — gerado a cada rodada)
    ├── respostas_json/                        # resposta bruta da IA, uma por execução
    ├── resultados_AAAAMMDD_HHMMSS.csv         # resumo: 1 linha por execução
    ├── resultados_detalhe_AAAAMMDD_HHMMSS.csv # detalhe: 1 linha por item comparado
    └── analise/
        ├── tabelas/*.csv
        └── graficos/*.png
```

---

# Descrição dos Arquivos

## `main.py`

Arquivo principal do projeto. Para cada combinação de imagem × idioma × nível de
prompt × modelo:

* envia a imagem + prompt ao modelo;
* salva a resposta bruta em JSON (`resultados/respostas_json/`);
* carrega o gabarito correspondente;
* avalia a resposta nas três dimensões (numérica, textual, componentes);
* grava uma linha no CSV resumo e as linhas correspondentes no CSV detalhe.

Cada execução de `main.py` gera um **novo** par de arquivos CSV, com o timestamp
da execução no nome — nunca sobrescreve/acumula com rodadas anteriores.

## `config.py`

Centraliza as configurações do projeto: caminhos de pastas, modelo usado,
`IDIOMA_PROMPT` ("pt", "en" ou "ambos" — roda os dois idiomas na mesma
execução), tolerância percentual usada nas comparações numéricas, e preços
usados para estimar o custo em USD/BRL.

## `prompts.py`

Três níveis de prompt (direto / guiado por passos / procedimento detalhado),
cada um em português e em inglês (sufixo `_EN`). As chaves do JSON de saída
são sempre em português nos dois idiomas — o código depende desses nomes
literalmente.

## `gemini_client.py`

Responsável pela comunicação com a API Gemini: autenticação via `.env`, envio
da imagem, contagem de tokens, parsing do JSON de resposta e retry automático
(3 tentativas) em caso de erro.

## `lmstudio_client.py`

Cliente para modelos abertos servidos pelo LM Studio (servidor local com API
compatível com a da OpenAI), com o mesmo contrato do `gemini_client.py`. Cada
modelo é ligado/desligado por um toggle em `config.py`
(`USAR_MODELO_GEMINI`, `USAR_MODELO_LMSTUDIO`). A URL do servidor fica no
`.env` (`LMSTUDIO_BASE_URL`, padrão `http://localhost:1234`);
`MODELOS_LMSTUDIO` é uma lista, cada identificador vira um modelo separado na
mesma rodada. Respostas cortadas por tamanho (loop de repetição) são
reexecutadas ao fim da rodada (`REEXECUTAR_TRUNCADAS` em `config.py`).

## `response_splitter.py`

Separa a resposta da IA (e o gabarito) nos blocos que os avaliadores esperam
(`numerico`, `textual`, mantendo `componentes` intacto).

## `evaluator_numerico.py`

Avalia tensões, correntes, potências, ganho teórico e Vout, com tolerância de
engenharia (padrão 5%, `config.TOLERANCIA_PADRAO_PERCENTUAL`) em vez de
igualdade exata — reconhece prefixos de engenharia (k, m, µ...), sinal e
unidade embutidos no valor.

## `evaluator_textual.py`

Avalia a região de operação (um dict por componente não-linear: diodos e
amp-ops) e o terra virtual (booleano único do circuito). A comparação de
região de operação é feita por **estado canônico**, não string exata —
reconhece sinônimos em português e inglês.

## `evaluator_componentes.py`

Avalia a identificação dos componentes (precision/recall/f1, por categoria
fechada: amplificadores operacionais, resistores, fontes, capacitores,
diodos, indutores) e, para os corretamente identificados, o valor informado
(mesma tolerância de engenharia do item numérico).

## `Analise_erros.py`

Lê o par de CSVs de uma rodada (por padrão, a mais recente) e gera tabelas
(CSV) e gráficos (PNG) prontos para uso no TCC: taxa de acerto por nível de
prompt, por família de circuito e por idioma (PT x EN), piores casos de Vout,
identificação de componentes, e distribuição de status (correto/incorreto/
faltante/alucinado) geral e por categoria. Não é executado automaticamente
por `main.py` — é um script separado, rodado sob demanda.

---

# Instalação

## 1. Criar ambiente virtual

```bash
python -m venv .venv
```

## 2. Ativar ambiente virtual

### Windows

```bash
.venv\Scripts\activate
```

### Linux/macOS

```bash
source .venv/bin/activate
```

## 3. Instalar dependências

```bash
pip install -r requirements.txt
```

---

# Configuração da API

Criar um arquivo `.env` na raiz do projeto:

```text
GEMINI_API_KEY=sua_chave_api
```

---

# Execução

## 1. Rodar o pipeline

```bash
python main.py
```

Antes de rodar, ajuste em `config.py`:

* `IDIOMA_PROMPT`: `"pt"`, `"en"` ou `"ambos"`;
* quais imagens processar (mova arquivos para fora de `imagens/` se quiser
  rodar só um subconjunto).

O programa irá, para cada imagem × idioma × nível de prompt × modelo:

1. enviar a imagem para o Gemini;
2. salvar a resposta bruta em JSON;
3. avaliar automaticamente contra o gabarito;
4. gravar o resultado no par de CSVs (resumo + detalhe) desta execução.

## 2. Gerar tabelas e gráficos

```bash
python Analise_erros.py
```

Lê o par de CSVs mais recente em `resultados/` e gera as saídas em
`resultados/analise/`. Ver `Metricas.txt` para a lista completa de métricas.

---

# Estrutura do Pipeline

```text
Imagem do circuito
        ↓
Prompt técnico (PT/EN, 3 níveis)
        ↓
Modelo multimodal (Gemini 2.5 Flash)
        ↓
Resposta JSON
        ↓
Avaliação automática (numérico + textual + componentes)
        ↓
CSV resumo + CSV detalhe
        ↓
Tabelas e gráficos (Analise_erros.py)
```

---

# Modelos previstos

## Proprietários

* Gemini 2.5 Flash

## Open Source

* A definir, assim que houver hardware disponível para rodar localmente

---

# Desenho experimental

* Modelo proprietário (PT): 24 circuitos × 3 níveis de prompt = 72 respostas;
* Modelo open source (PT): 24 × 3 = 72 respostas (planejado);
* Braço exploratório (PT x EN): mesmos 24 circuitos × 3 níveis × 2 idiomas,
  via `config.IDIOMA_PROMPT = "ambos"`.

---

# Objetivo Acadêmico

O projeto busca investigar a viabilidade do uso de modelos multimodais na
interpretação automatizada de circuitos eletrônicos, comparando desempenho
entre níveis de estruturação de prompt, idiomas, e (futuramente) entre
modelos proprietários e open source.

---

# Autor

Gustavo de Almeida Silva

Curso de Engenharia Elétrica

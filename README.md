# TCC_Gemini

Projeto de pesquisa para avaliação de modelos multimodais aplicados à análise e resolução de circuitos eletrônicos por imagem utilizando Large Language Models (LLMs).

## Objetivo

Este projeto tem como objetivo desenvolver uma estrutura automatizada capaz de:

* Ler imagens de circuitos eletrônicos;
* Enviar as imagens para modelos multimodais (Gemini, GPT, Qwen-VL, etc.);
* Solicitar a resolução do circuito via prompt;
* Receber respostas estruturadas em JSON;
* Extrair métricas quantitativas;
* Comparar desempenho entre modelos proprietários e open source.

O foco principal do trabalho é avaliar:

* identificação da topologia do circuito;
* reconhecimento da região de operação;
* coerência do raciocínio elétrico;
* precisão dos cálculos;
* robustez em diferentes qualidades de imagem.

---

# Estrutura do Projeto

```text
TCC_Gemini/
│
├── .venv/
├── .env
├── .gitignore
│
├── main.py
├── config.py
├── prompts.py
├── gemini_client.py
├── evaluator.py
├── requirements.txt
│
├── imagens/
│   ├── circuito_01.png
│   ├── circuito_02.png
│
├── resultados/
│   ├── respostas_json/
│   └── resultados.csv
│
└── gabaritos/
```

---

# Descrição dos Arquivos

## `main.py`

Arquivo principal do projeto.

Responsável por:

* carregar imagens;
* executar o pipeline;
* enviar requisições ao modelo;
* salvar respostas;
* gerar CSV final com métricas.

---

## `config.py`

Centraliza as configurações do projeto:

* caminhos de pastas;
* nome do modelo;
* diretórios de saída.

---

## `prompts.py`

Armazena os prompts utilizados nos experimentos.

Permite comparar:

* prompts simples;
* prompts estruturados;
* prompts passo a passo.

---

## `gemini_client.py`

Responsável pela comunicação com a API Gemini.

Funções principais:

* autenticação;
* envio da imagem;
* recebimento da resposta;
* tratamento de erros;
* retry automático.

---

## `evaluator.py`

Responsável pela avaliação das respostas geradas pelos modelos.

Exemplos de métricas:

* identificação correta de componentes;
* presença de conexões;
* funcionamento descrito;
* validade do JSON;
* erros de raciocínio.

---

# Instalação

## 1. Criar ambiente virtual

```bash
python -m venv .venv
```

---

## 2. Ativar ambiente virtual

### Windows

```bash
.venv\Scripts\activate
```

### Linux/macOS

```bash
source .venv/bin/activate
```

---

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

Executar:

```bash
python main.py
```

O programa irá:

1. carregar imagens da pasta `imagens/`;
2. enviar as imagens para o Gemini;
3. salvar respostas JSON;
4. gerar arquivo CSV com métricas.

---

# Estrutura do Pipeline

```text
Imagem do circuito
        ↓
Prompt técnico
        ↓
Modelo multimodal
        ↓
Resposta JSON
        ↓
Avaliação automática
        ↓
Métricas e resultados
```

---

# Modelos previstos

## Proprietários

* Gemini 2.5 Flash
* GPT-4o

## Open Source

* Qwen2.5-VL
* MiniCPM-V

---

# Possíveis Métricas

* Tempo de execução;
* Precisão dos cálculos;
* Reconhecimento da topologia;
* Região de operação correta;
* Robustez a ruído;
* Consistência do raciocínio;
* Taxa de falha da API;
* Qualidade do OCR.

---

# Objetivo Acadêmico

O projeto busca investigar a viabilidade do uso de modelos multimodais na interpretação automatizada de circuitos eletrônicos, comparando soluções proprietárias e open source em tarefas típicas da engenharia elétrica.

---

# Autor

Gustavo de Almeida Silva

Curso de Engenharia Elétrica

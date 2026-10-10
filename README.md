# Pipeline de avaliação de modelos multimodais em circuitos eletrônicos

Pipeline automatizada que submete imagens de esquemáticos de circuitos a
modelos multimodais (visão + linguagem), recebe a análise em JSON e a compara
com gabaritos validados por simulação no LTspice. Código de apoio ao Trabalho
Final de Curso em Engenharia Elétrica (UFSJ).

O que se mede: em que medida esses modelos conseguem **ler** um esquemático e
**resolvê-lo** eletricamente — e, principalmente, se essas duas capacidades
andam juntas.

## Desenho experimental

A rodada oficial cobre **576 execuções**: 24 circuitos × 3 níveis de prompt ×
2 idiomas (PT/EN) × 4 modelos, todos com temperatura 0.

Os quatro modelos foram escolhidos para separar dois fatores que costumam vir
confundidos em comparações desse tipo:

| Modelo | Origem | Precisão | Papel no desenho |
|---|---|---|---|
| Gemini 2.5 Flash | Google, API própria | — | referência proprietária |
| Qwen3-VL 32B | Alibaba, via OpenRouter | — | o par de porte do 8B |
| Qwen3-VL 8B | Parasail, via OpenRouter | bf16 | mesma família, ~4× menor |
| Gemma 3 27B | Novita, via OpenRouter | bf16 | segunda família, porte próximo ao 32B |

* **Efeito do porte** — Qwen3-VL 8B contra 32B: mesma família, mesma geração,
  mesma variante.
* **Efeito da família** — Qwen3-VL 32B contra Gemma 3 27B: portes próximos
  (32B contra 27,4B), ambos densos, ambos servidos em bf16.

Cada execução é avaliada em três dimensões independentes:

1. **numérica** — tensões, correntes, potências, ganho e Vout, com tolerância
   de engenharia de 5% (não igualdade exata);
2. **textual** — região de operação de cada componente não linear e presença
   de terra virtual;
3. **componentes** — identificação e valor de cada componente
   (*precision*, *recall*, $F_1$).

Cada item comparado recebe um status: `correto`, `incorreto`, `faltante`
(o modelo não se pronunciou), `alucinado` (o modelo citou algo que não existe
no gabarito) ou `nao_identificado`.

## Estrutura

```text
Pipeline_TFC/
│
├── .env                      (não versionado — chaves de API)
├── .venv/                    (não versionado)
│
├── main.py                   # orquestra a rodada completa
├── config.py                 # modelos ativos, idioma, tolerância, preços
├── prompts.py                # 3 níveis de prompt, em PT e EN
│
├── gemini_client.py          # API do Google
├── openrouter_client.py      # OpenRouter, com provedor fixado
├── lmstudio_client.py        # servidor local (LM Studio)
│
├── response_splitter.py      # separa a resposta nos blocos dos avaliadores
├── evaluator_numerico.py     # tensões, correntes, potências, ganho, Vout
├── evaluator_textual.py      # região de operação e terra virtual
├── evaluator_componentes.py  # identificação e valor dos componentes
│
├── consolidar_rodadas.py     # une sub-rodadas em um par único de CSVs
├── reavaliar_rodada.py       # reavalia respostas salvas, sem chamar modelos
│
├── Analise_erros.py          # tabelas (CSV) e gráficos (PNG) por modelo
├── gerar_tabelas_cap4.py     # tabelas LaTeX do capítulo de resultados
├── gerar_figuras_cap4.py     # figuras comparativas dos 4 modelos
├── intervalos_confianca.py   # intervalos por bootstrap
├── gerar_apendice_prompts.py # apêndice com o texto dos prompts
│
├── Metricas.txt              # documentação detalhada de cada métrica
├── requirements.txt
│
├── imagens/                  # Amp_op_1..8, Diodo_1..8, RLC_1..8
├── gabaritos/                # Resposta_Amp1.json, Resposta_Dio1.json, ...
├── resultados/               (não versionado)
└── TFC/                      # fontes LaTeX da monografia
```

## Instalação

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate no Linux/macOS
pip install -r requirements.txt
```

Criar um `.env` na raiz com as chaves dos provedores que forem usados:

```text
GEMINI_API_KEY=...      # API do Google
Router=...              # OpenRouter (aceita OPENROUTER_API_KEY como alternativa)
LMSTUDIO_BASE_URL=...   # opcional, padrão http://localhost:1234
```

## Execução

### 1. Rodar a pipeline

```bash
python main.py
```

Antes, em `config.py`: ligar os modelos desejados
(`USAR_MODELO_GEMINI`, `USAR_MODELO_OPENROUTER`, `USAR_MODELO_LMSTUDIO`),
definir `IDIOMA_PROMPT` (`"pt"`, `"en"` ou `"ambos"`) e, no caso do
OpenRouter, a lista `MODELOS_OPENROUTER`. Para rodar só um subconjunto de
circuitos, mova as demais imagens para fora de `imagens/`.

Cada execução de `main.py` gera um **novo** par de CSVs carimbado com a data e
a hora — nunca sobrescreve rodadas anteriores:

```text
resultados/resultados_AAAAMMDD_HHMMSS.csv          # 1 linha por execução
resultados/resultados_detalhe_AAAAMMDD_HHMMSS.csv  # 1 linha por item comparado
resultados/respostas_json/                         # resposta bruta de cada execução
```

### 2. Consolidar, se a rodada foi feita em partes

```bash
python consolidar_rodadas.py
```

Necessário porque cada chamada de `main.py` numera suas execuções a partir de
1. Unir os arquivos sem tratamento faria ids de rodadas diferentes colidirem,
e o CSV de detalhe — que referencia `id_execucao` — passaria a apontar para a
linha errada. O script renumera de forma contínua, preserva o id antigo em
`id_execucao_original` e acrescenta as colunas `provedor` e `quantizacao`.

### 2b. Reavaliar, se gabaritos ou avaliadores mudaram

```bash
python reavaliar_rodada.py 20260924_121301
```

Relê as respostas JSON já salvas e as avalia de novo com os gabaritos e
avaliadores atuais, sem chamar nenhum modelo. Tempo, tokens, custo e demais
metadados vêm da rodada indicada; o resultado é um novo par de CSVs. A rodada
oficial final (`20261010_144312`) foi gerada assim, a partir de
`20260927_160533`, depois que as correntes passaram a ser comparadas em
módulo; antes dela houve duas correções de gabarito: a dos `RLC_3`/`RLC_7`,
com a regra de taxa para categorias sem referência, e a dos circuitos com
amplificador operacional e diodo, que haviam sido simulados com componentes
não ideais (ver *Notas de implementação*).

### 3. Gerar as saídas

```bash
python Analise_erros.py           # tabelas e gráficos por modelo, em resultados/analise/
python gerar_tabelas_cap4.py      # tabelas LaTeX      -> TFC/tabelas_cap4/
python gerar_figuras_cap4.py      # figuras dos 4 modelos -> TFC/figuras/comparativo/
python intervalos_confianca.py    # bootstrap          -> TFC/tab_intervalos.tex
python gerar_apendice_prompts.py  # apêndice           -> TFC/apendice_prompts.tex
```

Os quatro últimos escrevem direto nas fontes LaTeX, de modo que **nenhum
número da monografia é digitado à mão**: refazer a rodada e reexecutar os
scripts atualiza tabelas, figuras e apêndice sem reescrever o texto.

## Notas de implementação

Alguns pontos que não são óbvios pelo código e que afetam a validade dos
resultados.

**Provedor fixado no OpenRouter.** Todas as chamadas enviam
`allow_fallbacks: false` com um único provedor em `order`. Sem isso o
OpenRouter roteia para o *backend* disponível no momento, que pode servir o
modelo em outra quantização — e a rodada deixa de ser reprodutível. O provedor
e a quantização que responderam ficam registrados por chamada em
`resultados/openrouter_chamadas.jsonl` e na chave `_openrouter` de cada
resposta.

**Limite de tokens de saída.** `MAX_TOKENS_PADRAO = 2000` em
`openrouter_client.py`. O OpenRouter **reserva** `max_tokens × preço de saída`
do saldo antes de executar a chamada, e não o consumo real. Um limite alto
derruba a rodada com HTTP 402 mesmo havendo saldo para o custo efetivo. O JSON
completo ocupa 400–700 tokens, então 2000 dá folga suficiente.

**O `.jsonl` só registra sucessos.** Para monitorar uma rodada em andamento,
conferir também o campo `falha_requisicao` nos arquivos de
`resultados/respostas_json/` — contar linhas do `.jsonl` esconde as falhas.

**Chaves do JSON sempre em português.** Nos prompts em inglês, apenas as
instruções e os marcadores são traduzidos; as chaves do JSON de saída
permanecem em português, porque os avaliadores dependem desses nomes
literalmente.

**Marcadores em vez de valores de exemplo.** O esquema de saída usa
`<numero a preencher>`, `<designador na imagem>` e afins. Uma versão anterior
trazia um exemplo preenchido, e os modelos copiavam aqueles valores em vez de
calcular — inclusive o proprietário. A troca alterou os resultados em
magnitude muito maior que qualquer outro fator testado, o que é discutido na
monografia.

**Reexecução de respostas truncadas.** Com `REEXECUTAR_TRUNCADAS = True`,
`main.py` refaz ao fim da rodada as execuções cortadas por tamanho ou por loop
de repetição. Como a temperatura 0 repetiria o mesmo loop, a reexecução usa
temperatura 0,3 e uma semente diferente por tentativa; as linhas afetadas ficam
marcadas na coluna `reexecucoes`, e a resposta original é preservada como
`<nome>_falha_original.json`.

**Tolerância, não igualdade.** As comparações numéricas usam 5%
(`config.TOLERANCIA_PADRAO_PERCENTUAL`), com reconhecimento de prefixos de
engenharia (k, m, µ), sinal e unidade embutidos no valor. A região de operação
é comparada por **estado canônico**, não por texto exato: "polarização direta
(ON)" e "condução direta" contam como a mesma resposta, em PT e em EN.

## Principais resultados

Os quatro modelos identificaram os componentes com $F_1 \geq 0{,}984$, dois
deles sem um único erro em 1.038 componentes — a percepção visual não é o
gargalo. Já na resolução elétrica, o Gemini acertou 61,5% dos itens comparados
e 51,7% das tensões nodais, contra 45,2% e 21,8% do melhor modelo aberto, com
intervalos de confiança que não se sobrepõem.

Dois achados contrariam o esperado: o porte não ordena o desempenho (o Gemma
3 27B ficou abaixo do Qwen3-VL 8B nas grandezas numéricas), e os modelos
abertos foram piores justamente nos circuitos **RLC**, os de menor
complexidade topológica — deixam de aplicar a condição de regime permanente
em CC, na qual o capacitor é circuito aberto e o indutor é curto-circuito.
Nem o nível de estruturação do prompt nem o idioma produziram efeitos
distinguíveis da variação aleatória.

Detalhamento das métricas em `Metricas.txt`; análise completa na monografia,
em `TFC/`.

## Autor

Gustavo de Almeida Silva — Engenharia Elétrica, UFSJ

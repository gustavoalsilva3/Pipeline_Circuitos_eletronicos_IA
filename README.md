# Pipeline de avaliação de modelos multimodais em circuitos eletrônicos

Pipeline automatizada que submete imagens de esquemáticos de circuitos a
modelos multimodais (visão + linguagem), recebe a análise em JSON e a compara
com gabaritos validados por simulação no LTspice. Código de apoio ao Trabalho
Final de Curso em Engenharia Elétrica (UFSJ).

O que se mede: em que medida esses modelos conseguem **ler** um esquemático e
**resolvê-lo** eletricamente — e, principalmente, se essas duas capacidades
andam juntas.

## Desenho experimental

A rodada oficial cobre **720 execuções**: 24 circuitos × 3 níveis de prompt ×
2 idiomas (PT/EN) × 5 configurações, todas com temperatura 0. São quatro
modelos, com o proprietário executado duas vezes, com e sem raciocínio
interno.

As configurações foram escolhidas para separar três fatores que costumam vir
confundidos em comparações desse tipo:

| Modelo | Origem | Precisão | Papel no desenho |
|---|---|---|---|
| Gemini 2.5 Flash | Google, API própria | — | referência proprietária |
| Gemini 2.5 Flash (sem raciocínio) | Google, API própria | — | o mesmo modelo com `thinking_budget=0` |
| Qwen3-VL 32B | Alibaba, via OpenRouter | — | o par de porte do 8B |
| Qwen3-VL 8B | Parasail, via OpenRouter | bf16 | mesma família, ~4× menor |
| Gemma 3 27B | Novita, via OpenRouter | bf16 | segunda família, porte próximo ao 32B |

* **Efeito do porte** — Qwen3-VL 8B contra 32B: mesma família, mesma geração,
  mesma variante.
* **Efeito da família** — Qwen3-VL 32B contra Gemma 3 27B: portes próximos
  (32B contra 27,4B), ambos densos, ambos servidos em bf16.
* **Efeito do raciocínio interno** — Gemini 2.5 Flash contra ele mesmo sem
  raciocínio: mesmo modelo, mesmo provedor, mesmos prompts e idiomas. É a
  única comparação do trabalho em que um único fator varia.

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
├── piloto_openrouter.py      # sonda um modelo em 6 circuitos, já pontuado
│
├── response_splitter.py      # separa a resposta nos blocos dos avaliadores
├── evaluator_numerico.py     # tensões, correntes, potências, ganho, Vout
├── evaluator_textual.py      # região de operação e terra virtual
├── evaluator_componentes.py  # identificação e valor dos componentes
│
├── consolidar_rodadas.py     # junta rodadas parciais num par unico de CSVs
├── reavaliar_rodada.py       # reavalia respostas salvas, sem chamar modelos
│
├── Analise_erros.py          # tabelas (CSV) e gráficos (PNG) por modelo
├── gerar_tabelas_cap4.py     # tabelas LaTeX do capítulo de resultados
├── intervalos_confianca.py   # intervalos por bootstrap
├── gerar_apendice_prompts.py # apêndice com o texto dos prompts
│
├── Metricas.txt              # documentação detalhada de cada métrica
├── requirements.txt
│
├── imagens/                  # Amp_op_1..8, Diodo_1..8, RLC_1..8
├── gabaritos/                # Resposta_Amp1.json, Resposta_Dio1.json, ...
├── resultados/               (não versionado — saída de rodadas novas)
├── rodada_oficial/           # dados da rodada final, versionados (ver Reprodução)
└── TFC/                      # fontes LaTeX da monografia
```

## Esquemáticos de referência

`simulacoes/` traz os 24 esquemáticos do LTspice (`.asc`) que produzem os
gabaritos, um por circuito, com o mesmo nome usado em `imagens/` e em
`gabaritos/`. Sem eles o gabarito não poderia ser refeito por outra pessoa.

Nos circuitos com amplificador operacional, o símbolo `UniversalOpAmp2` foi
substituído por uma fonte comportamental, declarada como diretiva no próprio
esquemático:

```
.subckt opideal 1 2 3 4 5
B1 5 0 V=limit(1e6*(V(1)-V(2)),V(4),V(3))
.ends
```

A troca é necessária porque o `UniversalOpAmp2` tem `Ron=10` fixo no modelo
(não é parâmetro) e `ilimit=25m` por padrão. Com os resistores de poucos ohms
usados no *dataset*, a saída nunca alcança o trilho de alimentação, e a
simulação não reproduz a análise ideal declarada no *prompt*. A fonte
comportamental entrega o que o trabalho assume: ganho de 1e6, impedância de
saída nula e saída limitada à alimentação.

Os 24 esquemáticos foram conferidos contra os gabaritos: todas as tensões
nodais e todas as correntes coincidem, com convergência limpa.

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
```

## Reprodução (sem chaves de API)

A pasta `rodada_oficial/` é versionada e guarda os dados que sustentam os
números do texto: o par de CSVs da rodada final e a resposta bruta de cada uma
das 720 execuções. São 741 arquivos, porque 21 requisições da rodada de
setembro falharam na primeira tentativa e a resposta original foi preservada
com o sufixo `_falha_original`. Com essa pasta, a análise inteira se refaz sem
chamar nenhuma API e sem precisar de chave nenhuma:

```bash
python reavaliar_rodada.py        # reavalia as 720 respostas -> novo par de CSVs
python gerar_tabelas_cap4.py      # tabelas LaTeX      -> TFC/tabelas_cap4/
python intervalos_confianca.py    # bootstrap          -> TFC/tab_pareado.tex e a figura dos intervalos
python Analise_erros.py oficial   # tabelas e gráficos -> resultados/analise/ e TFC/figuras/
```

O primeiro comando é a verificação mais direta: ele reavalia cada resposta
salva com os gabaritos e avaliadores do repositório e grava em `resultados/`
um par de CSVs **idêntico byte a byte** a `rodada_oficial/resultados.csv` e
`rodada_oficial/resultados_detalhe.csv`. Os outros três leem direto da pasta
versionada: os dois do meio reescrevem as 16 tabelas LaTeX do texto, também
idênticas às versionadas.

As cinco figuras do capítulo de resultados vão direto para
`TFC/figuras/comparativo/`: quatro gravadas pelo `Analise_erros.py` e a dos
intervalos de confiança pelo `intervalos_confianca.py`. Não há passo manual:
rodar os quatro comandos acima e compilar o `TFC/Main_TFC.tex` refaz todas as
tabelas e todas as figuras do texto.

`rodada_oficial/` é somente leitura: rodadas novas vão sempre para
`resultados/`, que o git ignora.

## Execução

Esta seção é para gerar uma rodada **nova**, chamando os modelos — o que exige
as chaves de API e não é necessário para conferir os resultados do texto.

### 1. Sondar um modelo, antes de pagar a rodada completa

```bash
python piloto_openrouter.py
python piloto_openrouter.py --modelo qwen/qwen3-vl-8b-instruct --provider Parasail
```

Roda 6 dos circuitos mais difíceis no prompt de nível 1 e imprime as mesmas
métricas do capítulo de resultados, o que torna os números diretamente
comparáveis aos das tabelas. Útil para decidir se vale avaliar um modelo novo
na rodada inteira, ou para colocar dois lado a lado sem gastar a rodada
completa. O `--provider` fixa o provedor pelo mesmo motivo da rodada oficial:
sem isso o OpenRouter pode rotear para backends com precisões diferentes, e as
execuções deixam de ser comparáveis entre si.

### 2. Rodar a pipeline

```bash
python main.py
```

Antes, em `config.py`: ligar os modelos desejados
(`USAR_MODELO_GEMINI` e `USAR_MODELO_OPENROUTER`),
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

#### Rodada sem raciocínio interno (quinto modelo)

O Gemini 2.5 Flash raciocina internamente por padrão, com orçamento dinâmico, e
foi assim que a rodada oficial foi feita. Para rodar o **mesmo** modelo com o
raciocínio desligado (`thinking_budget=0`):

```bash
python main.py --sem-raciocinio --somente-gemini
```

As linhas vão para o CSV com o rótulo `gemini-2.5-flash (sem raciocinio)`, e não
`gemini-2.5-flash`, para que as duas rodadas possam ser consolidadas e
comparadas como modelos distintos. A opção `--somente-gemini` deixa a rodada em
144 execuções, sem gastar chamadas nos modelos do OpenRouter.

Sem essas opções nenhum campo de raciocínio é enviado à API, e o comportamento é
exatamente o da rodada oficial.

Sem os tokens de raciocínio a rodada fica cerca de 7 vezes mais barata e 3 vezes
mais rápida (projeção de US$ 0,27 e 14 minutos, contra US$ 1,88 da rodada
oficial do Gemini), porque os tokens pensados são cobrados como saída.

Para que o modelo novo apareça nas tabelas e figuras geradas, acrescente o
rótulo às listas de ordem: `ORDEM` em `gerar_tabelas_cap4.py`, `ORDEM_MODELOS`
em `Analise_erros.py` e `modelos` em `intervalos_confianca.py`. Enquanto essas
listas não mudarem, as saídas continuam idênticas às do texto atual.

### 3. Consolidar, se a rodada foi feita em partes

```bash
python consolidar_rodadas.py                              # menu de escolha
python consolidar_rodadas.py 20260924_082751 20260924_091447
```

Sem argumentos, lista as rodadas que encontrou em `resultados/` — com o número
de execuções e os modelos de cada uma — e pergunta quais juntar. É o caminho
normal para quem avalia aos poucos: um modelo hoje, outro amanhã, ou alguns
circuitos por vez.

Necessário porque cada chamada de `main.py` numera suas execuções a partir de
1. Unir os arquivos sem tratamento faria ids de rodadas diferentes colidirem,
e o CSV de detalhe — que referencia `id_execucao` — passaria a apontar para a
linha errada. O script renumera de forma contínua na ordem escolhida, preserva
o id antigo em `id_execucao_original` e acrescenta as colunas `provedor` e
`quantizacao`. Modelos que não estiverem no dicionário `ORIGEM` são gravados
como `nao informado`, com aviso — acrescente uma entrada ali para registrar a
procedência dos seus.

### 4. Reavaliar, se gabaritos ou avaliadores mudaram

```bash
python reavaliar_rodada.py                   # a rodada oficial versionada (padrão)
python reavaliar_rodada.py 20261010_144312   # uma rodada de resultados/
```

Relê as respostas JSON já salvas e as avalia de novo com os gabaritos e
avaliadores atuais, sem chamar nenhum modelo. Tempo, tokens, custo e demais
metadados vêm da rodada indicada; o resultado é um novo par de CSVs em
`resultados/`. A rodada
oficial final (`20261010_144312`) foi gerada assim, a partir de
`20260927_160533`, depois que as correntes passaram a ser comparadas em
módulo; antes dela houve duas correções de gabarito: a dos `RLC_3`/`RLC_7`,
com a regra de taxa para categorias sem referência, e a dos circuitos com
amplificador operacional e diodo, que haviam sido simulados com componentes
não ideais (ver *Notas de implementação*).

A rodada original foi executada em partes, em 24/09, e reunida com o
`consolidar_rodadas.py` (passo 3) — daí vêm as colunas `id_execucao_original`,
`rodada_origem`, `provedor` e `quantizacao` dos CSVs versionados.

### 5. Gerar as saídas

```bash
python Analise_erros.py           # a rodada mais recente de resultados/
python Analise_erros.py oficial   # ou a rodada oficial versionada
python gerar_tabelas_cap4.py      # tabelas LaTeX      -> TFC/tabelas_cap4/
python intervalos_confianca.py    # bootstrap          -> TFC/tab_intervalos.tex
python gerar_apendice_prompts.py  # apêndice           -> TFC/apendice_prompts.tex
```

Todos escrevem direto nas fontes LaTeX ou em `resultados/analise/`, de modo
que **nenhum número da monografia é digitado à mão**: refazer a rodada e
reexecutar os scripts atualiza tabelas, figuras e apêndice sem tocar no texto.

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

As cinco configurações identificaram os componentes com $F_1 \geq 0{,}963$, três
delas sem um único falso positivo em 1.038 componentes: a leitura do inventário
não é o gargalo. Na resolução elétrica a distância é grande. O Gemini 2.5 Flash
acertou 76,0% dos itens comparados e 82,8% das tensões nodais, contra 52,3% e
39,0% do melhor modelo aberto, com intervalos de confiança que não se sobrepõem.

O maior efeito medido não é o da origem do modelo, e sim o do **raciocínio
interno**. Executado com `thinking_budget=0`, o mesmo Gemini cai para 51,8% dos
itens e 38,6% das tensões, ou seja, para a faixa dos modelos abertos. A
comparação pareada entre as duas configurações dá +44,2 pontos percentuais nas
tensões, a maior diferença do trabalho, e é a única em que modelo, provedor,
prompt e idioma são idênticos. A identificação dos componentes, porém, não muda
entre elas: o que o raciocínio sustenta é o cálculo, não a leitura do
esquemático. E ele domina o processamento, com 5.055 tokens de saída por
resposta contra 537, e 23,0 s contra 3,8 s.

Dois achados contrariam o esperado. O porte ordena o desempenho dentro de uma
mesma família (Qwen3-VL 32B acima do 8B nas tensões e nas potências), mas não
entre famílias: o Gemma 3 27B não se distingue do Qwen3-VL 8B em nenhuma das
três grandezas numéricas, apesar de ter mais que o triplo dos parâmetros. E nem
o nível de estruturação do prompt nem o idioma produziram efeito consistente:
das 60 comparações pareadas, 7 excluem zero, todas entre 4,7 e 7,7 pontos
percentuais e com sinais que mudam conforme o modelo.

Detalhamento das métricas em `Metricas.txt`; análise completa na monografia,
em `TFC/`.

## Autor

Gustavo de Almeida Silva — Engenharia Elétrica, UFSJ

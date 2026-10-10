"""Cliente para modelos multimodais servidos pelo OpenRouter.

Implementa o mesmo contrato de GeminiCircuitAnalyzer —
analisar_imagem(image_path, prompt) -> (resposta_dict, input_tokens,
output_tokens), SEMPRE 3 valores, mesmo em erro —, de modo que main.py trate
os modelos da mesma forma. A API segue o padrão da OpenAI
(/v1/chat/completions). Três pontos merecem nota:

1. AUTENTICAÇÃO: exige 'Authorization: Bearer <chave>'. A chave vem do .env
   (OPENROUTER_API_KEY, com 'Router' aceito como nome alternativo) e nunca é
   impressa.

2. RASTREABILIDADE DO PROVEDOR: o OpenRouter roteia a mesma requisição para
   provedores diferentes, que podem servir o modelo em quantizações
   diferentes (bf16, fp8, ...). Como a quantização afeta a qualidade — que é
   justamente o que o trabalho mede —, cada chamada registra qual provedor
   respondeu e em que quantização ele declara servir, tanto no dicionário de
   resposta (chave '_openrouter', ignorada pelos avaliadores, que leem apenas
   as chaves do esquema) quanto em um log .jsonl separado. Sem isso a rodada
   não é reprodutível.

3. FIXAÇÃO DO PROVEDOR: passando provider="Alibaba", a requisição desliga o
   fallback e obriga o OpenRouter a usar aquele provedor, ou falhar. É o modo
   recomendado para a rodada final.

A quantização não vem na resposta da chamada; é consultada uma única vez no
endpoint /models/<id>/endpoints e guardada em cache por provedor.
"""

import base64
import json
import mimetypes
import os
import re
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://openrouter.ai/api/v1"
LOG_CHAMADAS = Path(__file__).parent / "resultados" / "openrouter_chamadas.jsonl"

# Limite de tokens de SAÍDA enviado por padrão. É necessário porque o
# OpenRouter RESERVA o crédito do pior caso antes de executar: sem
# max_tokens, alguns provedores reservam o contexto inteiro do modelo e
# recusam a chamada com HTTP 402 ainda que a resposta real seja curta.
#
# O valor precisa ficar abaixo do teto de reserva da conta, que varia com o
# saldo disponível. Na rodada oficial, a resposta mais longa dos modelos
# abertos teve 742 tokens, então 2000 dá folga de mais de 2,5x sem se
# aproximar do teto.
#
# Um valor alto demais derruba a rodada inteira: com 8000, uma rodada chegou
# a falhar em 194 de 288 execuções por HTTP 402 ("can only afford N"), ainda
# que as respostas reais coubessem de sobra. Havendo saldo em conta, o teto
# deixa de ser restritivo.
MAX_TOKENS_PADRAO = 2000


def _chave() -> str:
    chave = os.getenv("OPENROUTER_API_KEY") or os.getenv("Router")
    if not chave:
        raise RuntimeError(
            "Chave do OpenRouter nao encontrada - defina OPENROUTER_API_KEY "
            "no .env (ver .env.example)."
        )
    return chave.strip()


def _extrair_json(texto: str):
    """Extrai o primeiro objeto JSON de um texto de modelo. Retorna o dict,
    ou None se não houver JSON válido.

    O prompt pede "APENAS um JSON válido", mas a API não força o formato:
    tolera cercas de markdown, texto antes/depois e blocos <think>...</think>
    de modelos de raciocínio."""
    if not texto:
        return None

    texto = re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL)
    texto = re.sub(r"^```(?:json)?\s*|\s*```$", "", texto.strip())

    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        pass

    inicio, fim = texto.find("{"), texto.rfind("}")
    if inicio != -1 and fim > inicio:
        try:
            return json.loads(texto[inicio:fim + 1])
        except json.JSONDecodeError:
            return None
    return None


class OpenRouterCircuitAnalyzer:
    """Analisa uma imagem de circuito com um modelo servido pelo OpenRouter.

    model_name: identificador do catálogo (ex: "qwen/qwen3-vl-32b-instruct").
    provider:   nome do provedor a fixar (ex: "Alibaba"); None deixa o
                OpenRouter rotear livremente (não recomendado na rodada final).
    max_tokens: limite de tokens de saída (ver MAX_TOKENS_PADRAO); passe 0
                para não enviar limite algum.
    """

    def __init__(self, model_name: str, provider: str | None = None,
                 max_tokens: int | None = None, timeout: int | None = None):
        self.model_name = model_name
        self.provider = provider
        self.max_tokens = MAX_TOKENS_PADRAO if max_tokens is None else (max_tokens or None)
        self.timeout = timeout or 600
        self.ultima_chamada: dict = {}
        self._quant_por_provedor: dict | None = None

    # ---------------------------------------------------------------- meta
    def _quantizacao(self, provedor):
        """Quantização declarada pelo provedor que respondeu, consultada uma
        única vez no catálogo e mantida em cache."""
        if provedor is None:
            return None
        if self._quant_por_provedor is None:
            self._quant_por_provedor = {}
            try:
                r = requests.get(
                    f"{BASE_URL}/models/{self.model_name}/endpoints",
                    headers={"Authorization": f"Bearer {_chave()}"},
                    timeout=30,
                )
                if r.ok:
                    for e in (r.json().get("data") or {}).get("endpoints", []):
                        self._quant_por_provedor[e.get("provider_name")] = e.get("quantization")
            except Exception as e:
                print(f"  [aviso] nao foi possivel consultar a quantizacao: {e}")
        return self._quant_por_provedor.get(provedor)

    def _registrar(self, image_path, dados, in_tok, out_tok, erro_json):
        provedor = dados.get("provider")
        meta = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "imagem": Path(image_path).name,
            "modelo": self.model_name,
            "provedor": provedor,
            "quantizacao": self._quantizacao(provedor),
            "id_geracao": dados.get("id"),
            "finish_reason": (dados.get("choices") or [{}])[0].get("finish_reason"),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "erro_json": erro_json,
        }
        self.ultima_chamada = meta
        try:
            LOG_CHAMADAS.parent.mkdir(parents=True, exist_ok=True)
            with open(LOG_CHAMADAS, "a", encoding="utf-8") as f:
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"  [aviso] nao foi possivel gravar o log de chamadas: {e}")
        return meta

    # ------------------------------------------------------------ chamada
    def analisar_imagem(self, image_path, prompt: str,
                        temperatura: float | None = None, seed: int | None = None):
        """Envia imagem + prompt ao OpenRouter. Retorna SEMPRE
        (resposta_dict, input_tokens, output_tokens)."""

        mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
        with open(image_path, "rb") as f:
            imagem_b64 = base64.b64encode(f.read()).decode("utf-8")

        corpo = {
            "model": self.model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime};base64,{imagem_b64}"}},
                ],
            }],
            "temperature": 0.0 if temperatura is None else temperatura,
            "stream": False,
        }
        if seed is not None:
            corpo["seed"] = seed
        if self.max_tokens is not None:
            corpo["max_tokens"] = self.max_tokens
        if self.provider:
            # allow_fallbacks=False: ou responde esse provedor, ou falha - para
            # que a quantizacao registrada seja de fato a usada.
            corpo["provider"] = {"order": [self.provider], "allow_fallbacks": False}

        max_tentativas = 3
        for tentativa in range(max_tentativas):
            try:
                response = requests.post(
                    f"{BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {_chave()}",
                        "Content-Type": "application/json",
                        "X-Title": "TFC - Analise de circuitos com IA multimodal",
                    },
                    json=corpo,
                    timeout=self.timeout,
                )
                if not response.ok:
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
                dados = response.json()

                if "error" in dados and not dados.get("choices"):
                    raise RuntimeError(f"Erro do OpenRouter: {str(dados['error'])[:400]}")

                uso = dados.get("usage") or {}
                input_tokens = uso.get("prompt_tokens", 0) or 0
                output_tokens = uso.get("completion_tokens", 0) or 0

                escolha = (dados.get("choices") or [{}])[0]
                texto = (escolha.get("message") or {}).get("content") or ""

                resposta = _extrair_json(texto)
                erro_json = resposta is None
                if erro_json:
                    resposta = {"erro_json": True, "resposta_bruta": texto}
                    if escolha.get("finish_reason") == "length":
                        resposta["truncado_por_tamanho"] = True

                meta = self._registrar(image_path, dados, input_tokens, output_tokens, erro_json)
                resposta["_openrouter"] = meta

                return resposta, input_tokens, output_tokens

            except Exception as e:
                print(f"\nErro na tentativa {tentativa + 1}: {e}")
                if tentativa < max_tentativas - 1:
                    print("Tentando novamente em 5 segundos...\n")
                    time.sleep(5)

        return {"erro_json": True, "resposta_bruta": "", "falha_requisicao": True}, 0, 0

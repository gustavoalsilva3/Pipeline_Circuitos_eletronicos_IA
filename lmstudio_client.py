"""Cliente para modelos multimodais servidos pelo LM Studio (servidor local
com API compatível com a da OpenAI, /v1/chat/completions).

Implementa o mesmo contrato de GeminiCircuitAnalyzer —
analisar_imagem(image_path, prompt) -> (resposta_dict, input_tokens,
output_tokens), SEMPRE 3 valores, mesmo em erro.

O LM Studio serve o modelo que estiver carregado; o campo "model" da
requisição é o identificador que aparece na aba do servidor local do app
(ex: "qwen/qwen3-vl-8b"). A URL padrão é http://localhost:1234 — para outra
máquina (ex: via túnel), defina LMSTUDIO_BASE_URL no .env.

Aqui não se força o JSON via parâmetro da API
(o suporte a response_format varia entre versões/modelos do LM Studio): o
prompt já pede "APENAS um JSON válido", e _extrair_json() tolera cercas de
markdown, texto antes/depois e blocos <think>...</think> de modelos de
raciocínio.
"""

import base64
import json
import mimetypes
import os
import re
import time

import requests
from dotenv import load_dotenv

from config import MAX_TOKENS_SAIDA_LOCAL

load_dotenv()

URL_PADRAO = "http://localhost:1234"


def _extrair_json(texto: str):
    """Extrai o primeiro objeto JSON de um texto de modelo. Retorna o dict,
    ou None se não houver JSON válido."""
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


class LMStudioCircuitAnalyzer:

    def __init__(self, model_name: str, base_url: str | None = None, max_tokens: int | None = None, timeout: int | None = None):
        self.model_name = model_name
        self.max_tokens = max_tokens or MAX_TOKENS_SAIDA_LOCAL
        self.timeout = timeout or 600
        self.base_url = (base_url or os.getenv("LMSTUDIO_BASE_URL") or URL_PADRAO).rstrip("/")

    def analisar_imagem(self, image_path, prompt: str, temperatura: float | None = None, seed: int | None = None):
        """Envia a imagem + prompt ao LM Studio e retorna SEMPRE uma tupla de
        3 valores: (resposta_dict, input_tokens, output_tokens) — mesmo
        contrato de gemini_client.GeminiCircuitAnalyzer.analisar_imagem."""

        mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
        with open(image_path, "rb") as f:
            imagem_b64 = base64.b64encode(f.read()).decode("utf-8")

        max_tentativas = 3

        for tentativa in range(max_tentativas):

            try:
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": self.model_name,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:{mime};base64,{imagem_b64}"},
                                    },
                                ],
                            }
                        ],
                        "temperature": 0.0 if temperatura is None else temperatura,
                        **({"seed": seed} if seed is not None else {}),
                        "max_tokens": self.max_tokens,
                        "stream": False,
                    },
                    # Modelo local pode ser lento (carregamento na 1ª chamada,
                    # imagem grande, raciocínio) — timeout generoso.
                    timeout=self.timeout,
                )
                if not response.ok:
                    # Inclui o corpo da resposta: o servidor explica a causa
                    # (ex: falta de memória ao carregar o modelo), que o
                    # raise_for_status() sozinho esconde.
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
                dados = response.json()

                uso = dados.get("usage") or {}
                input_tokens = uso.get("prompt_tokens", 0) or 0
                output_tokens = uso.get("completion_tokens", 0) or 0

                escolha = dados["choices"][0]
                texto = escolha["message"].get("content") or ""

                resposta = _extrair_json(texto)
                if resposta is None:
                    resposta = {
                        "erro_json": True,
                        "resposta_bruta": texto
                    }
                    # finish_reason == "length": a geração bateu no limite de
                    # tokens (ex: loop de repetição) e o JSON ficou incompleto.
                    if escolha.get("finish_reason") == "length":
                        resposta["truncado_por_tamanho"] = True

                return resposta, input_tokens, output_tokens

            except Exception as e:

                print(f"\nErro na tentativa {tentativa + 1}: {e}")

                if tentativa < max_tentativas - 1:

                    print("Tentando novamente em 5 segundos...\n")
                    time.sleep(5)

                else:

                    return {
                        "erro_api": True,
                        "mensagem_erro": str(e)
                    }, 0, 0

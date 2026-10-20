import os
import json
import time

from PIL import Image
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()


class GeminiCircuitAnalyzer:

    def __init__(self, model_name: str, thinking_budget: int | None = None):

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "Erro: GEMINI_API_KEY não encontrada no arquivo .env"
            )

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        # None = padrao do modelo (orcamento dinamico), que foi o da
        # rodada oficial. 0 desliga o raciocinio interno.
        self.thinking_budget = thinking_budget

    def analisar_imagem(self, image_path, prompt: str, temperatura: float | None = None, seed: int | None = None):
        """Envia a imagem + prompt ao Gemini e retorna SEMPRE uma tupla de
        3 valores: (resposta_dict, input_tokens, output_tokens) — mesmo em
        caso de erro (ver bloco except abaixo).

        Contrato importante: NUNCA retornar só o dict da resposta. O
        main.py desempacota o retorno como
        `resposta, input_tokens, output_tokens = analisar_imagem(...)`;
        se essa função devolvesse o dict diretamente, o desempacotamento
        iteraria as CHAVES do dict (não os valores) e quebraria com
        "too many/not enough values to unpack" sempre que o JSON de
        resposta não tivesse exatamente 3 chaves de topo.
        """

        image = Image.open(image_path)

        max_tentativas = 3

        for tentativa in range(max_tentativas):

            try:

                # O thinking_config so entra quando ha orcamento definido.
                # Nao passar o campo deixa o modelo no padrao dinamico, que
                # nao e equivalente a passar thinking_config com budget None
                # — por isso a configuracao e montada condicionalmente, para
                # que a rodada oficial continue reproduzivel byte a byte.
                cfg = {
                    "temperature": 0.0 if temperatura is None else temperatura,
                    "seed": seed,
                    "response_mime_type": "application/json",
                }

                if self.thinking_budget is not None:
                    cfg["thinking_config"] = types.ThinkingConfig(
                        thinking_budget=self.thinking_budget)

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[prompt, image],
                    config=types.GenerateContentConfig(**cfg)
                )

                # usage_metadata traz a contagem real de tokens da chamada.
                # getattr com default 0 evita quebrar caso a API não devolva
                # esse campo em algum cenário (ex: resposta bloqueada por
                # filtro de segurança, onde usage_metadata pode vir None).
                uso = getattr(response, "usage_metadata", None)
                input_tokens = getattr(uso, "prompt_token_count", 0) or 0
                # O Gemini 2.5 Flash "pensa" antes de responder e cobra esses
                # tokens de raciocínio (thoughts_token_count) como saída —
                # sem somá-los, o custo registrado ficava ~5x abaixo do real.
                output_tokens = (
                    (getattr(uso, "candidates_token_count", 0) or 0)
                    + (getattr(uso, "thoughts_token_count", 0) or 0)
                )

                texto = response.text

                try:
                    resposta = json.loads(texto)
                except json.JSONDecodeError:
                    resposta = {
                        "erro_json": True,
                        "resposta_bruta": texto
                    }

                return resposta, input_tokens, output_tokens

            except Exception as e:

                print(f"\nErro na tentativa {tentativa + 1}: {e}")

                if tentativa < max_tentativas - 1:

                    print("Tentando novamente em 5 segundos...\n")
                    time.sleep(5)

                else:

                    # Falha definitiva: sem chamada bem-sucedida não há
                    # tokens para contar, mas ainda retornamos 3 valores
                    # para não quebrar o desempacotamento no main.py.
                    return {
                        "erro_api": True,
                        "mensagem_erro": str(e)
                    }, 0, 0
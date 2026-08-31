import os
import json
import time

from PIL import Image
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()


class GeminiCircuitAnalyzer:

    def __init__(self, model_name: str):

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "Erro: GEMINI_API_KEY não encontrada no arquivo .env"
            )

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def analisar_imagem(self, image_path, prompt: str) -> dict:

        image = Image.open(image_path)

        max_tentativas = 3

        for tentativa in range(max_tentativas):

            try:

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[prompt, image],
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        response_mime_type="application/json"
                    )
                )

                texto = response.text

                try:
                    return json.loads(texto)

                except json.JSONDecodeError:

                    return {
                        "erro_json": True,
                        "resposta_bruta": texto
                    }

            except Exception as e:

                print(f"\nErro na tentativa {tentativa + 1}: {e}")

                if tentativa < max_tentativas - 1:

                    print("Tentando novamente em 5 segundos...\n")
                    time.sleep(5)

                else:

                    return {
                        "erro_api": True,
                        "mensagem_erro": str(e)
                    }
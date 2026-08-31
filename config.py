from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

IMAGENS_DIR = BASE_DIR / "imagens"
GABARITOS_DIR = BASE_DIR / "gabaritos"
RESULTADOS_DIR = BASE_DIR / "resultados"
JSON_DIR = RESULTADOS_DIR / "respostas_json"

CSV_RESULTADOS = RESULTADOS_DIR / "resultados.csv"

MODELO_GEMINI = "gemini-2.5-flash"
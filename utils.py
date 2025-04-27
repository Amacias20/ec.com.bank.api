import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

def get_openai_api_key():
    """Obtiene la clave API de OpenAI desde las variables de entorno."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY no está configurada en el archivo .env")
    return api_key

def format_response(response):
    """Formatea la respuesta para el usuario final."""
    return response.strip()

from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from typing import Type
import requests

class ExchangeRateInput(BaseModel):
    """Entradas para la herramienta de tipo de cambio."""
    query: str = Field(description="Consulta sobre tipos de cambio de divisas")

class ExchangeRateTool(BaseTool):
    name: str = "exchange_rate_tool"
    description: str = "Útil para cuando necesitas información sobre tipos de cambio entre divisas."
    args_schema: Type[BaseModel] = ExchangeRateInput

    def _run(self, query: str) -> str:
        try:
            # Usamos la API de exchangerate-api.com (gratuita con límites)
            response = requests.get("https://api.exchangerate-api.com/v4/latest/USD")
            data = response.json()
            
            rates = data.get("rates", {})
            
            if "dólar" in query.lower() and "euro" in query.lower():
                eur_rate = rates.get("EUR", 0)
                return f"El tipo de cambio actual de USD a EUR es: {eur_rate}"
            elif "euro" in query.lower():
                eur_rate = rates.get("EUR", 0)
                return f"El tipo de cambio actual de USD a EUR es: {eur_rate}"
            elif "dólar" in query.lower():
                return "El tipo de cambio se basa en USD. ¿Qué divisa específica te interesa?"
            else:
                # Intentar extraer alguna divisa mencionada en la consulta
                currencies = {
                    "peso mexicano": "MXN",
                    "peso argentino": "ARS",
                    "yen": "JPY",
                    "libra": "GBP",
                    "franco suizo": "CHF"
                }
                
                for currency_name, code in currencies.items():
                    if currency_name in query.lower():
                        rate = rates.get(code, 0)
                        return f"El tipo de cambio actual de USD a {code} es: {rate}"
                
                return "No pude identificar las divisas específicas. Puedo proporcionar información sobre USD, EUR, MXN, ARS, JPY, GBP, CHF y otras divisas principales."
        
        except Exception as e:
            return f"Error al consultar el tipo de cambio: {str(e)}"

def query_exchange_rate(question: str) -> str:
    """Consulta el tipo de cambio basado en la pregunta del usuario."""
    tool = ExchangeRateTool()
    return tool.run(question)

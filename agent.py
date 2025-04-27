from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
import unicodedata
import openai
import os
import re

def normalize_text(text):
    """Normaliza el texto eliminando caracteres especiales y acentos."""
    if not isinstance(text, str):
        return text
    # Normalizar a NFKD para descomponer caracteres acentuados
    text_normalized = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    # Eliminar caracteres especiales excepto puntuación básica
    text_normalized = re.sub(r'[^\w\s.,?!]', '', text_normalized)
    return text_normalized

# Configurar una clave API de OpenAI limpia (sin caracteres especiales en el entorno)
if 'OPENAI_API_KEY' in os.environ:
    openai.api_key = normalize_text(os.environ['OPENAI_API_KEY'])

def create_agent():
    """Crea un agente simple para probar la conexion con OpenAI."""

    import os
    api_key = os.environ.get("OPENAI_API_KEY")
    print(f"API KEY: {api_key} - longitud: {len(api_key)}")
    for i, c in enumerate(api_key):
        print(f"{i}: {c} ({ord(c)})")

    # Mensajes del sistema sin caracteres especiales
    system_message = "Eres un asistente virtual que ayuda a responder preguntas simples."
    system_message = normalize_text(system_message)
    
    # Usar MessagesPlaceholder para agent_scratchpad
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # Configurar el modelo sin el parámetro encoding incorrecto
    chat_model = ChatOpenAI(
        model_name="gpt-3.5-turbo", 
        temperature=0
    )
    
    # Crear el agente y el ejecutor
    agent = create_openai_functions_agent(chat_model, tools=[], prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True)
    
    return agent_executor

def query_agent(question):
    """Consulta a OpenAI con una pregunta simple en español."""
    chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    # Mensaje de sistema solo una vez
    messages = [
        SystemMessage(content="Eres un asistente virtual bancario. Responde SIEMPRE en español, de manera clara y profesional."),
        HumanMessage(content=question)
    ]
    response = chat_model.invoke(messages)
    return response.content 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from utils import get_openai_api_key
import unicodedata
import os

def normalize_text(text):
    """Elimina acentos y caracteres especiales."""
    return ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')

# Inicializar el almacén vectorial
def initialize_vector_store():
    """Inicializa y retorna el almacén vectorial con datos financieros."""
    # Verificar si hay documentos cargados
    if not os.path.exists("./chroma_db"):
        raise ValueError("No hay documentos cargados en la base vectorial.")
    
    # Obtener la API key
    api_key = get_openai_api_key()
    
    # Crear embeddings
    embeddings = OpenAIEmbeddings(api_key=api_key)
    
    # Crear y retornar el almacén vectorial existente
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    return db

def query_rag(question):
    """Consulta la base de conocimiento usando RAG."""
    try:
        api_key = get_openai_api_key()
        
        # Normalizar la pregunta para evitar problemas de codificación
        normalized_question = normalize_text(question)
        
        # Verificar si existe la base vectorial
        if not os.path.exists("./chroma_db"):
            return "No hay documentos financieros cargados. Por favor, sube algún documento PDF para poder analizar información financiera."
        
        # Cargar la base vectorial existente
        embeddings = OpenAIEmbeddings(api_key=api_key)
        db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        
        # Verificar si hay documentos en la base
        if db._collection.count() == 0:
            return "La base de datos está vacía. Por favor, sube algún documento PDF para poder analizar información financiera."
        
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
            api_key=api_key
        )
        
        retriever = db.as_retriever(search_kwargs={"k": 3})
        
        template = normalize_text("""
        Eres un asistente financiero experto. Usa la siguiente información para responder la pregunta del usuario.
        Si no sabes la respuesta, simplemente di que no tienes esa información y no inventes.
        
        Contexto: {context}
        
        Pregunta: {question}
        
        Respuesta:
        """)
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt}
        )
        
        result = qa_chain.invoke({"query": normalized_question})
        
        return result["result"]
    except Exception as e:
        return f"Error al consultar la base de conocimiento: {str(e)}"
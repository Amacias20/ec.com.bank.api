import os
import sys
import unicodedata
import re
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding='utf-8')

from flask import Flask, request, jsonify, send_file
from agent import query_agent
from flask_cors import CORS
from langchain.chat_models import ChatOpenAI
from tools import query_exchange_rate
from rag import initialize_vector_store
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from utils import get_openai_api_key

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '')
    
    result = query_agent(question)
    
    return jsonify({'response': result})

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No se envió ningún archivo'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Solo se permiten archivos PDF'}), 400

    # Guardar el archivo
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Extraer texto del PDF
    try:
        from PyPDF2 import PdfReader
        
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
            
        # Si el texto está vacío, puede ser un PDF escaneado
        if not text.strip():
            return jsonify({'error': 'No se pudo extraer texto del PDF. Podría ser un PDF escaneado.'}), 400
    except Exception as e:
        return jsonify({'error': f'Error al procesar el PDF: {str(e)}'}), 500

    # Agregar a la base vectorial
    api_key = get_openai_api_key()
    embeddings = OpenAIEmbeddings(api_key=api_key)
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    # Puedes dividir el texto en chunks aquí si es muy largo
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    db.add_texts(chunks)

    return jsonify({'message': 'PDF procesado y agregado a la base vectorial.'})

@app.route('/list_pdfs', methods=['GET'])
def list_pdfs():
    """Lista todos los PDFs subidos en el sistema con metadatos."""
    try:
        files = []
        for filename in os.listdir(UPLOAD_FOLDER):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                stats = os.stat(file_path)
                files.append({
                    'id': filename,  # Utilizamos el nombre como ID
                    'name': filename,
                    'size': stats.st_size,
                    'created_at': stats.st_ctime,
                    'modified_at': stats.st_mtime
                })
        return jsonify({'files': files})
    except Exception as e:
        return jsonify({'error': f'Error al listar PDFs: {str(e)}'}), 500

@app.route('/get_pdf/<pdf_id>', methods=['GET'])
def get_pdf(pdf_id):
    """Descarga un PDF específico por su ID (nombre del archivo)."""
    try:
        # Verificar si el id contiene caracteres no permitidos
        if '..' in pdf_id or '/' in pdf_id:
            return jsonify({'error': 'ID de PDF no válido'}), 400
            
        file_path = os.path.join(UPLOAD_FOLDER, pdf_id)
        
        # Verificar si el archivo existe
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            return jsonify({'error': 'PDF no encontrado'}), 404
            
        # Verificar si es un PDF
        if not pdf_id.lower().endswith('.pdf'):
            return jsonify({'error': 'El archivo solicitado no es un PDF'}), 400
            
        # Enviar el archivo para descarga
        return send_file(
            file_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=pdf_id
        )
    except Exception as e:
        return jsonify({'error': f'Error al obtener el PDF: {str(e)}'}), 500

def normalize_text(text):
    """Normaliza el texto eliminando caracteres especiales y acentos."""
    if not isinstance(text, str):
        return text
    text_normalized = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    text_normalized = re.sub(r'[^\w\s.,?!]', '', text_normalized)
    return text_normalized

def query_agent(question):
    """Consulta a OpenAI o a la herramienta de tipo de cambio según la pregunta."""
    normalized_question = normalize_text(question)
    
    # Palabras clave para identificar preguntas sobre divisas
    currency_keywords = [
        "divisa", "cambio", "tipo de cambio", "moneda", "dólar", "euro", "peso", "yen", "libra", "franco suizo"
    ]
    
    # Palabras clave para identificar preguntas sobre documentos
    document_keywords = [
        "documento", "pdf", "archivo", "informe", "reporte", "texto", "contenido"
    ]
    
    if any(k in normalized_question.lower() for k in currency_keywords):
        # Si la pregunta es sobre divisas, usa la herramienta
        return query_exchange_rate(normalized_question)
    elif any(k in normalized_question.lower() for k in document_keywords) or os.path.exists("./chroma_db"):
        # Si la pregunta es sobre documentos o existe la base vectorial, usa RAG
        from rag import query_rag
        return query_rag(normalized_question)
    else:
        chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        prompt = f"Responde SIEMPRE en español. Usuario: {normalized_question}"
        response = chat_model.invoke(prompt)
        return response.content

if __name__ == '__main__':
    app.run(debug=True)
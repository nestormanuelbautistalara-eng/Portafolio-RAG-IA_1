from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
import os
import shutil

# --- Borrar base de datos anterior ---
if os.path.exists("./chroma_db2"):
    shutil.rmtree("./chroma_db2")

# --- Cargar todos los PDFs de la carpeta docs/ ---
print("Cargando PDFs...")
documentos = []
for archivo in os.listdir("docs"):
    if archivo.endswith(".pdf"):
        ruta = os.path.join("docs", archivo)
        loader = PyPDFLoader(ruta)
        docs = loader.load()
        documentos.extend(docs)
        print(f"  cargado: {archivo} ({len(docs)} pagina(s))")

print(f"Total de paginas cargadas: {len(documentos)}")

# --- Dividir en fragmentos ---
print("Dividiendo en fragmentos...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
fragmentos = splitter.split_documents(documentos)
print(f"Fragmentos creados: {len(fragmentos)}")

# --- Crear embeddings y guardar ---
print("Creando embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(fragmentos, embeddings, persist_directory="./chroma_db2")
print("Base de datos lista")

# --- Conectar con Ollama ---
print("Conectando con Ollama...")
modelo = OllamaLLM(model="llama3.1:8b-instruct-q4_K_M")
retriever = db.as_retriever(search_kwargs={"k": 3})
print("Todo listo")
print("=" * 50)
print("Tengo cargados: Python, Git y Docker")
print("Escribe 'salir' para terminar")
print("=" * 50)

# --- Historial de conversacion ---
historial = []

while True:
    pregunta = input("\nTu: ")
    if pregunta.lower() == "salir":
        print("Hasta luego!")
        break

    # Buscar fragmentos relevantes
    docs = retriever.invoke(pregunta)
    contexto = "\n".join([d.page_content for d in docs])

    # Construir historial como texto
    historial_texto = ""
    for turno in historial[-4:]:
        historial_texto += f"Usuario: {turno['pregunta']}\nIA: {turno['respuesta']}\n\n"

    # Prompt con contexto e historial
    prompt = f"""Eres un asistente tecnico. Usa el contexto y el historial para responder.

Historial reciente:
{historial_texto}

Contexto de los documentos:
{contexto}

Pregunta actual: {pregunta}

Respuesta:"""

    respuesta = modelo.invoke(prompt)
    print(f"\nIA: {respuesta}")

    # Guardar en historial
    historial.append({"pregunta": pregunta, "respuesta": respuesta})

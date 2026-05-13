from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
import os
import shutil

# --- Inicializar FastAPI ---
app = FastAPI(title="Mi RAG API", version="1.0")

# --- Cargar todo al iniciar el servidor ---
print("Iniciando servidor y cargando documentos...")

if os.path.exists("./chroma_db3"):
    shutil.rmtree("./chroma_db3")

documentos = []
for archivo in os.listdir("docs"):
    if archivo.endswith(".pdf"):
        ruta = os.path.join("docs", archivo)
        loader = PyPDFLoader(ruta)
        docs = loader.load()
        documentos.extend(docs)
        print(f"  cargado: {archivo}")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
fragmentos = splitter.split_documents(documentos)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(fragmentos, embeddings, persist_directory="./chroma_db3")
retriever = db.as_retriever(search_kwargs={"k": 3})
modelo = OllamaLLM(model="llama3.1:8b-instruct-q4_K_M")

print("Servidor listo!")

# --- Modelos de datos ---
class Pregunta(BaseModel):
    texto: str
    historial: list = []

class Respuesta(BaseModel):
    respuesta: str
    fuentes: list

# --- Rutas de la API ---
@app.get("/")
def inicio():
    return {"estado": "funcionando", "version": "1.0", "docs": "/docs"}

@app.get("/salud")
def salud():
    return {"estado": "ok"}

@app.post("/preguntar", response_model=Respuesta)
def preguntar(pregunta: Pregunta):
    # Buscar fragmentos relevantes
    docs = retriever.invoke(pregunta.texto)
    contexto = "\n".join([d.page_content for d in docs])
    fuentes = list(set([d.metadata.get("source", "desconocido") for d in docs]))

    # Construir historial
    historial_texto = ""
    for turno in pregunta.historial[-4:]:
        historial_texto += f"Usuario: {turno.get('pregunta', '')}\nIA: {turno.get('respuesta', '')}\n\n"

    prompt = f"""Eres un asistente tecnico. Responde en base al contexto.

Historial:
{historial_texto}

Contexto:
{contexto}

Pregunta: {pregunta.texto}

Respuesta:"""

    respuesta = modelo.invoke(prompt)

    return Respuesta(respuesta=respuesta, fuentes=fuentes)

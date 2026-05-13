from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ddgs import DDGS
from datetime import datetime
import os, math

print("Cargando documentos...")
documentos = []
for archivo in os.listdir("docs"):
    if archivo.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join("docs", archivo))
        documentos.extend(loader.load())
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
fragmentos = splitter.split_documents(documentos)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(fragmentos, embeddings, persist_directory="./chroma_db3")
retriever = db.as_retriever(search_kwargs={"k": 3})
modelo = OllamaLLM(model="llama3.1:8b-instruct-q4_K_M")
print("Listo")

def herramienta_calculadora(e):
    try:
        r = eval(e.replace("^","**"),{"__builtins__":{}},{"sqrt":math.sqrt,"pi":math.pi,"abs":abs})
        return "Resultado: " + str(r)
    except Exception as ex:
        return "Error: " + str(ex)

def herramienta_fecha_hora(c):
    a = datetime.now()
    fmt_fecha = "%A %d de %B de %Y"
    fmt_hora = "%H:%M:%S"
    return "Fecha: " + a.strftime(fmt_fecha) + " | Hora: " + a.strftime(fmt_hora)

def herramienta_buscar_docs(p):
    docs = retriever.invoke(p)
    if not docs:
        return "Sin resultados en documentos."
    return "\n---\n".join([d.page_content for d in docs])

def herramienta_buscar_web(c):
    try:
        with DDGS() as d:
            r = list(d.text(c, max_results=3))
        return "\n".join(["- " + x["title"] + ": " + x["body"] for x in r]) if r else "Sin resultados."
    except Exception as ex:
        return "Error: " + str(ex)

HERRAMIENTAS = {
    "calculadora": {"funcion": herramienta_calculadora, "descripcion": "Para calculos matematicos"},
    "fecha_hora":  {"funcion": herramienta_fecha_hora,  "descripcion": "Para saber fecha y hora actual"},
    "buscar_docs": {"funcion": herramienta_buscar_docs, "descripcion": "Para preguntas sobre Python Git Docker"},
    "buscar_web":  {"funcion": herramienta_buscar_web,  "descripcion": "Para buscar en internet temas no en documentos"}
}

def decidir_herramienta(pregunta):
    lista = "\n".join(["- " + n + ": " + i["descripcion"] for n,i in HERRAMIENTAS.items()])
    prompt = "Elige UNA herramienta.\nHerramientas:\n" + lista + "\nPregunta: " + pregunta + "\nResponde SOLO el nombre:"
    decision = modelo.invoke(prompt).strip().lower()
    for nombre in HERRAMIENTAS:
        if nombre in decision:
            return nombre
    return "buscar_docs"

def agente(pregunta):
    tool = decidir_herramienta(pregunta)
    print("[herramienta: " + tool + "]")
    resultado = HERRAMIENTAS[tool]["funcion"](pregunta)
    prompt = "Responde claramente usando esta info:\n" + resultado + "\n\nPregunta: " + pregunta + "\nRespuesta:"
    return modelo.invoke(prompt), tool

print("=" * 50)
print("AGENTE ACTIVO")
print("=" * 50)
while True:
    pregunta = input("\nTu: ")
    if pregunta.lower() == "salir":
        print("Hasta luego!")
        break
    respuesta, tool = agente(pregunta)
    print("\nIA [" + tool + "]: " + respuesta)

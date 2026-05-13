from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM

print("Cargando PDF...")
loader = PyPDFLoader("docs/manual_python.pdf")
documentos = loader.load()
print(f"PDF cargado: {len(documentos)} pagina(s)")

print("Dividiendo en fragmentos...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
fragmentos = splitter.split_documents(documentos)
print(f"Fragmentos creados: {len(fragmentos)}")

print("Creando embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Guardando en ChromaDB...")
db = Chroma.from_documents(fragmentos, embeddings, persist_directory="./chroma_db")
print("Base de datos lista")

print("Conectando con Ollama...")
modelo = OllamaLLM(model="llama3.1:8b-instruct-q4_K_M")
retriever = db.as_retriever(search_kwargs={"k": 3})
print("Todo listo")
print("=" * 50)

while True:
    pregunta = input("\nTu: ")
    if pregunta.lower() == "salir":
        print("Hasta luego!")
        break
    docs = retriever.invoke(pregunta)
    contexto = "\n".join([d.page_content for d in docs])
    prompt = f"Usa este contexto para responder:\n{contexto}\n\nPregunta: {pregunta}"
    respuesta = modelo.invoke(prompt)
    print(f"\nIA: {respuesta}")

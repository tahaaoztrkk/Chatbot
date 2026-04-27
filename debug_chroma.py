from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vector_store = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")

data = vector_store.get()
sources = {}
for metadata in data['metadatas']:
    src = metadata.get('source', 'Unknown')
    sources[src] = sources.get(src, 0) + 1

print("ChromaDB'deki Belgeler ve Parca (Chunk) Sayilari:")
for src, count in sources.items():
    print(f"- {src}: {count} chunks")

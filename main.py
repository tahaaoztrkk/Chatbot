import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Ollama Phi FastAPI RAG Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Initialize the locally running Ollama model using the new package class
llm = ChatOllama(model="phi3-yerel", temperature=0)

# 2. Initialize the Sentence-Transformers embedding model
# Opted for a multilingual model robust on Turkish texts as per requirements.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# 3. Initialize ChromaDB instance (Persistent)
# Stores data persistently inside the chroma_db folder in the current directory.
persist_directory = "./chroma_db"
vector_store = Chroma(embedding_function=embeddings, persist_directory=persist_directory)

# Basic Text Splitter configuration
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

class ChatRequest(BaseModel):
    text: str
    history: Optional[List[dict]] = None

class ChatResponse(BaseModel):
    reply: str
    sources: List[str] = []

@app.post("/upload-doc")
async def upload_doc_endpoint(file: UploadFile = File(...)):
    """
    Receives a PDF or TXT file, reads/splits its contents, creates multilingual embeddings,
    and stores them in ChromaDB.
    """
    if file.filename is None:
        raise HTTPException(status_code=400, detail="Filename missing")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".txt", ".pdf"]:
        raise HTTPException(status_code=400, detail="Only .txt and .pdf files are supported")

    # Save to a temporary file locally for loader processing
    temp_file_path = f"temp_{file.filename}"
    file_bytes = await file.read()
    with open(temp_file_path, "wb") as buffer:
        buffer.write(file_bytes)

    try:
        # Load elements based on file extension
        if ext == ".pdf":
            loader = PyMuPDFLoader(temp_file_path)
            docs = loader.load()
        elif ext == ".txt":
            loader = TextLoader(temp_file_path, encoding="utf-8")
            docs = loader.load()

        # Split documents into chunks to fit embedding / LLM constraints
        chunks = text_splitter.split_documents(docs)

        # Insert successfully grouped documents to vector store
        vector_store.add_documents(chunks)
        
        return {
            "message": f"Successfully processed '{file.filename}'.",
            "chunks_added": len(chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        # Cleanup temporary file to avoid clutter
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Searches ChromaDB for relevant context, pushes a context-enriched prompt to Ollama,
    and returns its reply.
    """
    # 1. Retrieve the Top 3 relevant chunks asynchronously
    retrieved_docs = await vector_store.asimilarity_search(request.text, k=3)
    
    # 2. Join their texts to form a context body
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    # Extract unique source names (like the filename)
    source_names = list(set([doc.metadata.get("source", "Unknown") for doc in retrieved_docs]))
    
    # Extract and format history
    history_text = ""
    if request.history:
        for msg in request.history:
            role = "Kullanıcı" if msg.get("role") == "user" else "Asistan"
            history_text += f"{role}: {msg.get('content')}\n"
        history_text = f"\nÖnceki Konuşmalar:\n{history_text}\n"
        
    # 3. Create context-aware Chat messages
    messages = [
        SystemMessage(content=f"You are an intelligent assistant. Answer the user's question based strictly on the provided Context. If the answer cannot be found in the Context, return exactly 'I cannot find the answer in the provided document.' Do not make up answers.\n\nContext:\n{context_text}{history_text}"),
        HumanMessage(content=request.text)
    ]

    # 4. Asynchronously invoke Ollama with properly structured chat templates
    response = await llm.ainvoke(messages)
    
    return ChatResponse(reply=response.content, sources=source_names)

class AnalyzeRequest(BaseModel):
    text: str
    file_path: str

@app.post("/analyze")
async def analyze_endpoint(request: AnalyzeRequest):
    """
    Pandas DataFrame ajanı oluşturur ve belirtilen CSV dosyasını LLM ile analiz eder.
    """
    clean_path = request.file_path.strip('\"\'')
    if not os.path.exists(clean_path):
        raise HTTPException(status_code=400, detail="Belirtilen dosya bulunamadı.")
    
    try:
        try:
            df = pd.read_csv(clean_path, encoding='utf-8', sep=None, engine='python', on_bad_lines='skip')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(clean_path, encoding='cp1254', sep=None, engine='python', on_bad_lines='skip')
            except Exception:
                df = pd.read_csv(clean_path, encoding='utf-8', encoding_errors='ignore', sep=None, engine='python', on_bad_lines='skip')
        
        agent = create_pandas_dataframe_agent(
            llm, # Mevcut ChatOllama (phi3-yerel) modelimizi kullanıyoruz
            df,
            verbose=True,
            allow_dangerous_code=True,
            handle_parsing_errors=True # Küçük modeller genelde ayrıştırma hataları yaptığı için önemlidir
        )
        
        try:
            response = await agent.ainvoke({"input": request.text})
            return {"reply": response.get("output", str(response))}
        except Exception as e:
            error_str = str(e)
            # Eğer Phi-3 modeli cevabı bulur ancak LangChain re-act tablosuna formatlanamazsa, bulduğu cevabı zorla ayıklıyoruz:
            if "Could not parse LLM output:" in error_str:
                raw_output = error_str.split("Could not parse LLM output: ")[-1].strip("`\n ")
                return {"reply": raw_output}
            else:
                raise HTTPException(status_code=500, detail=error_str)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class AskRequest(BaseModel):
    text: str
    file_path: Optional[str] = None
    history: Optional[List[dict]] = None

@app.post("/ask")
async def ask_endpoint(request: AskRequest):
    """
    Akıllı Yönlendirici (Router). Gelen soruyu analiz edip RAG (TEXT) veya Pandas (DATA) ajanına yönlendirir.
    """
    router_messages = [
        SystemMessage(content="You are a binary decision router. Classify the user input. Output exactly 'DATA' or 'TEXT' and nothing else!"),
        HumanMessage(content="Hangi kategoriden kaç satış yapıldı?"),
        AIMessage(content="DATA"),
        HumanMessage(content="Mektubu kim yazmış?"),
        AIMessage(content="TEXT"),
        HumanMessage(content="En yüksek gelir nerede veri analiz et?"),
        AIMessage(content="DATA"),
        HumanMessage(content="Taha'nın CV'sinde bahsettiği teknolojiler nelerdir?"),
        AIMessage(content="TEXT"),
        HumanMessage(content="What is the main topic of the internship?"),
        AIMessage(content="TEXT"),
        HumanMessage(content="Sum up all the quantities in the data table."),
        AIMessage(content="DATA"),
        HumanMessage(content=request.text)
    ]

    # Modelden yönlendirme kararını al
    router_response = await llm.ainvoke(router_messages)
    decision = router_response.content.strip().upper()
    print(f"ROUTER DECISION FOR '{request.text}': {decision}") # Termilae loglamak için
    
    # Karar mekanizması (chatty model koruması)
    is_data = ("DATA" in decision and "TEXT" not in decision) or decision == "DATA"
    
    if is_data:
        clean_path = request.file_path.strip('\"\'') if request.file_path else None
        if not clean_path or not os.path.exists(clean_path):
            raise HTTPException(status_code=400, detail="Soru veri analizi içeriyor ancak geçerli bir file_path (CSV) verilmedi.")
        
        try:
            try:
                df = pd.read_csv(clean_path, encoding='utf-8', sep=None, engine='python', on_bad_lines='skip')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(clean_path, encoding='cp1254', sep=None, engine='python', on_bad_lines='skip')
                except Exception:
                    df = pd.read_csv(clean_path, encoding='utf-8', encoding_errors='ignore', sep=None, engine='python', on_bad_lines='skip')
                
            agent = create_pandas_dataframe_agent(
                llm, df, verbose=True, allow_dangerous_code=True, handle_parsing_errors=True
            )
            try:
                response = await agent.ainvoke({"input": request.text})
                return {"routed_to": "DATA", "reply": response.get("output", str(response))}
            except Exception as e:
                error_str = str(e)
                if "Could not parse LLM output:" in error_str:
                    raw_output = error_str.split("Could not parse LLM output: ")[-1].strip("`\n ")
                    return {"routed_to": "DATA", "reply": raw_output}
                else:
                    raise HTTPException(status_code=500, detail=error_str)
        except HTTPException as he:
            raise he
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
            
    else:
        # Varsayılan / Fallback durumunda RAG (TEXT) çalıştır
        retrieved_docs = await vector_store.asimilarity_search(request.text, k=5)
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        source_names = list(set([doc.metadata.get("source", "Unknown") for doc in retrieved_docs]))
        
        history_text = ""
        if request.history:
            for msg in request.history:
                role = "Kullanıcı" if msg.get("role") == "user" else "Asistan"
                history_text += f"{role}: {msg.get('content')}\n"
            history_text = f"\nÖnceki Konuşmalar:\n{history_text}\n"

        messages = [
            SystemMessage(content=f"You are an intelligent assistant. Answer the user's question based strictly on the provided Context. If the answer cannot be found in the Context, return exactly 'I cannot find the answer in the provided document.' Do not make up answers.\n\nContext:\n{context_text}{history_text}"),
            HumanMessage(content=request.text)
        ]
        
        response = await llm.ainvoke(messages)
        return {"routed_to": "TEXT", "reply": response.content, "sources": source_names}
                
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

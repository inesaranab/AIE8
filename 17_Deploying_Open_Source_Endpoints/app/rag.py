import tiktoken
import os
from typing import TypedDict, List, Annotated, NotRequired
from functools import lru_cache
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_together import TogetherEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START
from app.model import get_chat_model

def _tiktoken_len(text: str) -> int:
    """Return token length using tiktoken"""
    # Use cl100k_base encoding (GPT-4/3.5) as gpt-oss is not a standard tiktoken model
    tokens = tiktoken.get_encoding("cl100k_base").encode(text)
    return len(tokens)

class _RAGState(TypedDict):
    """State schema for the simple two-step RAG"""
    question: str
    context: List[Document]
    response: str  # Added response field for the generated answer

def _build_rag_graph(data_dir: str) -> "CompiledGraph":
    """Construct and compile a minimal RAG graph.

    Steps:
    1) Load PDFs from `data_dir` recursively (best-effort).
    2) Split documents into token-aware chunks.
    3) Create embeddings and an in-memory Qdrant vector store retriever.
    4) Define a chat prompt and generation model.
    5) Wire a two-node graph: retrieve -> generate.
    """
    # Load PDFs from data directory (recursive)
    try:
        directory_loader = DirectoryLoader(data_dir, glob="*.pdf", loader_cls = PyMuPDFLoader)
        documents = directory_loader.load()
    except Exception:
        documents = []
    
    # Split the documents
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
    except Exception:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

    # Reduce chunk size to 400 tokens to stay under Together AI embedding limit (512 tokens)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50, length_function=_tiktoken_len)
    chunks = text_splitter.split_documents(documents)
  
    # Create embeddings and an in-memory Qdrant vector store retriever
    embedding_model = TogetherEmbeddings(model="BAAI/bge-large-en-v1.5")
    qdrant_vectorstore = Qdrant.from_documents(
        documents=chunks, embedding=embedding_model, location=":memory:"
    )
    retriever = qdrant_vectorstore.as_retriever()

    # Prompt and model
    human_template = (
        "\n#CONTEXT:\n{context}\n\nQUERY:\n{query}\n\n"
        "Use the provide context to answer the provided user query. "
        "Only use the provided context to answer the query. If you do not know the answer, or it's not contained in the provided context respond with \"I don't know\""
    )
    chat_prompt = ChatPromptTemplate.from_messages([("human", human_template)])
    generator_llm = get_chat_model(model_name="openai/gpt-oss-20b")

    def retrieve(state: _RAGState) -> _RAGState:
        retrieved_docs = retriever.invoke(state["question"]) if retriever else []
        return {"context": retrieved_docs}  # type: ignore

    def generate(state: _RAGState) -> _RAGState:
        generator_chain = chat_prompt | generator_llm | StrOutputParser()
        response_text = generator_chain.invoke(
            {"query": state["question"], "context": state.get("context", [])}
        )
        return {"response": response_text}  # type: ignore

    graph_builder = StateGraph(_RAGState)
    graph_builder = graph_builder.add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    return graph_builder.compile()


@lru_cache(maxsize=1)
def _get_rag_graph():
    """Return a cached compiled RAG graph built from RAG_DATA_DIR."""
    data_dir = os.environ.get("RAG_DATA_DIR", "data")
    return _build_rag_graph(data_dir)


@tool
def retrieve_information(
    query: Annotated[str, "query to ask the retrieve information tool"]
):
    """Use Retrieval Augmented Generation to retrieve information about how people are using AI in their daily work."""
    graph = _get_rag_graph()
    result = graph.invoke({"question": query})
    # Prefer returning the response string if available
    if isinstance(result, dict) and "response" in result:
        return result["response"]
    return result


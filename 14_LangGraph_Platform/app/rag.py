from typing_extensions import TypedDict
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

from functools import lru_cache
import os

from langchain_core.tools import tool


class RAGState(TypedDict):
    """State schema for a simple RAG graph: retrieve then generate response"""
    question: str
    context: list[Document] | None
    response: str | None

def _build_rag_graph(data_dir: str) -> "CompiledGraph":
    """Construct and compile a minimal RAG graph.

    Steps:
    1) Load PDFs from `data_dir` recursively (best-effort).
    2) Split documents into token-aware chunks.
    3) Create embeddings and an in-memory Qdrant vector store retriever.
    4) Define a chat prompt and generation model.
    5) Wire a two-node graph: retrieve -> generate.
    """
    # 1) Load PDFs from `data_dir` recursively (best-effort).
    try:
        directory_loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyMuPDFLoader)
        documents = directory_loader.load()
    except Exception:
        documents = []
        
        # Preprocess the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # Embedding model and vector store
    embedding_model = OpenAIEmbeddings()

    # Qdrant vectorstore
    qdrant_client = QdrantClient(":memory:")
    qdrant_client.create_collection(
        collection_name="documents",
        vectors_config=VectorParams(size=len(embedding_model.embed_query("Hello")), distance=Distance.COSINE),
    )

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name="documents",
        embedding=embedding_model,
    )

    # Add documents to the vector store
    _ = vector_store.add_documents(texts)

    # Create retriever
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})


    ### RAG ChatTemplate for Generator
    TEMPLATE = """
    #CONTEXT:
    {context}

    #QUERY:
    {question}

    Use the provided context to answer the provided user query.
    Only use the provided context to answer the query. If you do not know the answer, or it's not contained in the provided context, respond with "I don't know".
    Keep the answer concise and helpful.
    """

    chat_prompt = ChatPromptTemplate.from_messages([
        ("human", TEMPLATE)
    ])

    # Generator LLM
    llm_generator = ChatOpenAI(model="gpt-4o", temperature=0)  # type: ignore

    # Generator LCEL chain
    generator_chain = chat_prompt | llm_generator | StrOutputParser()

    def retrieve(state: RAGState) -> RAGState:
        """Retrieve relevant FAQ documents from vector store"""
        question = state["question"]
        # Use retriever to get relevant documents
        retrieved_docs = retriever.invoke(question)
        return {"context": retrieved_docs}
    
    def generate(state: RAGState) -> RAGState:
        """Generate response using LCEL generator chain based on retrieved documents"""
        question = state["question"]
        context = state.get("context", [])
        response = generator_chain.invoke({"question": question, "context": context})
        return {"response": response}
    
    builder = StateGraph(RAGState)
    builder.add_sequence([retrieve, generate])
    builder.add_edge(START, "retrieve")
    return builder.compile()

@lru_cache(maxsize=1)
def _get_rag_graph():
    """Return a cached compiled RAG graph built from RAG_DATA_DIR"""
    data_dir = os.environ.get("RAG_DATA_DIR", "Data")
    return _build_rag_graph(data_dir)

@tool
def retrieve_information(question: str) -> str:
    """Retrieve information from the RAG graph"""
    graph = _get_rag_graph()
    result = graph.invoke({"question": question})
    if isinstance(result, dict) and "response" in result:
        return result["response"]
    return result


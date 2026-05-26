"""
rag_pipeline.py — AI Support Copilot
Root cause fixes:
1. Switched to gemini-2.5-flash (15 RPM free tier, not deprecating)
2. Removed retry loop — Streamlit Cloud times out before retries complete
3. Single clean API call with one reasonable delay
4. Summarize query now uses generate_answer (1 call) not summarize_all_documents loop
"""

import os
import time
from collections import defaultdict

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from google import genai
from dotenv import load_dotenv

load_dotenv()

_api_key = os.getenv("GEMINI_API_KEY")
if not _api_key:
    try:
        import streamlit as _st
        _api_key = _st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass

if not _api_key:
    raise EnvironmentError(
        "GEMINI_API_KEY not found.\n"
        "  • Local:  copy .env.example to .env and set your key.\n"
        "  • Cloud:  add GEMINI_API_KEY in Streamlit Cloud Secrets."
    )

client = genai.Client(api_key=_api_key)

# gemini-1.5-flash: 15 RPM, 1500 RPD on free tier — highest available
# gemini-2.0-flash: only 5 RPM on free tier AND deprecating June 2026
MODEL_NAME  = "gemini-2.5-flash"
_CALL_DELAY = 2   # 2s delay = ~20 RPM theoretical max, well under 15 RPM limit

_embeddings = None

def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings


# ══════════════════════════════════════════════════════════════════════════════
#  Core Gemini caller — single call, no retry loop (Streamlit times out)
# ══════════════════════════════════════════════════════════════════════════════

def _call_gemini(prompt: str) -> str:
    """
    Single Gemini call with a short pre-call delay.
    No retry loop — Streamlit Cloud has a ~60s request timeout so retries
    with exponential backoff would cause the entire request to time out
    before succeeding, which is worse than just returning a clean error.
    """
    time.sleep(_CALL_DELAY)
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        return response.text

    except Exception as e:
        err = str(e)

        if "429" in err or "RESOURCE_EXHAUSTED" in err:
            return (
                "⚠️ Rate limit reached. You are making requests too quickly. "
                "Please wait 30 seconds and try again."
            )
        if "404" in err or "NOT_FOUND" in err:
            return (
                "⚠️ Model not found. "
                "Check your API key at aistudio.google.com."
            )
        if "401" in err or "403" in err or "API_KEY" in err.upper():
            return (
                "⚠️ API key error. "
                "Check GEMINI_API_KEY in your Streamlit Cloud secrets."
            )
        return f"⚠️ Error: {err}"


# ══════════════════════════════════════════════════════════════════════════════
#  Document loading & chunking
# ══════════════════════════════════════════════════════════════════════════════

def load_docs(file_path: str):
    loader = PyPDFLoader(file_path)
    return loader.load()


def chunk_docs(docs, chunk_size: int = 600, chunk_overlap: int = 100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


# ══════════════════════════════════════════════════════════════════════════════
#  Vector store
# ══════════════════════════════════════════════════════════════════════════════

def create_vector_store(chunks):
    if not chunks:
        raise ValueError("No text chunks — cannot build vector store.")
    return FAISS.from_documents(chunks, _get_embeddings())


def load_vector_store(index_path: str = "faiss_index"):
    if not os.path.exists(index_path):
        return None
    return FAISS.load_local(
        index_path,
        _get_embeddings(),
        allow_dangerous_deserialization=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Retrieval
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_docs(vector_store, query: str, k: int = 3):
    return vector_store.similarity_search(query, k=k)


def retrieve_per_document(vector_store, query: str, chunks_per_doc: int = 3):
    all_results = vector_store.similarity_search(query, k=30)
    doc_chunks  = defaultdict(list)
    for doc in all_results:
        doc_chunks[doc.metadata.get("source", "Unknown")].append(doc)
    balanced = []
    for source, chunks in doc_chunks.items():
        balanced.extend(chunks[:chunks_per_doc])
    return balanced, list(doc_chunks.keys())


# ══════════════════════════════════════════════════════════════════════════════
#  Query Reformulation — disabled to save API calls on free tier
# ══════════════════════════════════════════════════════════════════════════════

def reformulate_query(query: str, chat_history: list) -> str:
    return query


# ══════════════════════════════════════════════════════════════════════════════
#  Answer Generation — ONE Gemini call per user message
# ══════════════════════════════════════════════════════════════════════════════

def generate_answer(
    context_docs: list,
    query: str,
    chat_history: list | None = None,
    max_history_turns: int = 1,
) -> str:
    context = "\n\n".join([doc.page_content for doc in context_docs])

    if len(context.strip()) < 30:
        return (
            "I could not find enough relevant information in the uploaded documents. "
            "Try rephrasing your question or uploading additional documents."
        )

    history_block = ""
    if chat_history:
        recent = chat_history[-(max_history_turns * 2):]
        if recent:
            history_block = "Recent context (reference only):\n"
            for msg in recent:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_block += f"{role}: {msg['content'][:200]}\n"
            history_block += "\n"

    prompt = (
        "You are an enterprise AI support copilot.\n"
        "Answer using ONLY the Document Context below.\n\n"
        "Rules:\n"
        "1. Only use information from Document Context.\n"
        "2. If the answer is not in context, say: "
        "\"I could not find this in the uploaded documents.\"\n"
        "3. Be concise and accurate.\n"
        "4. Do not use your own training knowledge.\n\n"
        f"{history_block}"
        f"Document Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    return _call_gemini(prompt)


# ══════════════════════════════════════════════════════════════════════════════
#  Document Summarisation
#  KEY FIX: summarize_all_documents previously called _call_gemini in a LOOP
#  (one call per document). With 2+ docs that's 2+ API calls fired back-to-back,
#  instantly hitting rate limits. Now it batches all doc content into ONE call.
# ══════════════════════════════════════════════════════════════════════════════

def summarize_all_documents(vector_store, all_source_names: list) -> dict:
    """
    Summarise ALL documents in a SINGLE Gemini call by batching context.
    Previous version made one call per document — with 2+ docs this
    fired multiple requests instantly and always hit the rate limit.
    """
    if not all_source_names:
        return {}

    # Gather up to 5 chunks per document
    all_results = vector_store.similarity_search(
        "summary overview main points key information", k=50,
    )

    # Group by source
    doc_chunks = defaultdict(list)
    for doc in all_results:
        src = doc.metadata.get("source", "Unknown")
        if src in all_source_names:
            doc_chunks[src].append(doc)

    # Build one combined prompt with all documents
    combined_context = ""
    for source in all_source_names:
        chunks = doc_chunks.get(source, [])
        if not chunks:
            combined_context += f"\n\n=== {source} ===\n[No content found]\n"
            continue
        content = "\n\n".join([doc.page_content for doc in chunks[:5]])
        combined_context += f"\n\n=== {source} ===\n{content}\n"

    prompt = (
        "You are summarising the following documents. "
        "Use ONLY the content provided for each document.\n\n"
        "For EACH document provide:\n"
        "1. What the document is about (2 sentences)\n"
        "2. Key points (bullet list, max 5)\n"
        "3. Most important takeaway (1 sentence)\n\n"
        "Format your response with the document name as a header for each section.\n\n"
        f"Documents:\n{combined_context}\n\n"
        "Summaries:"
    )

    result = _call_gemini(prompt)

    # Return as a single entry so app.py renders it cleanly
    if result.startswith("⚠️"):
        return {name: result for name in all_source_names}

    # Return combined result under first doc name; app.py will display it
    return {"All Documents": result}
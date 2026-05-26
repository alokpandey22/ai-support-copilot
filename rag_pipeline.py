"""
rag_pipeline.py — AI Support Copilot
Rate-limit safe: pacing, exponential backoff, reduced payloads.
"""

import os
import time
import random
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

MODEL_NAME    = "gemini-2.0-flash"
_CALL_DELAY   = 6      # seconds before every call — targets ~9 RPM (free tier = 15 RPM)
_MAX_RETRIES  = 4
_BASE_BACKOFF = 20     # seconds for first retry; doubles each attempt

_embeddings = None

def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings


# ══════════════════════════════════════════════════════════════════════════════
#  Core Gemini caller — pacing + retry
# ══════════════════════════════════════════════════════════════════════════════

def _call_gemini(prompt: str) -> str:
    """
    Call Gemini with pre-call sleep and exponential backoff on 429.
    Retry schedule: 20s → 40s → 80s → 160s (+ jitter up to 5s each).
    """
    for attempt in range(_MAX_RETRIES):
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
                if attempt < _MAX_RETRIES - 1:
                    backoff = _BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 5)
                    time.sleep(backoff)
                    continue
                return (
                    "⚠️ Still rate limited after retries. "
                    "Please wait 60 seconds and try again. "
                    "Free tier limit: 15 requests/minute."
                )

            if "404" in err or "NOT_FOUND" in err:
                return (
                    "⚠️ Model not available. "
                    "Check your API key has access to gemini-2.0-flash "
                    "at aistudio.google.com."
                )

            if "401" in err or "403" in err or "API_KEY" in err.upper():
                return (
                    "⚠️ API key error. "
                    "Check GEMINI_API_KEY in your .env or Streamlit secrets."
                )

            return f"⚠️ Error: {err}"

    return "⚠️ Failed after multiple retries. Please wait 60 seconds and try again."


# ══════════════════════════════════════════════════════════════════════════════
#  Document loading & chunking
# ══════════════════════════════════════════════════════════════════════════════

def load_docs(file_path: str):
    loader = PyPDFLoader(file_path)
    return loader.load()


def chunk_docs(docs, chunk_size: int = 600, chunk_overlap: int = 100):
    """600 chars per chunk (down from 1200) — cuts TPM by ~50% per call."""
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
    """k=3 (down from 5) — less context tokens per call."""
    return vector_store.similarity_search(query, k=k)


def retrieve_per_document(vector_store, query: str, chunks_per_doc: int = 3):
    """Balanced retrieval across docs. chunks_per_doc=3, over-fetch k=30."""
    all_results = vector_store.similarity_search(query, k=30)
    doc_chunks  = defaultdict(list)
    for doc in all_results:
        doc_chunks[doc.metadata.get("source", "Unknown")].append(doc)
    balanced = []
    for source, chunks in doc_chunks.items():
        balanced.extend(chunks[:chunks_per_doc])
    return balanced, list(doc_chunks.keys())


# ══════════════════════════════════════════════════════════════════════════════
#  Query Reformulation — disabled on free tier (saves 1 API call per message)
# ══════════════════════════════════════════════════════════════════════════════

def reformulate_query(query: str, chat_history: list) -> str:
    """Disabled to save free-tier quota. Re-enable on paid plan."""
    return query


# ══════════════════════════════════════════════════════════════════════════════
#  Answer Generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_answer(
    context_docs: list,
    query: str,
    chat_history: list | None = None,
    max_history_turns: int = 1,
) -> str:
    """
    Generate grounded answer. Token optimizations:
    - max_history_turns=1 (was 3)
    - history truncated to 200 chars each (was 400)
    - chunks already 600 chars from chunk_docs
    """
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
            history_block = "Recent context (reference only — not a fact source):\n"
            for msg in recent:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_block += f"{role}: {msg['content'][:200]}\n"
            history_block += "\n"

    prompt = (
        "You are an enterprise AI support copilot.\n"
        "Answer using ONLY the Document Context below.\n\n"
        "Rules:\n"
        "1. Only use information from Document Context.\n"
        "2. If the answer is not in the context, say: "
        "\"I could not find this in the uploaded documents.\"\n"
        "3. Be concise and accurate.\n"
        "4. Do not use your training knowledge.\n\n"
        f"{history_block}"
        f"Document Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    return _call_gemini(prompt)


# ══════════════════════════════════════════════════════════════════════════════
#  Document Summarisation
# ══════════════════════════════════════════════════════════════════════════════

def summarize_all_documents(vector_store, all_source_names: list) -> dict:
    """
    Summarise each document. Token optimizations:
    - k=20 (was 60)
    - 5 chunks per doc (was 20)
    - Each chunk 600 chars
    - Stops gracefully on rate limit
    """
    summaries = {}

    for i, source in enumerate(all_source_names):
        all_results = vector_store.similarity_search(
            "summary overview main points key information", k=20,
        )
        doc_chunks = [
            doc for doc in all_results
            if doc.metadata.get("source") == source
        ]

        if not doc_chunks:
            summaries[source] = "Could not retrieve content from this document."
            continue

        context = "\n\n".join([doc.page_content for doc in doc_chunks[:5]])

        prompt = (
            f"Summarise the document \"{source}\" using ONLY the context below.\n\n"
            "Provide:\n"
            "1. What this document is about (2 sentences max)\n"
            "2. Key points (bullet list, max 5)\n"
            "3. Most important takeaway (1 sentence)\n\n"
            f"Context:\n{context}\n\nSummary:"
        )

        result = _call_gemini(prompt)
        summaries[source] = result

        if result.startswith("⚠️"):
            for remaining in all_source_names[i + 1:]:
                summaries[remaining] = (
                    "⚠️ Skipped — rate limit reached. "
                    "Wait 60 seconds and try summarising again."
                )
            break

    return summaries
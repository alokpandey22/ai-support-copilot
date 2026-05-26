"""
rag_pipeline.py
───────────────
Core RAG pipeline for AI Support Copilot.
Rate-limit safe: pacing delays, reduced payloads, exponential backoff retry.
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

# ── Load environment variables ─────────────────────────────────────────────────
load_dotenv()

# ── Gemini client ──────────────────────────────────────────────────────────────
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
        "  • Local:  copy .env.example → .env and set your key.\n"
        "  • Cloud:  add GEMINI_API_KEY in the Streamlit Cloud Secrets dashboard."
    )

client = genai.Client(api_key=_api_key)

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_NAME = "gemini-2.0-flash"

# ── Rate limit config ──────────────────────────────────────────────────────────
# Free tier: 15 RPM. We target ~8 RPM to stay well under.
# Each call sleeps 4s before firing. On 429, retry with exponential backoff.
_CALL_DELAY    = 4      # seconds to sleep before every API call
_MAX_RETRIES   = 3      # max retry attempts on 429
_BASE_BACKOFF  = 15     # seconds for first retry (doubles each attempt)

# ── Embeddings (lazy-loaded) ───────────────────────────────────────────────────
_embeddings = None

def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings


# ══════════════════════════════════════════════════════════════════════════════
#  Core API caller — pacing + retry
# ══════════════════════════════════════════════════════════════════════════════

def _call_gemini(prompt: str) -> str:
    """
    Call Gemini with:
      • 4-second pre-call sleep (pacing — stays under 15 RPM)
      • Exponential backoff retry on 429: 15s → 30s → 60s + jitter
      • Clean user-facing error messages for all failure modes
    """
    for attempt in range(_MAX_RETRIES):
        time.sleep(_CALL_DELAY)   # pace every call regardless of attempt
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )
            return response.text

        except Exception as e:
            err = str(e)

            # ── 429 / resource exhausted → backoff and retry ───────────────────
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                if attempt < _MAX_RETRIES - 1:
                    backoff = _BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 5)
                    time.sleep(backoff)
                    continue
                return (
                    "⚠️ Rate limit: too many requests per minute. "
                    "The free tier allows 15 requests/minute. "
                    "Please wait 60 seconds and try again."
                )

            # ── Model not found ────────────────────────────────────────────────
            if "404" in err or "NOT_FOUND" in err:
                return (
                    "⚠️ Model not available. "
                    "Verify your API key has access to gemini-2.0-flash "
                    "at aistudio.google.com."
                )

            # ── Auth error ─────────────────────────────────────────────────────
            if "401" in err or "403" in err or "API_KEY" in err.upper():
                return (
                    "⚠️ API key error. "
                    "Check your GEMINI_API_KEY in Streamlit secrets."
                )

            # ── Any other error ────────────────────────────────────────────────
            return f"⚠️ Error generating answer: {err}"

    return "⚠️ Failed after multiple retries. Please wait 60 seconds and try again."


# ══════════════════════════════════════════════════════════════════════════════
#  Document loading & chunking
# ══════════════════════════════════════════════════════════════════════════════

def load_docs(file_path: str):
    """Load a PDF and return LangChain Document objects."""
    loader = PyPDFLoader(file_path)
    return loader.load()


def chunk_docs(docs, chunk_size: int = 600, chunk_overlap: int = 100):
    """
    Split documents into overlapping chunks.
    600 chars (down from 1200) cuts TPM usage by ~50% per API call.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


# ══════════════════════════════════════════════════════════════════════════════
#  Vector store
# ══════════════════════════════════════════════════════════════════════════════

def create_vector_store(chunks):
    """Build a FAISS vector store from document chunks."""
    if not chunks:
        raise ValueError("No text chunks provided — cannot build vector store.")
    return FAISS.from_documents(chunks, _get_embeddings())


def load_vector_store(index_path: str = "faiss_index"):
    """Load a saved FAISS index from disk."""
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
    """
    Return top-k relevant chunks.
    k=3 (down from 5) reduces context tokens per Gemini call.
    """
    return vector_store.similarity_search(query, k=k)


def retrieve_per_document(vector_store, query: str, chunks_per_doc: int = 3):
    """
    Retrieve chunks balanced across all documents.
    chunks_per_doc=3, over-fetch k=30 (down from 50).
    """
    all_results = vector_store.similarity_search(query, k=30)

    doc_chunks = defaultdict(list)
    for doc in all_results:
        source = doc.metadata.get("source", "Unknown")
        doc_chunks[source].append(doc)

    balanced = []
    for source, chunks in doc_chunks.items():
        balanced.extend(chunks[:chunks_per_doc])

    return balanced, list(doc_chunks.keys())


# ══════════════════════════════════════════════════════════════════════════════
#  Query Reformulation
#  DISABLED on free tier — saves 1 full API call per user message.
#  Uncomment the block below to re-enable when on a paid plan.
# ══════════════════════════════════════════════════════════════════════════════

def reformulate_query(query: str, chat_history: list) -> str:
    """Returns query unchanged. Disabled to conserve free-tier RPM quota."""
    return query

    # ── Uncomment below to re-enable on paid plan ─────────────────────────────
    # if len(query.split()) > 8 or not chat_history:
    #     return query
    # recent = chat_history[-4:]
    # history_text = ""
    # for msg in recent:
    #     role = "User" if msg["role"] == "user" else "Assistant"
    #     history_text += f"{role}: {msg['content'][:200]}\n"
    # prompt = (
    #     "Rewrite the question to be fully self-contained for document search.\n"
    #     "Return ONLY the rewritten query, nothing else.\n"
    #     f"History:\n{history_text}\nQuestion: {query}\nRewritten:"
    # )
    # result = _call_gemini(prompt)
    # if result.startswith("⚠️"):
    #     return query
    # rewritten = result.strip().strip('"').strip("'")
    # return rewritten if 5 < len(rewritten) < 500 else query


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
    Generate a grounded answer from retrieved document context.
    Token optimizations:
      - max_history_turns=1 (was 3) — 1 exchange of context is enough
      - history messages truncated to 200 chars (was 400)
      - context chunks already small (600 chars each from chunk_docs)
    """
    context = "\n\n".join([doc.page_content for doc in context_docs])

    if len(context.strip()) < 30:
        return (
            "I could not find enough relevant information in the uploaded documents. "
            "Try rephrasing your question or uploading additional documents."
        )

    # Compact history — only last 1 exchange, 200 chars max each
    history_block = ""
    if chat_history:
        recent = chat_history[-(max_history_turns * 2):]
        if recent:
            history_block = "Recent context (for reference only — not a fact source):\n"
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
        "3. Be concise. No padding.\n"
        "4. Do not use training knowledge.\n\n"
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
    Produce a structured summary for each document.
    Token optimizations:
      - k=20 (was 60) for similarity search
      - Only 5 chunks per doc (was 20)
      - Each chunk already 600 chars from chunk_docs
      - _call_gemini handles pacing (4s sleep built in)
    """
    summaries = {}

    for source in all_source_names:
        all_results = vector_store.similarity_search(
            "summary overview main points key information",
            k=20,
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

        # Stop if rate-limited — no point hammering further
        if "Rate limit" in result or result.startswith("⚠️"):
            for remaining in all_source_names[all_source_names.index(source) + 1:]:
                summaries[remaining] = (
                    "⚠️ Skipped — rate limit reached. "
                    "Wait 60 seconds and try summarising again."
                )
            break

    return summaries
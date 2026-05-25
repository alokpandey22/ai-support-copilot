"""
rag_pipeline.py
───────────────
Core RAG pipeline for AI Support Copilot.
Rate-limit safe: reduced chunk sizes, context limits, pacing delays.
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

# ── Embeddings (lazy-loaded) ───────────────────────────────────────────────────
_embeddings = None

def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings


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
    FIX: Reduced chunk_size 1200→600 and overlap 200→100 to cut TPM by ~50%.
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
    FIX: Reduced k 5→3 to cut context size per Gemini call.
    """
    return vector_store.similarity_search(query, k=k)


def retrieve_per_document(vector_store, query: str, chunks_per_doc: int = 3):
    """
    Retrieve chunks balanced across all documents.
    FIX: Reduced chunks_per_doc 5→3 and over-fetch k 50→30.
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
#  FIX: Disabled entirely — saves 1 full Gemini API call per user message.
#  Re-enable the commented block below when on a paid plan.
# ══════════════════════════════════════════════════════════════════════════════

def reformulate_query(query: str, chat_history: list) -> str:
    """Returns query unchanged. Reformulation disabled to save free-tier quota."""
    return query

    # ── Uncomment to re-enable on paid plan ───────────────────────────────────
    # if len(query.split()) > 8 or not chat_history:
    #     return query
    # recent = chat_history[-6:]
    # history_text = ""
    # for msg in recent:
    #     role = "User" if msg["role"] == "user" else "Assistant"
    #     history_text += f"{role}: {msg['content'][:200]}\n"
    # prompt = (
    #     "Rewrite the question to be fully self-contained for document search.\n"
    #     "Return ONLY the rewritten query, nothing else.\n"
    #     f"History:\n{history_text}\nQuestion: {query}\nRewritten:"
    # )
    # try:
    #     time.sleep(4)
    #     response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    #     rewritten = response.text.strip().strip('"').strip("'")
    #     return rewritten if 5 < len(rewritten) < 500 else query
    # except Exception:
    #     return query


# ══════════════════════════════════════════════════════════════════════════════
#  Answer Generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_answer(
    context_docs: list,
    query: str,
    chat_history: list | None = None,
    max_history_turns: int = 1,    # FIX: Reduced 3→1 to cut tokens per call
) -> str:
    """
    Generate a grounded answer from retrieved document context.
    FIX: max_history_turns reduced 3→1, history truncated to 200 chars each.
    FIX: time.sleep(4) added before API call to respect RPM limits.
    """
    context = "\n\n".join([doc.page_content for doc in context_docs])

    if len(context.strip()) < 30:
        return (
            "I could not find enough relevant information in the uploaded documents. "
            "Try rephrasing your question or uploading additional documents."
        )

    # Build compact history — only last 1 exchange, max 200 chars each
    history_block = ""
    if chat_history:
        recent = chat_history[-(max_history_turns * 2):]
        if recent:
            history_block = "Recent context (for reference only — not a fact source):\n"
            for msg in recent:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_block += f"{role}: {msg['content'][:200]}\n"
            history_block += "\n"

    prompt = f"""You are an enterprise AI support copilot.
Answer using ONLY the Document Context below.

Rules:
1. Only use information from Document Context.
2. If the answer is not in the context say: "I could not find this in the uploaded documents."
3. Be concise. No padding or filler.
4. Do not use training knowledge.

{history_block}Document Context:
{context}

Question: {query}

Answer:"""

    try:
        time.sleep(4)    # FIX: pace requests — stay under free-tier RPM limit
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        return response.text
    except Exception as e:
        err = str(e)
        if "429" in err or "RESOURCE_EXHAUSTED" in err:
            return (
                "⚠️ Rate limit hit. Too many requests in the last 60 seconds. "
                "Wait 60 seconds and try again. "
                "Check your limits at aistudio.google.com."
            )
        return f"⚠️ Error generating answer: {err}"


# ══════════════════════════════════════════════════════════════════════════════
#  Document Summarisation
# ══════════════════════════════════════════════════════════════════════════════

def summarize_all_documents(vector_store, all_source_names: list) -> dict:
    """
    Produce a structured summary for each document.
    FIX: k reduced 60→20, chunk limit [:20]→[:5], time.sleep(4) added.
    """
    summaries = {}

    for source in all_source_names:
        # FIX: k=20 instead of k=60 — massive TPM saver
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

        # FIX: only use 5 chunks instead of 20 — cuts tokens per call ~75%
        context = "\n\n".join([doc.page_content for doc in doc_chunks[:5]])

        prompt = f"""Summarise the document "{source}" using ONLY the context below.

Provide:
1. What this document is about (2 sentences max)
2. Key points (bullet points, max 5)
3. Most important takeaway (1 sentence)

Context:
{context}

Summary:"""

        try:
            time.sleep(4)    # FIX: pace between summarisation calls
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )
            summaries[source] = response.text
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                summaries[source] = "⚠️ Rate limit hit. Wait 60 seconds and try again."
                break
            summaries[source] = f"⚠️ Error: {err}"

    return summaries
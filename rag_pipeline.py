"""
rag_pipeline.py
───────────────
Core RAG (Retrieval-Augmented Generation) pipeline for AI Support Copilot.
Handles document loading, chunking, vector-store creation, retrieval, query
reformulation, and answer generation using Gemini 1.5 Flash.
"""

import os
from collections import defaultdict

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from google import genai
from dotenv import load_dotenv

# ── Load environment variables ─────────────────────────────────────────────────
load_dotenv()

# ── Gemini client (API key loaded from .env or Streamlit secrets) ──────────────
_api_key = os.getenv("GEMINI_API_KEY")

# Fall back to Streamlit secrets when running on Streamlit Cloud
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

# ── Model config ───────────────────────────────────────────────────────────────
MODEL_NAME = "models/gemini-1.5-flash-latest"

# ── Embedding model (downloaded once, cached by sentence-transformers) ─────────
_embeddings = None


def _get_embeddings():
    """Lazy-load embeddings so startup is fast."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings


# ──────────────────────────────────────────────────────────────────────────────
# Document loading & chunking
# ──────────────────────────────────────────────────────────────────────────────

def load_docs(file_path: str):
    """Load a PDF and return a list of LangChain Document objects."""
    loader = PyPDFLoader(file_path)
    return loader.load()


def chunk_docs(docs, chunk_size: int = 1200, chunk_overlap: int = 200):
    """
    Split documents into overlapping chunks for retrieval.

    Args:
        docs:          List of LangChain Document objects.
        chunk_size:    Target characters per chunk.
        chunk_overlap: Characters shared between adjacent chunks (context bridge).

    Returns:
        List of chunked Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


# ──────────────────────────────────────────────────────────────────────────────
# Vector store
# ──────────────────────────────────────────────────────────────────────────────

def create_vector_store(chunks):
    """
    Build a FAISS vector store from a list of document chunks.

    Raises:
        ValueError: If chunks is empty (nothing to embed).
    """
    if not chunks:
        raise ValueError("No text chunks provided — cannot build vector store.")
    embeddings = _get_embeddings()
    return FAISS.from_documents(chunks, embeddings)


def load_vector_store(index_path: str = "faiss_index"):
    """
    Load a previously saved FAISS index from disk.

    Returns:
        FAISS vector store, or None if the index doesn't exist.
    """
    if not os.path.exists(index_path):
        return None
    embeddings = _get_embeddings()
    return FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────────────────────────────────────

def retrieve_docs(vector_store, query: str, k: int = 5):
    """
    Return the top-k most relevant document chunks for a query.

    Args:
        vector_store: FAISS vector store.
        query:        Search query string.
        k:            Number of chunks to return.

    Returns:
        List of LangChain Document objects.
    """
    return vector_store.similarity_search(query, k=k)


def retrieve_per_document(vector_store, query: str, chunks_per_doc: int = 5):
    """
    Retrieve chunks balanced across all documents in the index.
    Useful for broad "summarize everything" queries so one large doc
    doesn't crowd out smaller ones.

    Args:
        vector_store:  FAISS vector store.
        query:         Search query string.
        chunks_per_doc: Max chunks to include per source document.

    Returns:
        Tuple of (balanced_chunks: list, source_names: list[str]).
    """
    # Over-retrieve then balance
    all_results = vector_store.similarity_search(query, k=50)

    doc_chunks: dict[str, list] = defaultdict(list)
    for doc in all_results:
        source = doc.metadata.get("source", "Unknown")
        doc_chunks[source].append(doc)

    balanced = []
    for source, chunks in doc_chunks.items():
        balanced.extend(chunks[:chunks_per_doc])

    return balanced, list(doc_chunks.keys())


# ──────────────────────────────────────────────────────────────────────────────
# Query Reformulation
# ──────────────────────────────────────────────────────────────────────────────

def reformulate_query(query: str, chat_history: list) -> str:
    """
    Rewrite a short or ambiguous follow-up query into a fully self-contained
    question, improving retrieval accuracy.

    Example:
        History: "What is the return policy?"
        Query:   "How long does it take?"
        Output:  "How long does the enterprise return process take?"

    Skips reformulation if:
    - The query is already long (> 8 words).
    - There is no chat history to draw context from.

    Args:
        query:        The raw user query.
        chat_history: List of {"role": str, "content": str} dicts.

    Returns:
        Rewritten query string (falls back to original on any error).
    """
    # Long, explicit queries don't need rewriting
    if len(query.split()) > 8:
        return query

    # Nothing to resolve pronouns against
    if not chat_history:
        return query

    # Build compact history (last 3 exchanges = 6 messages)
    recent = chat_history[-6:]
    history_text = ""
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content'][:300]}\n"

    prompt = f"""You are a search query optimizer.

Given the conversation history and the current user question, rewrite the \
question to be fully self-contained and specific — as if there were no \
conversation history. Optimize for searching a document database.

Rules:
- Return ONLY the rewritten query, nothing else
- No explanations, no quotes, no preamble
- If the question is already clear and self-contained, return it unchanged
- Resolve pronouns (it, they, this, that) using context from history
- Expand vague time/duration questions using context

Conversation History:
{history_text}

Current Question: {query}

Rewritten Query:"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        rewritten = response.text.strip().strip('"').strip("'")
        if 5 < len(rewritten) < 500:
            return rewritten
        return query
    except Exception:
        return query  # Always fall back gracefully


# ──────────────────────────────────────────────────────────────────────────────
# Answer Generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_answer(
    context_docs: list,
    query: str,
    chat_history: list | None = None,
    max_history_turns: int = 3,
) -> str:
    """
    Generate a grounded answer using retrieved document context.

    Args:
        context_docs:       Retrieved LangChain Document objects.
        query:              Current user question (already reformulated).
        chat_history:       Optional conversation history for follow-up awareness.
        max_history_turns:  How many past exchange pairs to include.

    Returns:
        Answer string from Gemini, or a descriptive error/fallback message.
    """
    context = "\n\n".join([doc.page_content for doc in context_docs])

    if len(context.strip()) < 30:
        return (
            "I could not find enough relevant information in the uploaded documents. "
            "Try rephrasing your question or uploading additional documents."
        )

    # ── Build conversation history block ──────────────────────────────────────
    history_block = ""
    if chat_history:
        recent_messages = chat_history[-(max_history_turns * 2):]
        if recent_messages:
            history_block = (
                "Conversation History "
                "(use ONLY to understand context — NOT as a fact source):\n"
            )
            for msg in recent_messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_block += f"{role}: {msg['content'][:400]}\n"
            history_block += "\n"

    prompt = f"""You are an enterprise AI support copilot.

Use ONLY the provided Document Context below to answer the question.

STRICT RULES:
1. Use ONLY information from the Document Context section.
2. You may use the Conversation History to understand context (resolve pronouns,
   understand follow-ups), but NEVER use it as a source of facts.
3. Do NOT use any knowledge from your training data.
4. Do NOT guess or make assumptions.
5. If the exact answer is not in the context, say:
   "I could not find this information in the uploaded documents."
6. Keep answers concise and accurate.
7. When relevant, indicate which part of the document supports your answer.

{history_block}Document Context:
{context}

Current Question:
{query}

Answer:"""

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
                "⚠️ API rate limit reached. You have used your daily free quota. "
                "Please wait 24 hours or upgrade your plan at aistudio.google.com."
            )
        return f"⚠️ Error generating answer: {err}"


# ──────────────────────────────────────────────────────────────────────────────
# Document Summarisation
# ──────────────────────────────────────────────────────────────────────────────

def summarize_all_documents(vector_store, all_source_names: list) -> dict:
    """
    Produce a structured summary for each document in the index.

    Args:
        vector_store:     FAISS vector store.
        all_source_names: List of document names (as stored in chunk metadata).

    Returns:
        Dict mapping source_name → summary_string.
    """
    summaries: dict[str, str] = {}

    for source in all_source_names:
        # Fetch broadly then filter to this doc
        all_results = vector_store.similarity_search(
            "summary overview main points key information",
            k=60,
        )
        doc_chunks = [
            doc for doc in all_results
            if doc.metadata.get("source") == source
        ]

        if not doc_chunks:
            summaries[source] = "Could not retrieve content from this document."
            continue

        context = "\n\n".join([doc.page_content for doc in doc_chunks[:20]])

        prompt = f"""You are summarising a document called "{source}".

Using ONLY the context below, provide:
1. What this document is about (2–3 sentences)
2. Key points (bullet points)
3. Most important information the reader should know

Context:
{context}

Summary:"""

        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )
            summaries[source] = response.text
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                summaries[source] = (
                    "⚠️ Rate limit reached. Please wait and try again."
                )
                break  # Stop processing further docs
            summaries[source] = f"⚠️ Error: {err}"

    return summaries
"""
app.py — AI Support Copilot
Persistence fix: documents survive page refresh.

How persistence works:
- On upload, each user gets a unique session_id stored in a browser cookie
  (via st.query_params as a fallback, or URL param).
- Chunks are pickled to  ./session_data/<session_id>/<docname>.pkl
- On every page load, we detect the session_id and reload any saved chunks.
- Privacy: each session_id is a UUID — other users cannot guess or access
  another session's folder.
- Documents persist until the user explicitly clears them (or the server
  admin purges ./session_data/).
"""

import datetime, glob, os, pickle, shutil, uuid
import streamlit as st

from rag_pipeline import (
    load_docs, chunk_docs, create_vector_store,
    retrieve_docs, retrieve_per_document,
    reformulate_query, generate_answer,
)

# ── Startup cleanup of stray temp PDFs ──────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _startup_cleanup():
    for f in glob.glob("*.pdf"):
        if "-" in f:
            try: os.remove(f)
            except Exception: pass
    return True
_startup_cleanup()

os.makedirs("session_data", exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CopilotAI — Enterprise Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600&family=Instrument+Sans:wght@400;500&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
  --base:#0C0D10;--elevated:#131519;--surface:#191C22;--surface-2:#1F2229;
  --border:rgba(255,255,255,0.06);--border-md:rgba(255,255,255,0.11);
  --text-1:#EAE8DF;--text-2:#797772;
  --gold:#C9A96E;--gold-bright:#E2C090;--gold-dim:rgba(201,169,110,0.11);
  --gold-bg:rgba(201,169,110,0.06);--gold-border:rgba(201,169,110,0.25);
  --emerald:#3D9B78;--emerald-dim:rgba(61,155,120,0.13);--emerald-border:rgba(61,155,120,0.28);
  --red:#C85555;--red-dim:rgba(200,85,85,0.12);
  --f-brand:'Cormorant Garamond',Georgia,serif;
  --f-ui:'Instrument Sans',system-ui,sans-serif;
  --f-mono:'JetBrains Mono',monospace;
}
*,*::before,*::after{box-sizing:border-box}
html,body,[data-testid="stApp"],[data-testid="stAppViewContainer"]{
  background:var(--base)!important;color:var(--text-1)!important;
  font-family:var(--f-ui)!important;-webkit-font-smoothing:antialiased}
[data-testid="stHeader"]{background:transparent!important}
[data-testid="stDecoration"],#MainMenu,footer{display:none!important}
[data-testid="stSidebarCollapsedControl"]{display:flex!important;position:fixed!important;top:12px!important;left:12px!important;z-index:999!important}
[data-testid="stSidebarCollapsedControl"] button{background:var(--surface)!important;border:1px solid var(--gold-border)!important;border-radius:8px!important;color:var(--gold)!important;width:34px!important;height:34px!important}
[data-testid="stSidebar"]{background:var(--elevated)!important;border-right:1px solid var(--border)!important}
button[kind="header"]{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:8px!important;color:var(--gold)!important;width:34px!important;height:34px!important;transition:all 0.18s ease!important}
button[kind="header"]:hover{border-color:var(--gold-border)!important;background:var(--gold-bg)!important}
button[kind="header"] svg{fill:var(--gold)!important}
[data-testid="stSidebar"] section[data-testid="stSidebarContent"]{padding:1rem 1rem 2rem!important;gap:0.5rem!important}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label,[data-testid="stSidebar"] small,[data-testid="stSidebar"] .stMarkdown p{font-family:var(--f-mono)!important;font-size:11px!important;color:var(--text-2)!important;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
[data-testid="stSidebar"] .stMarkdown span,[data-testid="stSidebar"] [data-testid="stMetricLabel"] span,[data-testid="stSidebar"] [data-testid="stMetricValue"] span{font-family:var(--f-mono)!important;font-size:11px!important;color:var(--text-2)!important}
[data-testid="stSidebar"] details>summary{list-style:none!important}
[data-testid="stSidebar"] [data-testid="stExpanderToggleIcon"]{display:none!important}
[data-testid="stSidebar"] .stFileUploader button span{display:none!important}
[data-testid="stSidebar"] .stFileUploader button::after{content:"↑ Upload PDFs";font-family:var(--f-mono)!important;font-size:11px!important;color:var(--gold)!important;white-space:nowrap;letter-spacing:0.04em}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p{font-size:9.5px!important;letter-spacing:0.14em;text-transform:uppercase;color:rgba(255,255,255,0.22)!important;margin-bottom:4px!important;white-space:normal!important}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"],[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] *,[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]>div{display:none!important}
[data-testid="stSidebar"] [data-testid="stMetric"]{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:10px 12px!important}
[data-testid="stSidebar"] [data-testid="stMetricValue"]{font-family:var(--f-brand)!important;font-size:24px!important;color:var(--text-1)!important}
[data-testid="stSidebar"] [data-testid="stMetricLabel"] p{font-size:9px!important;text-transform:uppercase;letter-spacing:0.1em;color:rgba(255,255,255,0.2)!important;white-space:normal!important}
[data-testid="stSidebar"] .stCheckbox{background:var(--surface);border:1px solid var(--border);border-radius:7px;padding:6px 10px!important;margin-bottom:4px!important;transition:border-color 0.15s,background 0.15s}
[data-testid="stSidebar"] .stCheckbox:has(input:checked){background:var(--gold-dim);border-color:var(--gold-border)}
[data-testid="stSidebar"] .stCheckbox label{font-family:var(--f-mono)!important;font-size:11px!important;color:var(--text-2)!important;cursor:pointer;white-space:nowrap!important;overflow:hidden!important;text-overflow:ellipsis!important;max-width:180px;display:inline-block}
[data-testid="stSidebar"] .stCheckbox:has(input:checked) label{color:var(--gold)!important}
[data-testid="stSidebar"] .stCheckbox input[type="checkbox"]{accent-color:var(--gold)!important}
[data-testid="stSidebar"] .stButton>button{width:100%!important;background:transparent!important;border:1px solid var(--border-md)!important;border-radius:7px!important;color:var(--text-2)!important;font-family:var(--f-mono)!important;font-size:11px!important;padding:5px 10px!important;transition:all 0.18s!important}
[data-testid="stSidebar"] .stButton>button:hover{border-color:var(--gold)!important;color:var(--gold)!important;background:var(--gold-bg)!important}
[data-testid="stSidebar"] .stFileUploader>div{background:var(--gold-bg)!important;border:1px dashed var(--gold-border)!important;border-radius:8px!important;padding:8px 12px!important}
[data-testid="stSidebar"] .stFileUploader label{color:var(--gold)!important}
[data-testid="stSidebar"] .stDownloadButton>button{width:100%!important;background:transparent!important;border:1px solid var(--border-md)!important;border-radius:8px!important;color:var(--text-2)!important;font-family:var(--f-mono)!important;font-size:11px!important;padding:8px 14px!important;transition:all 0.2s!important}
[data-testid="stSidebar"] .stDownloadButton>button:hover{border-color:var(--gold)!important;color:var(--gold)!important;background:var(--gold-bg)!important}
[data-testid="stSidebar"] .stExpander{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:8px!important}
[data-testid="stSidebar"] hr{border-color:var(--border)!important;margin:8px 0!important}
.sb-status{display:inline-flex;align-items:center;gap:6px;font-size:10px;font-family:var(--f-mono);color:var(--emerald);background:var(--emerald-dim);border:1px solid var(--emerald-border);border-radius:100px;padding:4px 10px;margin-bottom:12px;width:fit-content}
.sb-pulse{width:5px;height:5px;border-radius:50%;background:var(--emerald);display:inline-block;animation:hb 2.4s ease-in-out infinite}
@keyframes hb{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.3;transform:scale(.8)}}
.scope-pill{display:inline-flex;align-items:center;gap:5px;font-size:10px;font-family:var(--f-mono);background:var(--gold-bg);border:1px solid var(--gold-border);color:var(--gold);border-radius:100px;padding:3px 10px}
.sb-quality{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px;display:grid;grid-template-columns:repeat(3,1fr);width:100%}
.sb-qm{text-align:center;position:relative}
.sb-qm:not(:last-child)::after{content:'';position:absolute;right:0;top:50%;transform:translateY(-50%);width:1px;height:55%;background:var(--border)}
.sb-qm-num{font-family:var(--f-brand)!important;font-size:20px!important;font-weight:600;line-height:1.1}
.sb-qm-lbl{font-size:8.5px!important;letter-spacing:0.1em;text-transform:uppercase;color:rgba(255,255,255,0.2)!important;font-family:var(--f-mono)!important;margin-top:2px}
[data-testid="stMain"]{background:var(--base)!important}
.block-container{padding:0!important;max-width:100%!important}
.topbar{height:56px;border-bottom:1px solid var(--border);display:flex;align-items:center;padding:0 40px;gap:12px;background:var(--base);flex-wrap:wrap}
.tb-crumb{font-size:12px;color:var(--text-2);font-family:var(--f-mono)}
.tb-sep{width:1px;height:14px;background:var(--border);flex-shrink:0}
.chat-pad{padding:32px 40px 12px;max-width:820px;margin:0 auto}
.date-sep{display:flex;align-items:center;gap:12px;margin-bottom:24px}
.ds-line{flex:1;height:1px;background:var(--border)}
.ds-text{font-size:9.5px;font-family:var(--f-mono);letter-spacing:.1em;color:rgba(255,255,255,0.17);text-transform:uppercase;white-space:nowrap}
.msg-u{display:flex;justify-content:flex-end;margin-bottom:20px}
.u-bubble{background:var(--surface);border:1px solid var(--border-md);border-radius:14px;border-bottom-right-radius:3px;padding:11px 16px;max-width:68%;font-size:14px;color:var(--text-1);line-height:1.7;word-break:break-word}
.msg-a{margin-bottom:4px}
.a-row{display:flex;gap:11px;align-items:flex-start}
.a-gem{width:30px;height:30px;border-radius:8px;flex-shrink:0;background:var(--gold-dim);border:1px solid var(--gold-border);display:flex;align-items:center;justify-content:center;margin-top:2px}
.a-gem svg{width:13px;height:13px;stroke:var(--gold);fill:none;stroke-linecap:round;stroke-linejoin:round;stroke-width:1.6}
.a-bubble{background:var(--elevated);border:1px solid var(--border);border-radius:14px;border-top-left-radius:3px;padding:12px 18px;font-size:14px;color:var(--text-1);line-height:1.75;word-break:break-word}
.a-meta{display:flex;align-items:center;gap:7px;margin-top:8px;flex-wrap:wrap;padding-left:41px}
.chip{display:inline-flex;align-items:center;gap:4px;font-family:var(--f-mono);font-size:9.5px;color:rgba(255,255,255,0.28);background:var(--surface);border:1px solid var(--border);border-radius:5px;padding:2px 8px}
.chip-dot{width:4px;height:4px;border-radius:50%;background:var(--emerald);flex-shrink:0}
.conf-wrap{display:flex;align-items:center;gap:6px}
.conf-track{width:60px;height:3px;background:var(--surface-2);border-radius:3px;overflow:hidden}
.conf-fill{height:100%;background:var(--gold);border-radius:3px}
.conf-text{font-size:9.5px;font-family:var(--f-mono);color:rgba(255,255,255,0.2)}
.fb-rated{font-size:9.5px;font-family:var(--f-mono);color:rgba(255,255,255,0.2);padding-left:41px;margin-top:3px}
[data-testid="stMain"] .stButton>button{background:transparent!important;border:1px solid var(--border)!important;border-radius:6px!important;color:rgba(255,255,255,0.3)!important;font-size:13px!important;padding:2px 8px!important;min-width:32px!important;transition:all 0.15s!important}
[data-testid="stMain"] .stButton>button:hover{border-color:var(--border-md)!important;color:var(--text-1)!important}
.input-hint{text-align:center;font-size:10px;font-family:var(--f-mono);color:rgba(255,255,255,0.13);letter-spacing:.04em;margin:6px 0 2px}
[data-testid="stChatInputContainer"]{background:var(--elevated)!important;border:1px solid var(--border-md)!important;border-radius:13px!important;max-width:820px!important;margin:0 auto!important}
[data-testid="stChatInputContainer"]:focus-within{border-color:var(--gold-border)!important}
[data-testid="stChatInputContainer"] textarea{background:transparent!important;color:var(--text-1)!important;font-family:var(--f-ui)!important;font-size:14px!important}
[data-testid="stChatInputContainer"] textarea::placeholder{color:rgba(255,255,255,0.18)!important}
[data-testid="stChatInputContainer"] button{background:var(--gold)!important;border-radius:8px!important;border:none!important}
[data-testid="stChatInputContainer"] button:hover{background:var(--gold-bright)!important}
[data-testid="stChatMessage"]{background:transparent!important;border:none!important;padding:0!important}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"]{display:none!important}
[data-testid="stAlert"],.stSuccess,.stWarning,.stError,.stInfo{background:var(--surface)!important;border:1px solid var(--border-md)!important;border-radius:8px!important;font-family:var(--f-mono)!important;font-size:11px!important}
[data-testid="stExpander"]{background:var(--elevated)!important;border:1px solid var(--border)!important;border-radius:8px!important}
.ob-card{background:linear-gradient(145deg,#131519 0%,#181b22 100%);border:1px solid var(--gold-border);border-radius:14px;padding:2.4rem 2rem;text-align:center;max-width:580px;margin:3rem auto;position:relative;overflow:hidden}
.ob-card::before{content:'';position:absolute;inset:0;background:radial-gradient(ellipse 70% 40% at 50% 0%,rgba(201,169,110,.07) 0%,transparent 70%);pointer-events:none}
.ob-title{font-family:var(--f-brand)!important;font-size:1.85rem;color:var(--text-1);margin-bottom:.5rem;font-weight:600}
.ob-sub{color:var(--text-2);font-size:.9rem;line-height:1.6;margin:.35rem 0}
.ob-sub strong{color:var(--gold);font-weight:500}
.ob-tiles{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;max-width:660px;margin:20px auto 0}
.ob-tile{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 12px}
.ob-tile-icon{font-size:1.3rem;margin-bottom:6px}
.ob-tile-title{font-family:var(--f-brand);font-size:.95rem;font-weight:600;color:var(--text-1);margin-bottom:3px}
.ob-tile-desc{font-size:10.5px;color:var(--text-2);line-height:1.45}
*{scrollbar-width:thin;scrollbar-color:var(--border-md) transparent}
::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-thumb{background:var(--border-md);border-radius:3px}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  SESSION ID — persists in URL query param so refresh restores the same data
#  ?sid=<uuid>  is appended once and stays in the browser URL bar.
# ════════════════════════════════════════════════════════════════════════════

def _get_or_create_session_id() -> str:
    """
    Return a stable session ID for this browser tab.
    - First visit:  generate a UUID, write to URL param, store in session_state.
    - Refresh:      URL param survives → same ID → same data folder reloaded.
    - New tab:      no param → new UUID → separate data folder.
    """
    params = st.query_params
    if "sid" in params:
        sid = params["sid"]
        # Validate it looks like a UUID (basic safety check)
        if len(sid) == 36 and sid.count("-") == 4:
            st.session_state["_sid"] = sid
            return sid

    # No valid param → create new ID and push it to the URL
    if "_sid" not in st.session_state:
        st.session_state["_sid"] = str(uuid.uuid4())
    st.query_params["sid"] = st.session_state["_sid"]
    return st.session_state["_sid"]


SESSION_ID   = _get_or_create_session_id()
SESSION_DIR  = os.path.join("session_data", SESSION_ID)
CHUNKS_DIR   = os.path.join(SESSION_DIR, "chunks")
META_FILE    = os.path.join(SESSION_DIR, "meta.pkl")

os.makedirs(CHUNKS_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
#  PERSISTENCE HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _save_doc_chunks(doc_name: str, chunks: list):
    """Pickle one document's chunks to disk under this session's folder."""
    safe = doc_name.replace("/", "_").replace("\\", "_")
    path = os.path.join(CHUNKS_DIR, safe + ".pkl")
    with open(path, "wb") as f:
        pickle.dump(chunks, f)


def _load_all_chunks_from_disk() -> dict:
    """Reload all pickled chunk files for this session from disk."""
    dc = {}
    for pkl_path in sorted(glob.glob(os.path.join(CHUNKS_DIR, "*.pkl"))):
        try:
            with open(pkl_path, "rb") as f:
                chunks = pickle.load(f)
            if chunks:
                # Recover original doc name from chunk metadata
                doc_name = chunks[0].metadata.get("source", os.path.basename(pkl_path))
                dc[doc_name] = chunks
        except Exception:
            pass
    return dc


def _delete_doc_from_disk(doc_name: str):
    safe = doc_name.replace("/", "_").replace("\\", "_")
    path = os.path.join(CHUNKS_DIR, safe + ".pkl")
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _save_meta(source_names: list, selected_docs: list, messages: list):
    """Persist lightweight metadata (names, selections, chat history)."""
    with open(META_FILE, "wb") as f:
        pickle.dump({
            "source_names":  source_names,
            "selected_docs": selected_docs,
            "messages":      messages,
        }, f)


def _load_meta() -> dict:
    if not os.path.exists(META_FILE):
        return {}
    try:
        with open(META_FILE, "rb") as f:
            return pickle.load(f)
    except Exception:
        return {}


# ════════════════════════════════════════════════════════════════════════════
#  SESSION STATE + ON-LOAD RESTORE
#  On every page load (including refresh) we reload chunks from disk and
#  rebuild the in-memory vector store for this session.
# ════════════════════════════════════════════════════════════════════════════

def _init_session():
    """
    Initialise session_state. If this is a refresh (sid already in URL),
    reload persisted data from disk and rebuild the vector store.
    Called once per page load via st.cache_resource keyed on SESSION_ID.
    """
    if "session_initialised" in st.session_state:
        return  # already done for this session

    # Try to reload from disk
    dc   = _load_all_chunks_from_disk()
    meta = _load_meta()

    st.session_state["doc_chunks"]    = dc
    st.session_state["source_names"]  = meta.get("source_names",  list(dc.keys()))
    st.session_state["selected_docs"] = meta.get("selected_docs", list(dc.keys()))
    st.session_state["messages"]      = meta.get("messages",      [])
    st.session_state["query_log"]     = []
    st.session_state["uploaded_count"]= len(dc)

    # Rebuild FAISS from reloaded chunks (fast — no API call, just embedding)
    if dc:
        all_chunks = [c for cs in dc.values() for c in cs]
        st.session_state["vector_store"] = create_vector_store(all_chunks)
    else:
        st.session_state["vector_store"] = None

    st.session_state["session_initialised"] = True

_init_session()


# ════════════════════════════════════════════════════════════════════════════
#  VECTOR STORE HELPERS
# ════════════════════════════════════════════════════════════════════════════

def rebuild_vs(dc: dict):
    chunks = [c for cs in dc.values() for c in cs]
    return create_vector_store(chunks) if chunks else None


def build_scoped_vs(selected: list, doc_chunks: dict):
    chunks = [c for name in selected for c in doc_chunks.get(name, [])]
    return create_vector_store(chunks) if chunks else None


def export_txt(msgs):
    lines = ["="*56, "  CopilotAI — Conversation Export",
             f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", "="*56, ""]
    for m in msgs:
        if m["role"] == "user":
            lines.append(f"USER:  {m['content']}")
        else:
            lines.append(f"COPILOT:  {m['content']}")
            if "sources" in m:    lines.append(f"Sources:    {' | '.join(m['sources'])}")
            if "confidence" in m: lines.append(f"Confidence: {m['confidence']}%")
            if m.get("feedback"): lines.append(f"Feedback:   {m['feedback']}")
        lines.append("")
    return "\n".join(lines)


GEM = ('<svg viewBox="0 0 24 24" fill="none" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">'
       '<path d="M12 2 2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5M2 12l10 5 10-5"/></svg>')

def truncate(s, n=22):
    return s[:n] + "…" if len(s) > n else s


# ════════════════════════════════════════════════════════════════════════════
#  UPLOAD WIDGET (rendered first so processing can happen before sidebar)
#  We use a numeric key suffix stored in session_state. After every successful
#  upload we increment it, which forces Streamlit to mount a brand-new empty
#  widget on the next rerun — so the uploader is always visible and ready.
# ════════════════════════════════════════════════════════════════════════════
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

with st.sidebar:
    _uploaded_files = st.file_uploader(
        "", type="pdf", accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"file_uploader_{st.session_state['uploader_key']}",
    )

# ── Process new uploads ──────────────────────────────────────────────────────
if _uploaded_files:
    _any_new = False
    _temps   = []

    for uf in _uploaded_files:
        if uf.name in st.session_state.source_names:
            continue

        _tmp = f"{uuid.uuid4()}_{uf.name}"
        _temps.append(_tmp)
        with open(_tmp, "wb") as fh:
            fh.write(uf.read())

        _docs   = load_docs(_tmp)
        _chunks = chunk_docs(_docs)

        if not _chunks:
            st.warning(f"⚠️ Skipped '{uf.name}' — no extractable text.")
            continue

        for c in _chunks:
            c.metadata["source"] = uf.name

        # Store in session AND persist to disk
        st.session_state.doc_chunks[uf.name]  = _chunks
        st.session_state.source_names.append(uf.name)
        if uf.name not in st.session_state.selected_docs:
            st.session_state.selected_docs.append(uf.name)

        _save_doc_chunks(uf.name, _chunks)   # ← disk persistence
        _any_new = True

    for _t in _temps:
        try: os.remove(_t)
        except Exception: pass

    if _any_new and st.session_state.doc_chunks:
        st.session_state.vector_store    = rebuild_vs(st.session_state.doc_chunks)
        st.session_state.uploaded_count  = len(st.session_state.source_names)
        st.session_state.messages        = []
        _save_meta(                            # ← persist metadata
            st.session_state.source_names,
            st.session_state.selected_docs,
            st.session_state.messages,
        )
        # Increment key → Streamlit mounts a fresh empty uploader on next rerun
        st.session_state["uploader_key"] += 1
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════
#  RESOLVE ACTIVE VECTOR STORE
# ════════════════════════════════════════════════════════════════════════════
_sel       = st.session_state.selected_docs
_all_names = st.session_state.source_names

if _sel and set(_sel) == set(_all_names):
    active_vs    = st.session_state.vector_store
    active_names = _all_names
elif _sel:
    active_vs    = build_scoped_vs(_sel, st.session_state.doc_chunks)
    active_names = _sel
else:
    active_vs    = None
    active_names = []


# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    n_docs    = len(st.session_state.source_names)
    n_queries = len([m for m in st.session_state.messages if m["role"] == "user"])
    sel       = st.session_state.selected_docs

    st.markdown(f"""
    <div style="padding:20px 0 16px">
      <div style="display:flex;align-items:center;gap:10px">
        <div style="width:32px;height:32px;border-radius:8px;background:var(--gold-dim);
             border:1px solid var(--gold-border);display:flex;align-items:center;
             justify-content:center;flex-shrink:0">{GEM}</div>
        <div>
          <div style="font-family:var(--f-brand);font-size:20px;font-weight:600;
               color:var(--text-1);line-height:1">
            Copilot<span style="color:var(--gold)">AI</span>
          </div>
          <div style="font-size:9px;letter-spacing:0.14em;text-transform:uppercase;
               color:rgba(255,255,255,0.2);font-family:var(--f-mono);margin-top:2px">
            Enterprise Intelligence
          </div>
        </div>
      </div>
      <div class="sb-status" style="margin-top:12px">
        <span class="sb-pulse"></span>
        {n_docs} doc{'s' if n_docs!=1 else ''} indexed · {len(sel)} active
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.caption("Upload Documents")
    st.markdown(
        '<p style="font-size:9px;color:rgba(255,255,255,0.15);font-family:var(--f-mono);'
        'margin-top:-6px">Docs persist across refresh. Private to your session.</p>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    c1.metric("Documents", n_docs)
    c2.metric("Queries",   n_queries)

    st.divider()

    if st.session_state.source_names:
        st.caption("Active Document Scope")
        new_sel = list(st.session_state.selected_docs)

        for name in st.session_state.source_names:
            label   = (name[:26] + "…") if len(name) > 27 else name
            checked = st.checkbox(label, value=(name in new_sel), key=f"cb_{name}")
            if checked and name not in new_sel:
                new_sel.append(name)
            elif not checked and name in new_sel:
                new_sel.remove(name)

        if sorted(new_sel) != sorted(st.session_state.selected_docs):
            st.session_state.selected_docs = new_sel
            _save_meta(st.session_state.source_names, new_sel, st.session_state.messages)
            st.rerun()

        st.write("")

        def _select_all():
            st.session_state.selected_docs = list(st.session_state.source_names)
            for _d in st.session_state.source_names:
                st.session_state[f"cb_{_d}"] = True
            _save_meta(st.session_state.source_names, st.session_state.selected_docs, st.session_state.messages)

        def _clear_all():
            st.session_state.selected_docs = []
            for _d in st.session_state.source_names:
                st.session_state[f"cb_{_d}"] = False
            _save_meta(st.session_state.source_names, [], st.session_state.messages)

        _qa, _qb = st.columns(2)
        with _qa: st.button("☑ All",  key="sel_all",  on_click=_select_all)
        with _qb: st.button("☐ None", key="sel_none", on_click=_clear_all)

    st.divider()

    st.caption("Response Quality")
    helpful   = sum(1 for m in st.session_state.messages if m.get("feedback") == "helpful")
    unhelpful = sum(1 for m in st.session_state.messages if m.get("feedback") == "unhelpful")
    total_fb  = helpful + unhelpful
    score_str = f"{int(helpful/total_fb*100)}%" if total_fb else "—"
    st.markdown(f"""
    <div class="sb-quality">
      <div class="sb-qm"><div class="sb-qm-num" style="color:var(--emerald)">{helpful}</div><div class="sb-qm-lbl">Helpful</div></div>
      <div class="sb-qm"><div class="sb-qm-num" style="color:var(--red)">{unhelpful}</div><div class="sb-qm-lbl">Flagged</div></div>
      <div class="sb-qm"><div class="sb-qm-num" style="color:var(--gold)">{score_str}</div><div class="sb-qm-lbl">Score</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    if st.session_state.source_names:
        with st.expander("🗂️ Manage Documents"):
            for name in list(st.session_state.source_names):
                _c1, _c2 = st.columns([5, 1])
                _c1.caption(truncate(name, 20))
                if _c2.button("✕", key=f"rm_{name}"):
                    st.session_state.doc_chunks.pop(name, None)
                    st.session_state.source_names.remove(name)
                    st.session_state.selected_docs = [
                        d for d in st.session_state.selected_docs if d != name
                    ]
                    _delete_doc_from_disk(name)    # ← remove from disk too
                    st.session_state.vector_store = (
                        rebuild_vs(st.session_state.doc_chunks)
                        if st.session_state.doc_chunks else None
                    )
                    _save_meta(st.session_state.source_names,
                               st.session_state.selected_docs,
                               st.session_state.messages)
                    st.rerun()

            st.divider()
            if st.button("🗑️ Clear All Documents"):
                # Wipe disk folder for this session
                shutil.rmtree(SESSION_DIR, ignore_errors=True)
                os.makedirs(CHUNKS_DIR, exist_ok=True)
                st.session_state.update({
                    "vector_store": None, "doc_chunks": {}, "source_names": [],
                    "messages": [], "uploaded_count": 0, "selected_docs": [],
                })
                _save_meta([], [], [])
                st.rerun()

    if st.session_state.messages:
        _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        st.download_button(
            "⬇ Download conversation",
            data=export_txt(st.session_state.messages).encode(),
            file_name=f"copilot_{_ts}.txt",
            mime="text/plain",
        )
    else:
        st.caption("No conversation to export yet.")


# ════════════════════════════════════════════════════════════════════════════
#  TOPBAR
# ════════════════════════════════════════════════════════════════════════════
n_docs = len(st.session_state.source_names)
if _sel:
    scope_html = (
        f'<span class="scope-pill">🎯 {len(_sel)} of {n_docs} '
        f'doc{"s" if n_docs!=1 else ""} active</span>'
    )
else:
    scope_html = (
        '<span style="font-size:11px;font-family:var(--f-mono);'
        'color:var(--red);opacity:.7">⚠ No documents selected</span>'
    )

st.markdown(f"""
<div class="topbar">
  <span class="tb-crumb">Conversations</span>
  <div class="tb-sep"></div>
  {scope_html}
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  ONBOARDING
# ════════════════════════════════════════════════════════════════════════════
if not st.session_state.source_names:
    st.markdown("""
    <div class="ob-card">
      <div class="ob-title">👋 Welcome to CopilotAI</div>
      <p class="ob-sub">Upload company PDFs using the uploader in the sidebar.</p>
      <p class="ob-sub">Every answer is grounded exclusively in your documents —<br>no hallucinations, no guessing.</p>
      <br>
      <p class="ob-sub">📎 <strong>Tip:</strong> Upload multiple PDFs at once.</p>
      <p class="ob-sub">🔒 <strong>Private &amp; persistent</strong> — your documents survive page refresh and are never visible to other users.</p>
    </div>
    <div class="ob-tiles">
      <div class="ob-tile"><div class="ob-tile-icon">🔍</div><div class="ob-tile-title">Ask questions</div><div class="ob-tile-desc">Instant grounded answers from your documents</div></div>
      <div class="ob-tile"><div class="ob-tile-icon">📄</div><div class="ob-tile-title">Summarize</div><div class="ob-tile-desc">Structured summaries of any document</div></div>
      <div class="ob-tile"><div class="ob-tile-icon">🗂️</div><div class="ob-tile-title">Manage library</div><div class="ob-tile-desc">Add or remove PDFs without restarting</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if not _sel:
    st.info("⚠️ No documents selected. Use the sidebar checkboxes to choose which documents to query.")
    st.stop()


# ════════════════════════════════════════════════════════════════════════════
#  CONVERSATION HISTORY
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.messages:
    today     = datetime.datetime.now().strftime("%b %d")
    scope_ctx = f"{len(_sel)} doc{'s' if len(_sel)!=1 else ''} in scope"
    st.markdown(f"""
    <div class="chat-pad">
    <div class="date-sep">
      <div class="ds-line"></div>
      <div class="ds-text">Today {today} · {scope_ctx}</div>
      <div class="ds-line"></div>
    </div>""", unsafe_allow_html=True)

    for i, msg in enumerate(st.session_state.messages):
        content      = msg["content"].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        content_html = content.replace("\n\n","</p><p>").replace("\n","<br>")

        if msg["role"] == "user":
            st.markdown(
                f'<div class="msg-u"><div class="u-bubble">{content_html}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            chips = "".join(
                f'<span class="chip"><span class="chip-dot"></span>{s}</span>'
                for s in msg.get("sources", [])
            )
            conf      = msg.get("confidence", 0)
            cc        = "var(--emerald)" if conf == 100 else "var(--gold)"
            conf_html = (
                f'<div class="conf-wrap">'
                f'<div class="conf-track"><div class="conf-fill" '
                f'style="width:{conf}%;background:{cc}"></div></div>'
                f'<span class="conf-text">{conf}% conf.</span></div>'
            )
            st.markdown(f"""
            <div class="msg-a">
              <div class="a-row">
                <div class="a-gem">{GEM}</div>
                <div class="a-bubble"><p>{content_html}</p></div>
              </div>
              <div class="a-meta">{chips}{conf_html}</div>
            </div>""", unsafe_allow_html=True)

            fb = msg.get("feedback")
            if fb:
                icon = "👍" if fb == "helpful" else "👎"
                st.markdown(f'<div class="fb-rated">{icon} You rated this response</div>', unsafe_allow_html=True)
            else:
                fc1, fc2, fc3 = st.columns([0.05, 0.05, 0.9])
                with fc1:
                    if st.button("👍", key=f"up_{i}"):
                        st.session_state.messages[i]["feedback"] = "helpful"
                        _save_meta(st.session_state.source_names, st.session_state.selected_docs, st.session_state.messages)
                        st.rerun()
                with fc2:
                    if st.button("👎", key=f"dn_{i}"):
                        st.session_state.messages[i]["feedback"] = "unhelpful"
                        _save_meta(st.session_state.source_names, st.session_state.selected_docs, st.session_state.messages)
                        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  CHAT INPUT
# ════════════════════════════════════════════════════════════════════════════
sel_str = ", ".join(truncate(d, 18) for d in _sel[:3])
if len(_sel) > 3:
    sel_str += f" +{len(_sel)-3} more"

st.markdown(f'<div class="input-hint">Searching: {sel_str}</div>', unsafe_allow_html=True)

query = st.chat_input("Ask anything about your documents…")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Thinking…"):
        hist = st.session_state.messages[:-1]
        ref  = reformulate_query(query, hist)
        ql   = query.lower()

        is_multi_doc_query = any(w in ql for w in [
            "all documents","every document","each document","all docs",
            "summarize all","summary of all","one by one","across all",
            "compare","both documents","both docs","all files",
        ])
        is_broad_query = any(w in ql for w in [
            "summarize","summary","overview","everything","tell me about",
            "what is this","what does this","key points","main points","highlights",
        ])

        if is_multi_doc_query and len(active_names) > 1:
            ctx_docs, srcs = retrieve_per_document(active_vs, ref, chunks_per_doc=4)
            answer = generate_answer(ctx_docs, ref, hist)
            conf   = 100
        elif is_broad_query:
            ctx_docs, srcs = retrieve_per_document(active_vs, ref, chunks_per_doc=4)
            answer = generate_answer(ctx_docs, ref, hist)
            conf   = min(len(ctx_docs) * 10, 100)
        else:
            ctx_docs = retrieve_docs(active_vs, ref, k=4)
            answer   = generate_answer(ctx_docs, ref, hist)
            srcs     = list({d.metadata.get("source","Unknown") for d in ctx_docs})
            conf     = min(len(ctx_docs) * 25, 100)

    st.session_state.messages.append({
        "role": "assistant", "content": answer,
        "sources": srcs, "confidence": conf,
        "feedback": None, "reformulated": None,
    })
    _save_meta(st.session_state.source_names, st.session_state.selected_docs, st.session_state.messages)
    st.rerun()

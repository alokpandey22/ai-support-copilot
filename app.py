"""
app.py — AI Support Copilot
Premium dark-gold UI. Clean spinner text. Rate-limit safe.
"""

import datetime, glob, os, pickle, shutil, uuid
import streamlit as st

from rag_pipeline import (
    load_docs, chunk_docs, create_vector_store,
    retrieve_docs, retrieve_per_document,
    reformulate_query, generate_answer, summarize_all_documents,
)

for f in glob.glob("*.pdf"):
    if "-" in f:
        try: os.remove(f)
        except: pass

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
  --base:           #0C0D10;
  --elevated:       #131519;
  --surface:        #191C22;
  --surface-2:      #1F2229;
  --border:         rgba(255,255,255,0.06);
  --border-md:      rgba(255,255,255,0.11);
  --text-1:         #EAE8DF;
  --text-2:         #797772;
  --gold:           #C9A96E;
  --gold-bright:    #E2C090;
  --gold-dim:       rgba(201,169,110,0.11);
  --gold-bg:        rgba(201,169,110,0.06);
  --gold-border:    rgba(201,169,110,0.25);
  --emerald:        #3D9B78;
  --emerald-dim:    rgba(61,155,120,0.13);
  --emerald-border: rgba(61,155,120,0.28);
  --red:            #C85555;
  --red-dim:        rgba(200,85,85,0.12);
  --f-brand: 'Cormorant Garamond', Georgia, serif;
  --f-ui:    'Instrument Sans', system-ui, sans-serif;
  --f-mono:  'JetBrains Mono', monospace;
}

*, *::before, *::after { box-sizing: border-box; }
html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"] {
  background: var(--base) !important;
  color: var(--text-1) !important;
  font-family: var(--f-ui) !important;
  -webkit-font-smoothing: antialiased;
}
[data-testid="stHeader"],
[data-testid="stDecoration"],
#MainMenu, footer { display: none !important; }

/* ─── Sidebar ────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--elevated) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] section[data-testid="stSidebarContent"] {
  padding: 1rem 1rem 2rem !important;
  gap: 0.5rem !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stMarkdown p {
  font-family: var(--f-mono) !important;
  font-size: 11px !important;
  color: var(--text-2) !important;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
[data-testid="stSidebar"] .stMarkdown span,
[data-testid="stSidebar"] [data-testid="stMetricLabel"] span,
[data-testid="stSidebar"] [data-testid="stMetricValue"] span {
  font-family: var(--f-mono) !important;
  font-size: 11px !important;
  color: var(--text-2) !important;
}
[data-testid="stSidebar"] details > summary { list-style: none !important; }
[data-testid="stSidebar"] [data-testid="stExpanderToggleIcon"] { display: none !important; }

[data-testid="stSidebar"] .stFileUploader button span { display: none !important; }
[data-testid="stSidebar"] .stFileUploader button::after {
  content: "↑ Upload PDFs";
  font-family: var(--f-mono) !important;
  font-size: 11px !important;
  color: var(--gold) !important;
  white-space: nowrap;
  letter-spacing: 0.04em;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
  font-size: 9.5px !important;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: rgba(255,255,255,0.22) !important;
  margin-bottom: 4px !important;
  white-space: normal !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] *,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] > div { display: none !important; }

[data-testid="stSidebar"] [data-testid="stMetric"] {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 12px !important;
}
[data-testid="stSidebar"] [data-testid="stMetricValue"] {
  font-family: var(--f-brand) !important;
  font-size: 24px !important;
  color: var(--text-1) !important;
}
[data-testid="stSidebar"] [data-testid="stMetricLabel"] p {
  font-size: 9px !important;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: rgba(255,255,255,0.2) !important;
  white-space: normal !important;
}

[data-testid="stSidebar"] .stCheckbox {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 7px;
  padding: 6px 10px !important;
  margin-bottom: 4px !important;
  transition: border-color 0.15s, background 0.15s;
}
[data-testid="stSidebar"] .stCheckbox:has(input:checked) {
  background: var(--gold-dim);
  border-color: var(--gold-border);
}
[data-testid="stSidebar"] .stCheckbox label {
  font-family: var(--f-mono) !important;
  font-size: 11px !important;
  color: var(--text-2) !important;
  cursor: pointer;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  max-width: 180px;
  display: inline-block;
}
[data-testid="stSidebar"] .stCheckbox:has(input:checked) label { color: var(--gold) !important; }
[data-testid="stSidebar"] .stCheckbox input[type="checkbox"] { accent-color: var(--gold) !important; }

[data-testid="stSidebar"] .stButton > button {
  width: 100% !important;
  background: transparent !important;
  border: 1px solid var(--border-md) !important;
  border-radius: 7px !important;
  color: var(--text-2) !important;
  font-family: var(--f-mono) !important;
  font-size: 11px !important;
  padding: 5px 10px !important;
  transition: all 0.18s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  border-color: var(--gold) !important;
  color: var(--gold) !important;
  background: var(--gold-bg) !important;
}

[data-testid="stSidebar"] .stFileUploader > div {
  background: var(--gold-bg) !important;
  border: 1px dashed var(--gold-border) !important;
  border-radius: 8px !important;
  padding: 8px 12px !important;
}
[data-testid="stSidebar"] .stFileUploader label { color: var(--gold) !important; }

[data-testid="stSidebar"] .stDownloadButton > button {
  width: 100% !important;
  background: transparent !important;
  border: 1px solid var(--border-md) !important;
  border-radius: 8px !important;
  color: var(--text-2) !important;
  font-family: var(--f-mono) !important;
  font-size: 11px !important;
  padding: 8px 14px !important;
  transition: all 0.2s !important;
}
[data-testid="stSidebar"] .stDownloadButton > button:hover {
  border-color: var(--gold) !important;
  color: var(--gold) !important;
  background: var(--gold-bg) !important;
}

[data-testid="stSidebar"] .stExpander {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}
[data-testid="stSidebar"] hr { border-color: var(--border) !important; margin: 8px 0 !important; }

.sb-status {
  display: inline-flex; align-items: center; gap: 6px;
  font-size: 10px; font-family: var(--f-mono);
  color: var(--emerald); background: var(--emerald-dim);
  border: 1px solid var(--emerald-border);
  border-radius: 100px; padding: 4px 10px;
  margin-bottom: 12px; width: fit-content;
}
.sb-pulse {
  width: 5px; height: 5px; border-radius: 50%;
  background: var(--emerald); display: inline-block;
  animation: hb 2.4s ease-in-out infinite;
}
@keyframes hb { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.3;transform:scale(.8)} }

.scope-pill {
  display: inline-flex; align-items: center; gap: 5px;
  font-size: 10px; font-family: var(--f-mono);
  background: var(--gold-bg); border: 1px solid var(--gold-border);
  color: var(--gold); border-radius: 100px; padding: 3px 10px;
}

.sb-quality {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 12px;
  display: grid; grid-template-columns: repeat(3,1fr); width: 100%;
}
.sb-qm { text-align: center; position: relative; }
.sb-qm:not(:last-child)::after {
  content:''; position:absolute; right:0; top:50%; transform:translateY(-50%);
  width:1px; height:55%; background:var(--border);
}
.sb-qm-num { font-family: var(--f-brand) !important; font-size: 20px !important; font-weight: 600; line-height: 1.1; }
.sb-qm-lbl { font-size: 8.5px !important; letter-spacing: 0.1em; text-transform: uppercase; color: rgba(255,255,255,0.2) !important; font-family: var(--f-mono) !important; margin-top: 2px; }

[data-testid="stMain"] { background: var(--base) !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

.topbar {
  height: 56px; border-bottom: 1px solid var(--border);
  display: flex; align-items: center; padding: 0 40px; gap: 12px;
  background: var(--base); flex-wrap: wrap;
}
.tb-crumb { font-size: 12px; color: var(--text-2); font-family: var(--f-mono); }
.tb-sep   { width: 1px; height: 14px; background: var(--border); flex-shrink: 0; }

.chat-pad { padding: 32px 40px 12px; max-width: 820px; margin: 0 auto; }
.date-sep { display:flex; align-items:center; gap:12px; margin-bottom:24px; }
.ds-line  { flex:1; height:1px; background:var(--border); }
.ds-text  { font-size:9.5px; font-family:var(--f-mono); letter-spacing:.1em; color:rgba(255,255,255,0.17); text-transform:uppercase; white-space:nowrap; }

.msg-u { display:flex; justify-content:flex-end; margin-bottom:20px; }
.u-bubble {
  background: var(--surface); border: 1px solid var(--border-md);
  border-radius: 14px; border-bottom-right-radius: 3px;
  padding: 11px 16px; max-width: 68%;
  font-size: 14px; color: var(--text-1); line-height: 1.7; word-break: break-word;
}
.msg-a { margin-bottom: 4px; }
.a-row { display:flex; gap:11px; align-items:flex-start; }
.a-gem {
  width:30px; height:30px; border-radius:8px; flex-shrink:0;
  background:var(--gold-dim); border:1px solid var(--gold-border);
  display:flex; align-items:center; justify-content:center; margin-top:2px;
}
.a-gem svg { width:13px; height:13px; stroke:var(--gold); fill:none; stroke-linecap:round; stroke-linejoin:round; stroke-width:1.6; }
.a-bubble {
  background: var(--elevated); border: 1px solid var(--border);
  border-radius: 14px; border-top-left-radius: 3px;
  padding: 12px 18px; font-size: 14px; color: var(--text-1); line-height: 1.75; word-break: break-word;
}
.a-meta { display:flex; align-items:center; gap:7px; margin-top:8px; flex-wrap:wrap; padding-left:41px; }
.chip {
  display:inline-flex; align-items:center; gap:4px;
  font-family:var(--f-mono); font-size:9.5px; color:rgba(255,255,255,0.28);
  background:var(--surface); border:1px solid var(--border);
  border-radius:5px; padding:2px 8px;
}
.chip-dot { width:4px; height:4px; border-radius:50%; background:var(--emerald); flex-shrink:0; }
.conf-wrap { display:flex; align-items:center; gap:6px; }
.conf-track { width:60px; height:3px; background:var(--surface-2); border-radius:3px; overflow:hidden; }
.conf-fill  { height:100%; background:var(--gold); border-radius:3px; }
.conf-text  { font-size:9.5px; font-family:var(--f-mono); color:rgba(255,255,255,0.2); }
.fb-rated { font-size:9.5px; font-family:var(--f-mono); color:rgba(255,255,255,0.2); padding-left:41px; margin-top:3px; }

[data-testid="stMain"] .stButton > button {
  background: transparent !important; border: 1px solid var(--border) !important;
  border-radius: 6px !important; color: rgba(255,255,255,0.3) !important;
  font-size: 13px !important; padding: 2px 8px !important;
  min-width: 32px !important; transition: all 0.15s !important;
}
[data-testid="stMain"] .stButton > button:hover {
  border-color: var(--border-md) !important; color: var(--text-1) !important;
}

.input-hint {
  text-align:center; font-size:10px; font-family:var(--f-mono);
  color:rgba(255,255,255,0.13); letter-spacing:.04em; margin: 6px 0 2px;
}

[data-testid="stChatInputContainer"] {
  background: var(--elevated) !important; border: 1px solid var(--border-md) !important;
  border-radius: 13px !important; max-width: 820px !important; margin: 0 auto !important;
}
[data-testid="stChatInputContainer"]:focus-within { border-color: var(--gold-border) !important; }
[data-testid="stChatInputContainer"] textarea {
  background: transparent !important; color: var(--text-1) !important;
  font-family: var(--f-ui) !important; font-size: 14px !important;
}
[data-testid="stChatInputContainer"] textarea::placeholder { color: rgba(255,255,255,0.18) !important; }
[data-testid="stChatInputContainer"] button { background: var(--gold) !important; border-radius: 8px !important; border: none !important; }
[data-testid="stChatInputContainer"] button:hover { background: var(--gold-bright) !important; }

[data-testid="stChatMessage"] { background: transparent !important; border: none !important; padding: 0 !important; }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] { display: none !important; }

[data-testid="stAlert"], .stSuccess, .stWarning, .stError, .stInfo {
  background: var(--surface) !important; border: 1px solid var(--border-md) !important;
  border-radius: 8px !important; font-family: var(--f-mono) !important; font-size: 11px !important;
}
[data-testid="stExpander"] {
  background: var(--elevated) !important; border: 1px solid var(--border) !important; border-radius: 8px !important;
}

.ob-card {
  background: linear-gradient(145deg, #131519 0%, #181b22 100%);
  border: 1px solid var(--gold-border); border-radius: 14px;
  padding: 2.4rem 2rem; text-align: center;
  max-width: 580px; margin: 3rem auto; position: relative; overflow: hidden;
}
.ob-card::before {
  content:''; position:absolute; inset:0;
  background: radial-gradient(ellipse 70% 40% at 50% 0%, rgba(201,169,110,.07) 0%, transparent 70%);
  pointer-events:none;
}
.ob-title { font-family: var(--f-brand) !important; font-size: 1.85rem; color: var(--text-1); margin-bottom: .5rem; font-weight: 600; }
.ob-sub   { color: var(--text-2); font-size: .9rem; line-height: 1.6; margin: .35rem 0; }
.ob-sub strong { color: var(--gold); font-weight: 500; }
.ob-tiles { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; max-width:660px; margin:20px auto 0; }
.ob-tile  { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px 12px; }
.ob-tile-icon  { font-size: 1.3rem; margin-bottom: 6px; }
.ob-tile-title { font-family:var(--f-brand); font-size:.95rem; font-weight:600; color:var(--text-1); margin-bottom:3px; }
.ob-tile-desc  { font-size:10.5px; color:var(--text-2); line-height:1.45; }

* { scrollbar-width: thin; scrollbar-color: var(--border-md) transparent; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: var(--border-md); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
for k, v in {
    "vector_store":   None,
    "uploaded_count": 0,
    "messages":       [],
    "source_names":   [],
    "doc_chunks":     {},
    "query_log":      [],
    "selected_docs":  [],
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def save_doc_chunks(dc):
    with open("doc_chunks.pkl", "wb") as f: pickle.dump(dc, f)

def load_doc_chunks():
    if os.path.exists("doc_chunks.pkl"):
        with open("doc_chunks.pkl", "rb") as f: return pickle.load(f)
    return {}

def rebuild_vs(dc):
    chunks = [c for cs in dc.values() for c in cs]
    if not chunks: return None
    vs = create_vector_store(chunks)
    vs.save_local("faiss_index")
    return vs

def build_scoped_vs(selected_names, doc_chunks):
    chunks = [c for name in selected_names for c in doc_chunks.get(name, [])]
    if not chunks: return None
    return create_vector_store(chunks)

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


# ══════════════════════════════════════════════════════════════════════════════
#  AUTO-LOAD
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.vector_store is None:
    saved = load_doc_chunks()
    if saved:
        st.session_state.doc_chunks      = saved
        st.session_state.source_names    = list(saved.keys())
        st.session_state.vector_store    = rebuild_vs(saved)
        st.session_state.uploaded_count  = "saved"
        if not st.session_state.selected_docs:
            st.session_state.selected_docs = list(saved.keys())


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
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
    uploaded_files = st.file_uploader(
        "", type="pdf", accept_multiple_files=True, label_visibility="collapsed"
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
            st.rerun()

        st.write("")
        qa, qb = st.columns(2)
        with qa:
            if st.button("☑ All", key="sel_all"):
                st.session_state.selected_docs = list(st.session_state.source_names)
                st.rerun()
        with qb:
            if st.button("☐ None", key="sel_none"):
                st.session_state.selected_docs = []
                st.rerun()

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
                c1, c2 = st.columns([5, 1])
                c1.caption(truncate(name, 20))
                if c2.button("✕", key=f"rm_{name}"):
                    st.session_state.doc_chunks.pop(name, None)
                    st.session_state.source_names.remove(name)
                    st.session_state.selected_docs = [
                        d for d in st.session_state.selected_docs if d != name
                    ]
                    if st.session_state.doc_chunks:
                        st.session_state.vector_store = rebuild_vs(st.session_state.doc_chunks)
                        save_doc_chunks(st.session_state.doc_chunks)
                    else:
                        st.session_state.vector_store = None
                        for p in ("faiss_index", "doc_chunks.pkl"):
                            if os.path.isdir(p): shutil.rmtree(p)
                            elif os.path.isfile(p): os.remove(p)
                    st.rerun()
            st.divider()
            if st.button("🗑️ Clear All Documents"):
                st.session_state.update({
                    "vector_store": None, "doc_chunks": {}, "source_names": [],
                    "messages": [], "uploaded_count": 0, "selected_docs": [],
                })
                for p in ("faiss_index", "doc_chunks.pkl"):
                    if os.path.isdir(p): shutil.rmtree(p)
                    elif os.path.isfile(p): os.remove(p)
                st.rerun()

    if st.session_state.messages:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        st.download_button(
            "⬇ Download conversation",
            data=export_txt(st.session_state.messages).encode(),
            file_name=f"copilot_{ts}.txt",
            mime="text/plain",
        )
    else:
        st.caption("No conversation to export yet.")


# ══════════════════════════════════════════════════════════════════════════════
#  PROCESS UPLOADS
# ══════════════════════════════════════════════════════════════════════════════
if uploaded_files:
    temps = []
    for uf in uploaded_files:
        uname = f"{uuid.uuid4()}_{uf.name}"; temps.append(uname)
        with open(uname, "wb") as f: f.write(uf.read())
        docs   = load_docs(uname)
        chunks = chunk_docs(docs)
        if not chunks:
            st.warning(f"⚠️ Skipped '{uf.name}' — no text found."); continue
        for c in chunks: c.metadata["source"] = uf.name
        st.session_state.doc_chunks[uf.name] = chunks
        if uf.name not in st.session_state.source_names:
            st.session_state.source_names.append(uf.name)
            st.session_state.selected_docs.append(uf.name)
    for t in temps:
        try: os.remove(t)
        except: pass
    if st.session_state.doc_chunks:
        st.session_state.vector_store   = rebuild_vs(st.session_state.doc_chunks)
        save_doc_chunks(st.session_state.doc_chunks)
        st.session_state.uploaded_count = len(st.session_state.source_names)
        st.session_state.messages = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  RESOLVE ACTIVE VECTOR STORE
# ══════════════════════════════════════════════════════════════════════════════
sel       = st.session_state.selected_docs
all_names = st.session_state.source_names

if sel and set(sel) == set(all_names):
    active_vs    = st.session_state.vector_store
    active_names = all_names
elif sel:
    active_vs    = build_scoped_vs(sel, st.session_state.doc_chunks)
    active_names = sel
else:
    active_vs    = None
    active_names = []


# ══════════════════════════════════════════════════════════════════════════════
#  TOPBAR
# ══════════════════════════════════════════════════════════════════════════════
n_docs = len(st.session_state.source_names)
if sel:
    scope_html = (
        f'<span class="scope-pill">🎯 {len(sel)} of {n_docs} '
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


# ══════════════════════════════════════════════════════════════════════════════
#  ONBOARDING
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.vector_store:
    st.markdown("""
    <div class="ob-card">
      <div class="ob-title">👋 Welcome to CopilotAI</div>
      <p class="ob-sub">Upload company PDFs in the sidebar to get started.</p>
      <p class="ob-sub">Every answer is grounded exclusively in your documents —<br>no hallucinations, no guessing.</p>
      <br>
      <p class="ob-sub">📎 <strong>Tip:</strong> Upload multiple PDFs at once.</p>
      <p class="ob-sub">🔒 <strong>Private by default</strong> — stored only on your machine.</p>
    </div>
    <div class="ob-tiles">
      <div class="ob-tile"><div class="ob-tile-icon">🔍</div><div class="ob-tile-title">Ask questions</div><div class="ob-tile-desc">Instant grounded answers from your documents</div></div>
      <div class="ob-tile"><div class="ob-tile-icon">📄</div><div class="ob-tile-title">Summarize</div><div class="ob-tile-desc">Structured summaries of any document</div></div>
      <div class="ob-tile"><div class="ob-tile-icon">🗂️</div><div class="ob-tile-title">Manage library</div><div class="ob-tile-desc">Add or remove PDFs without restarting</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if not sel:
    st.info("⚠️ No documents selected. Use the sidebar checkboxes to choose which documents to query.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  CONVERSATION
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.messages:
    today     = datetime.datetime.now().strftime("%b %d")
    scope_ctx = f"{len(sel)} doc{'s' if len(sel)!=1 else ''} in scope"
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
                st.markdown(
                    f'<div class="fb-rated">{icon} You rated this response</div>',
                    unsafe_allow_html=True,
                )
            else:
                fc1, fc2, fc3 = st.columns([0.05, 0.05, 0.9])
                with fc1:
                    if st.button("👍", key=f"up_{i}"):
                        st.session_state.messages[i]["feedback"] = "helpful"
                        st.rerun()
                with fc2:
                    if st.button("👎", key=f"dn_{i}"):
                        st.session_state.messages[i]["feedback"] = "unhelpful"
                        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CHAT INPUT
# ══════════════════════════════════════════════════════════════════════════════
sel_str = ", ".join(truncate(d, 18) for d in sel[:3])
if len(sel) > 3:
    sel_str += f" +{len(sel)-3} more"

st.markdown(
    f'<div class="input-hint">Searching: {sel_str}</div>',
    unsafe_allow_html=True,
)

query = st.chat_input("Ask anything about your documents…")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Thinking…"):
        hist = st.session_state.messages[:-1]
        ref  = reformulate_query(query, hist)

        ql = query.lower()

        if any(w in ql for w in [
            "summarize all", "summary of all", "all documents",
            "each document", "one by one", "every document",
        ]):
            sums   = summarize_all_documents(active_vs, active_names)
            answer = "".join(f"### 📄 {k}\n{v}\n\n---\n\n" for k, v in sums.items())
            srcs   = active_names
            conf   = 100

        elif any(w in ql for w in [
            "summarize", "summary", "overview", "everything", "tell me about",
        ]):
            ctx_docs, srcs = retrieve_per_document(active_vs, ref, chunks_per_doc=3)
            answer = generate_answer(ctx_docs, ref, hist)
            conf   = min(len(ctx_docs) * 10, 100)

        else:
            ctx_docs = retrieve_docs(active_vs, ref, k=3)
            answer   = generate_answer(ctx_docs, ref, hist)
            srcs     = list({d.metadata.get("source", "Unknown") for d in ctx_docs})
            conf     = min(len(ctx_docs) * 33, 100)

    st.session_state.messages.append({
        "role":       "assistant",
        "content":    answer,
        "sources":    srcs,
        "confidence": conf,
        "feedback":   None,
        "reformulated": None,
    })
    st.rerun()
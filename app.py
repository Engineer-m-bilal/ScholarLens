# Streamlit single-file app: PDF RAG with Groq
# Keep your repo to only: app.py + requirements.txt

import os, io, json, math, pickle, textwrap, shutil, re, zipfile, tempfile
from typing import List, Dict, Any, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
import fitz  # PyMuPDF
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
from groq import Groq
import streamlit as st

# =========================
# Config
# =========================
EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K_DEFAULT = 5
MAX_CONTEXT_CHARS = 12000

INDEX_PATH = "rag_index.faiss"
STORE_PATH = "rag_store.pkl"

MODEL_CHOICES = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
]

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Session state
# =========================
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "docstore" not in st.session_state:
    st.session_state.docstore: List[Dict[str, Any]] = []
if "embedder" not in st.session_state:
    st.session_state.embedder = None

# =========================
# Utils
# =========================
def ensure_upload_dir() -> Path:
    p = Path("uploads")
    p.mkdir(exist_ok=True)
    return p

def write_uploaded_file(uploaded_file) -> str:
    """Save a Streamlit UploadedFile to disk and return path."""
    upload_dir = ensure_upload_dir()
    out_path = upload_dir / uploaded_file.name
    with open(out_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(out_path)

# ---------- PDF utils ----------
def extract_text_from_pdf(pdf_path: str) -> List[Tuple[int, str]]:
    pages = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            txt = page.get_text("text") or ""
            if not txt.strip():
                blocks = page.get_text("blocks")
                if isinstance(blocks, list):
                    txt = "\n".join(b[4] for b in blocks if isinstance(b, (list, tuple)) and len(b) > 4)
            pages.append((i, txt or ""))
    return pages

def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    text = text.replace("\x00", " ").strip()
    if len(text) <= chunk_size:
        return [text] if text else []
    out, start = [], 0
    while start < len(text):
        end = start + chunk_size
        out.append(text[start:end])
        start = max(end - overlap, start + 1)
    return out

# ---------- Embeddings / FAISS ----------
@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME, device=device)

def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return (vecs / norms).astype("float32")

def embed_passages(texts: List[str]) -> np.ndarray:
    model = st.session_state.embedder or load_embedder()
    st.session_state.embedder = model
    inputs = [f"passage: {t}" for t in texts]
    embs = model.encode(inputs, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    return _normalize(embs)

def embed_query(q: str) -> np.ndarray:
    model = st.session_state.embedder or load_embedder()
    st.session_state.embedder = model
    embs = model.encode([f"query: {q}"], convert_to_numpy=True)
    return _normalize(embs)

def build_faiss(embs: np.ndarray):
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index

def save_index(index, store_list: List[Dict[str, Any]]):
    faiss.write_index(index, INDEX_PATH)
    with open(STORE_PATH, "wb") as f:
        pickle.dump({"docstore": store_list, "embed_model": EMBED_MODEL_NAME}, f)

def load_index() -> bool:
    if Path(INDEX_PATH).exists() and Path(STORE_PATH).exists():
        st.session_state.faiss_index = faiss.read_index(INDEX_PATH)
        with open(STORE_PATH, "rb") as f:
            data = pickle.load(f)
        st.session_state.docstore = data["docstore"]
        _ = load_embedder()  # warm cache
        return True
    return False

# ---------- Ingest ----------
def ingest_pdfs(paths: List[str]) -> Tuple[Any, List[Dict[str, Any]]]:
    entries: List[Dict[str, Any]] = []
    for pdf in tqdm(paths, total=len(paths), desc="Parsing PDFs"):
        try:
            pages = extract_text_from_pdf(pdf)
            base = os.path.basename(pdf)
            for pno, ptxt in pages:
                if not ptxt.strip():
                    continue
                for ci, ch in enumerate(chunk_text(ptxt)):
                    entries.append({
                        "text": ch,
                        "source": base,
                        "page_start": pno,
                        "page_end": pno,
                        "chunk_id": f"{base}::p{pno}::c{ci}",
                    })
        except Exception as e:
            st.warning(f"Failed to parse {pdf}: {e}")
    if not entries:
        raise RuntimeError("No text extracted. If PDFs are scanned images, run OCR before indexing.")
    texts = [e["text"] for e in entries]
    embs = embed_passages(texts)
    index = build_faiss(embs)
    return index, entries

# ---------- Retrieval (supports required keywords) ----------
def retrieve(query: str, top_k=5, must_contain: str = ""):
    if st.session_state.faiss_index is None or not st.session_state.docstore:
        raise RuntimeError("Index not built or loaded. Use Build Index or Reload Saved Index first.")
    k = int(top_k) if top_k else TOP_K_DEFAULT

    pool = min(max(10 * k, 200), len(st.session_state.docstore))
    qemb = embed_query(query)
    D, I = st.session_state.faiss_index.search(qemb, pool)
    pairs = [(int(i), float(s)) for i, s in zip(I[0], D[0]) if i >= 0]

    must_words = [w.strip().lower() for w in must_contain.split(",") if w.strip()]
    if must_words:
        filtered = []
        for idx, score in pairs:
            t = st.session_state.docstore[idx]["text"].lower()
            if all(w in t for w in must_words):
                filtered.append((idx, score))
        if filtered:
            pairs = filtered

    pairs = pairs[:k]
    hits = []
    for idx, score in pairs:
        item = st.session_state.docstore[idx].copy()
        item["score"] = float(score)
        hits.append(item)
    return hits

# ---------- Groq LLM ----------
def groq_answer(query: str, contexts, model_name="llama-3.1-70b-versatile", temperature=0.2, max_tokens=1000):
    try:
        if not os.environ.get("GROQ_API_KEY"):
            return "GROQ_API_KEY is not set. Add it in Settings or the field above."
        client = Groq(api_key=os.environ["GROQ_API_KEY"])

        packed, used = [], 0
        for c in contexts:
            tag = f"[{c['source']} p.{c['page_start']}]"
            piece = f"{tag}\n{c['text'].strip()}\n"
            if used + len(piece) > MAX_CONTEXT_CHARS:
                break
            packed.append(piece); used += len(piece)
        context_str = "\n---\n".join(packed)

        system_prompt = (
            "You are a scholarly assistant. Answer using ONLY the provided context. "
            "If the answer is not present, say so. Always include a 'References' section with sources and page numbers."
        )
        user_prompt = (
            f"Question:\n{query}\n\n"
            f"Context snippets (use these only):\n{context_str}\n\n"
            "Write a precise answer. Keep claims traceable to the snippets."
        )

        resp = client.chat.completions.create(
            model=model_name,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        import traceback
        return f"Groq API error: {e}\n```\n{traceback.format_exc()}\n```"

# ---------- Build/Reload/Download ----------
def build_index_from_uploads(paths: List[str]) -> str:
    if not paths:
        return "Please upload at least one PDF."
    if len(paths) > 150:
        return "Please limit to about 100–150 PDFs per build."

    index, entries = ingest_pdfs(paths)
    save_index(index, entries)
    st.session_state.faiss_index = index
    st.session_state.docstore = entries
    return f"✅ Index built with {len(entries)} chunks from {len(paths)} PDFs."

def reload_index() -> str:
    ok = load_index()
    return f"🔁 Index reloaded. Chunks: {len(st.session_state.docstore)}" if ok else "No saved index found."

def make_index_zip() -> str | None:
    if not (Path(INDEX_PATH).exists() and Path(STORE_PATH).exists()):
        return None
    zp = "rag_index_bundle.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(INDEX_PATH)
        z.write(STORE_PATH)
    return zp

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="ScholarLens — PDF RAG", page_icon="📚", layout="wide")

st.title("📚 ScholarLens — Ask your PDFs with page-level proof")
st.caption("Upload PDFs • Build a semantic index • Ask questions with cited answers")

with st.expander("Optional: set your GROQ API key", expanded=False):
    key = st.text_input("GROQ_API_KEY", type="password", placeholder="sk_...")
    if st.button("Save key"):
        if key.strip():
            os.environ["GROQ_API_KEY"] = key.strip()
            st.success("API key set for this session.")
        else:
            st.info("No key provided.")

tab1, tab2 = st.tabs(["1) Build or Load Index", "2) Ask Questions"])

with tab1:
    st.subheader("Upload PDFs and build index")
    uploads = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    col_b1, col_b2, col_b3 = st.columns([1,1,1])
    build_clicked = col_b1.button("Build Index", type="primary", use_container_width=True)
    reload_clicked = col_b2.button("Reload Saved Index", use_container_width=True)
    download_clicked = col_b3.button("Download Index (.zip)", use_container_width=True)

    if build_clicked:
        try:
            if uploads:
                paths = [write_uploaded_file(f) for f in uploads]
                with st.spinner("Indexing…"):
                    msg = build_index_from_uploads(paths)
                st.success(msg)
            else:
                st.warning("Please upload at least one PDF.")
        except Exception as e:
            st.error(f"Error while building index: {e}")

    if reload_clicked:
        msg = reload_index()
        (st.success if msg.startswith("🔁") else st.warning)(msg)

    if download_clicked:
        zp = make_index_zip()
        if zp:
            with open(zp, "rb") as f:
                st.download_button("Download rag_index_bundle.zip", f, file_name="rag_index_bundle.zip", mime="application/zip", use_container_width=True)
        else:
            st.warning("No saved index found to package.")

    st.write("---")
    st.write(f"**Chunks:** {len(st.session_state.docstore)}")
    if st.session_state.docstore:
        st.caption("You can switch to the next tab to start asking questions.")

with tab2:
    st.subheader("Ask a question")
    query = st.text_area("Your question", height=90, placeholder="Ask something present in the uploaded papers…")
    col1, col2, col3 = st.columns([1,1,1])
    top_k = col1.slider("Top-K passages", 1, 20, TOP_K_DEFAULT, 1)
    model_name = col2.selectbox("Groq model", MODEL_CHOICES, index=0)
    temperature = col3.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    must = st.text_input("Must contain (comma-separated keywords)", placeholder="camera, CMOS, frame rate")

    if st.button("Answer", type="primary"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Retrieving and generating…"):
                    ctx = retrieve(query, top_k=top_k, must_contain=must)
                    answer = groq_answer(query, ctx, model_name=model_name, temperature=temperature)
                st.markdown(answer)

                # Show top passages as a table
                rows = []
                for c in ctx:
                    preview = c["text"][:220].replace("\n", " ") + ("…" if len(c["text"]) > 220 else "")
                    rows.append([c["source"], c["page_start"], round(c["score"], 3), preview])
                df = pd.DataFrame(rows, columns=["Source", "Page", "Score", "Snippet"])
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(str(e))

import streamlit as st
import os
from backend.pdf_parser import chunk_pages_for_embeddings, extract_pages_lines
from backend.vector_store import VectorStore, get_local_embeddings
from backend.qa_engine import answer_question, summarize_paper
from backend.export_docx import export_answer_docx
from PyPDF2 import PdfReader

# ✅ Browser tab config (App name set here)
st.set_page_config(
    page_title="ScholarLense",  # 👈 Tab title
    page_icon="📚",             # 👈 Icon
    layout="wide"
)

# ✅ Fix for DATA_DIR (no __file__ issue in Streamlit)
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "data", "papers"))
os.makedirs(BASE_DIR, exist_ok=True)

# ✅ Global vector store instance
vs = VectorStore()


# ✅ Extract title from PDF (metadata → fallback first page)
def extract_paper_title(pdf_path):
    try:
        reader = PdfReader(pdf_path)

        # 1. Metadata title
        if reader.metadata and "/Title" in reader.metadata:
            title = reader.metadata["/Title"]
            if title and len(title.strip()) > 3:
                return title.strip()

        # 2. Fallback: first line of first page
        first_page = reader.pages[0].extract_text().split("\n")
        if first_page:
            return first_page[0].strip()
    except Exception:
        pass
    return "Untitled Paper"


# ✅ Custom CSS
def add_custom_css():
    st.markdown(
        """
        <style>
        h1 {
            text-align: center;
            color: #4CAF50;
            animation: fadeIn 2s ease-in-out;
        }
        .paper-title {
            padding: 10px;
            background: #5fc76f;
            border-left: 6px solid #1f77b4;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        .delete-btn {
            background-color: #e74c3c;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 2px 8px;
            cursor: pointer;
        }
        .delete-btn:hover {
            background-color: #c0392b;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            transition: 0.3s;
        }
        .stButton button:hover {
            background-color: #45a049;
            transform: scale(1.05);
        }
        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_ui():
    st.sidebar.title("📚 Paper Library")

    # ✅ Upload new file
    uploaded = st.sidebar.file_uploader("📂 Upload a PDF", type=["pdf"])
    if uploaded is not None:
        save_path = os.path.join(BASE_DIR, uploaded.name)
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.sidebar.success(f"✅ Saved {uploaded.name} to library")

        # ✅ Extract + embed immediately
        with st.spinner("⚡ Processing PDF for embeddings..."):
            chunks = chunk_pages_for_embeddings(save_path)
            texts = [c["text"] for c in chunks]
            embeddings = get_local_embeddings(texts)
            vs.add_embeddings(embeddings, chunks)

        st.sidebar.success("🎉 PDF indexed into vector store")

    st.sidebar.markdown("---")

    # ✅ Show all saved PDFs even after refresh
    st.sidebar.subheader("📑 Your Library")
    pdf_files = os.listdir(BASE_DIR)

    if pdf_files:
        # Selectbox for choosing paper
        selected_pdf = st.sidebar.selectbox("Select a paper", pdf_files)
        st.session_state["selected_pdf"] = selected_pdf

        pdf_path = os.path.join(BASE_DIR, selected_pdf)
        paper_title = extract_paper_title(pdf_path)

        # ✅ Show title nicely in sidebar
        st.sidebar.markdown(
            f"""
            <div class="paper-title">
                <h4>📄 {paper_title}</h4>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.session_state["paper_title"] = paper_title

        # ✅ Delete button
        if st.sidebar.button("🗑️ Delete Selected Paper"):
            os.remove(pdf_path)
            st.sidebar.success(f"🗑️ Deleted {selected_pdf}")
            if "selected_pdf" in st.session_state:
                del st.session_state["selected_pdf"]
            st.rerun()

    else:
        st.sidebar.warning("⚠️ No papers uploaded yet.")


def main_ui():
    add_custom_css()
    st.markdown("<h1>✨ Research Insight Assistant ✨</h1>", unsafe_allow_html=True)

    # ✅ Show selected paper title in main UI
    if "selected_pdf" in st.session_state:
        st.markdown(
            f"""
            <div class="paper-title">
                <h3>📘 {st.session_state.get('paper_title', st.session_state['selected_pdf'])}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 🔍 Question Answering
    st.markdown("#### 🔍 Ask Questions from Your Uploaded Papers")
    q = st.text_input("Type your research query here...")

    if st.button("🚀 Get Answer") and q.strip():
        if "selected_pdf" not in st.session_state:
            st.warning("⚠️ Please select a paper from the sidebar.")
        else:
            with st.spinner("🔎 Searching the knowledge base..."):
                res = answer_question(q)

            st.success("✅ Answer Ready!")

            # ✅ Title above Answer
            st.markdown(
                f"<h4>📄 Answer from: {st.session_state.get('paper_title', st.session_state['selected_pdf'])}</h4>",
                unsafe_allow_html=True
            )

            with st.expander("💡 View Answer", expanded=True):
                st.write(res["answer"])

            st.progress(int(res["confidence"] * 100))
            st.caption(f"Confidence Score: **{res['confidence']}**")

            st.subheader("📖 Citations")
            for c in res["citations"]:
                st.markdown(
                    f"**Page {c['page']} (lines {c['start_line']}-{c['end_line']})**"
                )
                st.info(c["snippet"])

            if st.button("💾 Export as .docx"):
                out = export_answer_docx(res, "exported_answer.docx")
                st.success(f"📂 Exported to {out}")

    st.markdown("---")
    st.markdown("#### 📑 Generate Paper Summary")

    if st.button("📝 Summarize Selected Paper"):
        if "selected_pdf" not in st.session_state:
            st.warning("⚠️ Please select a paper from the sidebar.")
        else:
            with st.spinner("✨ Summarizing the paper..."):
                pdf_path = os.path.join(BASE_DIR, st.session_state["selected_pdf"])
                pages = extract_pages_lines(pdf_path)
                all_texts = [" ".join(p) for p in pages]

                summary = summarize_paper(all_texts)

                # ✅ Title above Summary
                st.markdown(
                    f"<h4>📄 Summary of: {st.session_state.get('paper_title', st.session_state['selected_pdf'])}</h4>",
                    unsafe_allow_html=True
                )
                st.success(summary)


if __name__ == "__main__":
    sidebar_ui()
    main_ui()

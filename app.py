import os
import hashlib
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Cellular KPI RAG", page_icon="📡", layout="wide")
load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DB_DIR = Path("vector_db")


@st.cache_resource
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def read_uploaded_text(uploaded_file):
    raw = uploaded_file.getvalue()
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="ignore")


def split_text(text, chunk_size=1200, overlap=200):
    text = text.strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind("\n", start, end)
            if boundary > start + 300:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def database_id(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_faiss_index(chunks):
    model = get_embedding_model()
    vectors = model.encode(
        chunks, convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def save_database(index, chunks, db_id):
    VECTOR_DB_DIR.mkdir(exist_ok=True)
    index_path = VECTOR_DB_DIR / f"{db_id}.faiss"
    chunks_path = VECTOR_DB_DIR / f"{db_id}.txt"
    faiss.write_index(index, str(index_path))
    chunks_path.write_text("\n\n---CHUNK---\n\n".join(chunks), encoding="utf-8")
    return index_path


def search_database(question, index, chunks, top_k=5):
    model = get_embedding_model()
    vector = model.encode(
        [question], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    k = min(top_k, len(chunks))
    scores, indices = index.search(vector, k)
    return [
        {"text": chunks[idx], "score": float(score)}
        for score, idx in zip(scores[0], indices[0])
        if idx >= 0
    ]


def get_api_key():
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY")


def ask_groq(question, results):
    api_key = get_api_key()
    if not api_key:
        return "Groq API key not found. Add GROQ_API_KEY to .env or Streamlit Secrets."

    context = "\n\n".join(
        f"[Source {i+1}]\n{item['text']}"
        for i, item in enumerate(results)
    )

    system_prompt = """You are an expert Cellular Network KPI engineer.
Use only the retrieved KPI knowledge provided by the user.
Do not invent formulas, counters, targets, thresholds, or vendor parameters.
If the answer is missing, say that it is not available in the uploaded KPI knowledge base.
Be clear and technically accurate."""

    user_prompt = f"""RETRIEVED KPI KNOWLEDGE:
{context}

QUESTION:
{question}
"""

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        max_tokens=1500,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


for key, value in {
    "index": None, "chunks": [], "file_name": None,
    "db_id": None, "text": ""
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


st.title("📡 Cellular KPI RAG Database")
st.caption("Open-source Streamlit + FAISS + Sentence Transformers + Groq")

with st.sidebar:
    st.header("⚙️ Build Knowledge Base")
    uploaded_file = st.file_uploader("Upload Cellular KPI TXT File", type=["txt"])

    if uploaded_file and st.button("🚀 Build FAISS Database", type="primary"):
        text = read_uploaded_text(uploaded_file)
        chunks = split_text(text)

        if not chunks:
            st.error("The uploaded TXT file is empty.")
        else:
            with st.spinner("Creating embeddings and FAISS index..."):
                index = build_faiss_index(chunks)
                db_id = database_id(text)
                index_path = save_database(index, chunks, db_id)

            st.session_state.index = index
            st.session_state.chunks = chunks
            st.session_state.file_name = uploaded_file.name
            st.session_state.db_id = db_id
            st.session_state.text = text
            st.success("FAISS vector database created successfully.")
            st.caption(f"Saved: {index_path}")

    st.divider()
    st.write("**Groq status:**")
    st.success("API key detected") if get_api_key() else st.warning("API key not detected")


c1, c2, c3 = st.columns(3)
c1.metric("Database", "Ready" if st.session_state.index else "Not Built")
c2.metric("Knowledge Chunks", len(st.session_state.chunks))
c3.metric("Model", GROQ_MODEL)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 KPI Dashboard", "🔍 Semantic Search", "🤖 RAG Assistant", "📚 Database Info"]
)

with tab1:
    st.subheader("KPI Knowledge Dashboard")
    if not st.session_state.chunks:
        st.info("Upload a KPI TXT file and build the database.")
    else:
        term = st.text_input("Filter knowledge", placeholder="throughput, RACH, VoLTE, PRB")
        shown = st.session_state.chunks
        if term:
            shown = [x for x in shown if term.lower() in x.lower()]
        st.metric("Displayed Sections", len(shown))
        for i, chunk in enumerate(shown, 1):
            with st.expander(f"KPI Knowledge Section {i}"):
                st.write(chunk)

with tab2:
    st.subheader("Semantic KPI Search")
    query = st.text_input("Search by meaning", placeholder="Why does downlink user throughput become low?")
    if query:
        if not st.session_state.index:
            st.warning("Build the FAISS database first.")
        else:
            for i, result in enumerate(
                search_database(query, st.session_state.index, st.session_state.chunks), 1
            ):
                with st.expander(f"Result {i} — Similarity {result['score']:.3f}", expanded=i == 1):
                    st.write(result["text"])

with tab3:
    st.subheader("🤖 Cellular KPI RAG Assistant")
    st.markdown("""Examples:
- What causes low 5G DL throughput?
- What is the formula for RACH Success Rate?
- Which counters are related to VoLTE Drop Rate?
- How can high PRB utilization be optimized?""")
    question = st.text_area("Ask a KPI question", height=110)
    if st.button("Ask Groq", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        elif not st.session_state.index:
            st.warning("Build the FAISS database first.")
        else:
            with st.spinner("Retrieving relevant KPI knowledge..."):
                results = search_database(
                    question, st.session_state.index, st.session_state.chunks
                )
            with st.spinner("Generating answer with Groq..."):
                try:
                    st.markdown("### Answer")
                    st.write(ask_groq(question, results))
                except Exception as error:
                    st.error(f"Groq request failed: {error}")

            with st.expander("🔎 View retrieved RAG context"):
                for i, result in enumerate(results, 1):
                    st.markdown(f"**Source {i} | Similarity {result['score']:.3f}**")
                    st.write(result["text"])

with tab4:
    st.subheader("Database Information")
    if not st.session_state.index:
        st.info("No database has been built yet.")
    else:
        st.write(f"**Uploaded file:** {st.session_state.file_name}")
        st.write(f"**Database ID:** {st.session_state.db_id}")
        st.write("**Vector database:** FAISS")
        st.write(f"**Embedding model:** {EMBEDDING_MODEL_NAME}")
        st.write(f"**Vectors:** {st.session_state.index.ntotal}")
        st.write(f"**LLM:** Groq / {GROQ_MODEL}")
        st.download_button(
            "⬇️ Download uploaded KPI text",
            st.session_state.text,
            st.session_state.file_name or "kpi_knowledge.txt",
            "text/plain",
        )

st.divider()
st.caption("Never commit your GROQ_API_KEY to GitHub. Use Streamlit Secrets for deployment.")

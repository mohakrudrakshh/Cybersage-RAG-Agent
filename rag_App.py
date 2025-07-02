import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from tools.local_search_tool import LocalSemanticSearch
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import torch
import tempfile
import os
import hashlib
import pickle
import re

# ------------------------------
# Load CVE CSV-based search engine
search_tool = LocalSemanticSearch("CVE_embeddings.csv")

# ------------------------------
# Load SentenceTransformer for PDF chunks
embed_model = SentenceTransformer("all-mpnet-base-v2")

# ------------------------------
# Load LLM
@st.cache_resource
def load_llm():
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return tokenizer, model

tokenizer, model = load_llm()

# ------------------------------
# Extract and chunk PDF
def extract_pdf_chunks(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text() or ""

    # Basic paragraph/sentence chunking
    chunks = re.split(r"\n+|(?<=[.])\s", raw_text)
    return [chunk.strip() for chunk in chunks if len(chunk.strip()) > 50]

# ------------------------------
# Cache embeddings using PDF hash
@st.cache_data(show_spinner=False)
def get_pdf_embeddings(pdf_bytes):
    pdf_hash = hashlib.md5(pdf_bytes).hexdigest()
    cache_path = os.path.join(tempfile.gettempdir(), f"{pdf_hash}_chunks.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    with open(tempfile.mktemp(suffix=".pdf"), "wb") as tmp_pdf:
        tmp_pdf.write(pdf_bytes)
        chunks = extract_pdf_chunks(tmp_pdf.name)

    embeddings = embed_model.encode(chunks, convert_to_tensor=True)
    with open(cache_path, "wb") as f:
        pickle.dump((chunks, embeddings), f)
    return chunks, embeddings

# ------------------------------
# Prompt template
def generate_prompt(query, context):
    return f"""You are a cybersecurity analyst. Use the following threat intelligence to answer the query.

[Context]: {context}

[Query]: {query}

Answer:"""

def get_response(query, context):
    prompt = generate_prompt(query, context)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ------------------------------
# Streamlit UI
st.set_page_config(page_title="CyberSage", layout="wide")
st.title("🛡️ CyberSage: Threat Intel RAG Assistant")
st.write("Upload a threat intel PDF and ask CVE-related security questions using local context + LLM.")

uploaded_pdf = st.file_uploader("📄 Upload PDF", type=["pdf"])

if uploaded_pdf:
    pdf_bytes = uploaded_pdf.read()
    with st.spinner("🔍 Embedding PDF content..."):
        pdf_chunks, pdf_embeddings = get_pdf_embeddings(pdf_bytes)

    query = st.text_input("🔐 Enter your cybersecurity question")

    if query:
        with st.spinner("⚙️ Searching CVE + PDF context..."):
            # CVE search
            cve_context = search_tool.search(query, top_k=1)[0][0]

            # PDF chunk most similar
            query_embedding = embed_model.encode(query, convert_to_tensor=True)
            scores = util.cos_sim(query_embedding, pdf_embeddings)[0]
            best_idx = torch.argmax(scores).item()
            pdf_context = pdf_chunks[best_idx]

            # Combine both
            final_context = f"CVE Info: {cve_context}\n\nReport Insight: {pdf_context}"
            response = get_response(query, final_context)

        st.subheader("🧠 LLM Response")
        st.write(response)

        st.subheader("📚 CVE Context Used")
        st.info(cve_context)

        st.subheader("📄 PDF Context Used")
        st.info(pdf_context)

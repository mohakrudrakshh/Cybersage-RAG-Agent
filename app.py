# # app.py

# import streamlit as st
# from tools.local_search_tool import LocalSemanticSearch

# st.set_page_config(page_title="CyberSage CVE Search", layout="wide")
# st.title("🔍 CyberSage: Local CVE Semantic Search")
# st.markdown("Search critical CVEs from a 2024 threat report PDF using local embeddings.")

# # Initialize search tool
# csv_path = "CVE_embeddings.csv"  # Path to your preprocessed CSV file
# search_tool = LocalSemanticSearch(csv_path)

# # Query input
# query = st.text_input("Enter your search query:")

# # Top-k selector
# top_k = st.slider("Number of top results to show:", 1, 10, 3)

# if query:
#     results = search_tool.search(query, top_k=top_k)
#     st.subheader("🔎 Top Matches")
#     for i, (chunk, score) in enumerate(results, 1):
#         st.markdown(f"---\n### 📝 Result {i} (Score: `{score:.4f}`)")
#         st.write(chunk)
import streamlit as st
import pandas as pd
import tempfile
import os
from tools.local_search_tool import LocalSemanticSearch
from localEmbedding4 import process_pdf

st.set_page_config(page_title="CyberSage CVE Search", layout="wide")
st.title("🔐 CyberSage: Local CVE Semantic Search + PDF Uploader")

st.markdown("""
Upload a **CVE-related PDF report**, and CyberSage will:
1. 🧠 Extract & chunk text
2. 🤖 Generate sentence embeddings
3. 🔍 Let you semantically search for CVEs and explanations
""")

# Sidebar filters
st.sidebar.header("🔧 Filters & Settings")
sentence_size = st.sidebar.slider("Sentence Chunk Size", min_value=3, max_value=20, value=10)
top_k = st.sidebar.slider("Top Results to Display", 1, 10, 3)

# File uploader
uploaded_pdf = st.file_uploader("📄 Upload a CVE PDF Report", type=["pdf"])

if uploaded_pdf:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(uploaded_pdf.read())
        pdf_path = tmp_pdf.name

    st.success("✅ File uploaded successfully!")

    output_csv_path = "CVE_embeddings_uploaded.csv"

    with st.spinner("🔄 Generating sentence embeddings..."):
        process_pdf(pdf_path, sentence_size, output_csv_path)

    st.success("✅ Embeddings generated!")

    # Option to download CSV
    with open(output_csv_path, "rb") as f:
        st.download_button("⬇️ Download Embedding CSV", f, file_name="CVE_embeddings.csv", mime="text/csv")

    # Load search engine
    search_tool = LocalSemanticSearch(output_csv_path)

    # Search box
    st.subheader("🔍 Enter your search query")
    query = st.text_input("e.g., Remote code execution in Windows 11")

    if query:
        with st.spinner("Searching..."):
            results = search_tool.search(query, top_k=top_k)

        st.subheader("🧠 Top Semantic Matches")
        for i, (chunk, score) in enumerate(results, 1):
            st.markdown(f"---\n### Result {i} (Score: `{score:.4f}`)")
            st.write(chunk)
            st.markdown("👍 Was this relevant?")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.button("👍 Yes", key=f"yes_{i}")
            with col2:
                st.button("👎 No", key=f"no_{i}")
else:
    st.info("👈 Upload a PDF file to start searching CVEs!")

# 🧠 CyberSage: Agentic RAG for Cybersecurity Intelligence

**CyberSage** is an advanced Retrieval-Augmented Generation (RAG) pipeline enhanced with agentic workflows for cybersecurity applications. It builds upon the [CyberScienceLab RAG_LLM_CVE](https://github.com/CyberScienceLab/RAG_LLM_CVE) repository, extending its capabilities with intelligent agents, enriched threat analysis, and a modular architecture that supports multiple cybersecurity tasks like CVE summarization, log analysis, and threat hunting.
![image](https://github.com/user-attachments/assets/cc78016b-8447-4432-9ef2-1141b9247741)

---

## 🚀 Key Features

- 🔍 Vector-based Semantic Retrieval over PDF threat reports  
- 🤖 Agentic Architecture for modular, extensible workflows  
- 📄 CVE Intelligence Pipeline with metadata validation & generation  
- 🧩 Local JSON CVE validation using NVD-like structure  
- 🗂️ Chunked Document Processing with Sentence Transformers  
- 🧠 Meta LLaMA-3 8B Instruct Integration via HuggingFace Transformers  
- 🧪 Streamlit UI & CLI Support (optional, toggleable)  
- 📚 Designed for cybersecurity research, SOC augmentation, and analyst workflows  

---

## 🔧 Project Structure

```
Cybersage-RAG-Agent/
├── agents/
│   ├── log_analysis_agent.py
│   └── cve_summarizer_agent.py
├── retriever/
│   └── local_semantic_search.py
├── cve_tools/
│   ├── cve_validator.py
│   └── cve_extractor.py
├── data/
│   ├── sample_pdf_reports/
│   └── local_cve_db.json
├── rag_App.py
├── theRag.py
├── requirements.txt
└── README.md
```

---

## 🛠 Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/Cybersage-RAG-Agent.git
cd Cybersage-RAG-Agent
```

### 2. Create and Activate a Virtual Environment (recommended)

```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Authenticate with HuggingFace and Load LLaMA-3

```bash
huggingface-cli login
```

Model loads automatically with:
- `torch_dtype=torch.bfloat16`
- `device_map="auto"`
- Memory optimization as needed

---

## ⚙️ How It Works

- **Semantic Search**: Query embedded via sentence transformer → top-k context chunks via cosine similarity  
- **Agent Pipeline**: Calls log or CVE summarizer agent based on input type  
- **RAG + LLM**: Sends retrieved context + prompt to Meta LLaMA-3 model  
- **CVE Validation**: CVEs are validated from a local JSON CVE metadata set  

---

## ✅ Enhancements Made

| Feature                    | Description |
|----------------------------|-------------|
| 🧠 Agentic Workflow         | Modular agents (CVE Summarizer, Log Analyzer) |
| 🔍 Vector Search Fix        | Fixed `semantic_search()` bug in `LocalSemanticSearch` |
| 📊 CVE Validation Layer     | Local NVD-style JSON cross-checking |
| 🧠 LLaMA-3 Integration      | Efficient model loading with HuggingFace |
| 🔧 Inference Tuning         | Configured top_p, temperature, token limits |
| 🖥️ Streamlit Interface      | Simple interactive UI |
| 🪛 Attribute Fixes          | Patched critical errors |
| 🗃️ Modular Data Handling    | Supports multiple PDF threat reports |
![image](https://github.com/user-attachments/assets/e8e7136c-c1e4-4159-8679-34967d558e27)


---

## 🧪 Sample Usage

```bash
streamlit run rag_App.py
```
![image](https://github.com/user-attachments/assets/eea83b4e-bec5-462c-a261-e2bc509ceaa6)


Try out:
- “Explain CVE-2023-23397”
- Upload PDF reports for semantic retrieval
- Log simulation via log agent

---

## 🧠 Example Prompt Template

```
Context:
<retrieved documents>

Question:
What is the impact of CVE-2023-23397 on Outlook clients?

Answer (based on the context only):
```

---

## 🔒 Security Focus Areas

- Threat Intelligence Summarization  
- Local CVE Metadata Lookup  
- SOC Log Analysis  
- Generative Threat Report Generation  

---

## 👨‍💻 Author

Developed & extended by **Rudraksh Gupta**  
[@mohakrudrakshh](mailto:rudrakshgupta022@gmail.com)  
Cybersecurity MSc | AI x Threat Intelligence  
GitHub: [@mohakrudrakshh](https://github.com/mohakrudrakshh)

---

## 🧠 Future Directions

- Multi-agent collaboration using LangGraph / CrewAI  
- Dynamic PDF parsing and NVD integration  
- Web-based dashboard with history logging  
- Real-time threat feed summarization  

---

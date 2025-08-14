Here’s a complete `README.md` for your code:

---

# 🌐 AI Research Discovery Tool

An **interactive Streamlit application** that integrates **multi-source literature search**, **AI-assisted research ideation**, **deep study summarization**, and **dataset/tool discovery** for biomedical and AI research topics.
It connects to sources like **PubMed**, **EuropePMC**, **Semantic Scholar**, and **arXiv**, and uses **Groq's LLaMA3 model** for structured idea generation, study plan creation, and research gap identification.

---

## 🚀 Features

### 1. 🎯 **Generate Project Ideas**

* Search multiple literature sources for a given research topic.
* Apply **year**, **citation count**, and **source** filters.
* Automatically **cluster similar studies** (with optional sentence-transformer embeddings).
* Generate **3 high-impact, distinct project ideas per cluster**.
* View **cross-paper synthesis** (themes, contradictions, unified direction).
* Export results as **Markdown**.
* Create an **independent study plan** for any idea.

### 2. 📚 **Deep Study Analyzer**

* Paste a **paper link** (PubMed, arXiv, or generic URL).
* Automatically fetch and summarize:

  * Background, objective, methodology
  * Figures & results
  * Core findings
  * Limitations & next steps
* Ask **interactive questions** about the study.

### 3. 🔍 **Dataset & Tool Finder**

* Search for datasets, tools, and pre-trained models from:

  * HuggingFace
  * OpenML
  * Kaggle
* AI-assisted tool discovery via **Groq LLaMA3**.
* Filter results by **modality**, **accessibility**, and **relevance score**.
* Export search results as **Markdown**.

---

## 📦 Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/ai-research-discovery-tool.git
cd ai-research-discovery-tool
```

2. **Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
   Create a `.env` file in the project root with:

```
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🖥 Usage

Run the app locally:

```bash
streamlit run app.py
```

Once running:

* Navigate between **Project Ideas**, **Deep Study Analyzer**, and **Dataset & Tool Finder** using the sidebar.
* Use **Advanced Filters** to refine your search.
* Click action buttons to generate **study plans**, **find gaps**, or **download results**.

---

## 📚 Data Sources

| Source               | Data Type            | Notes                          |
| -------------------- | -------------------- | ------------------------------ |
| **PubMed**           | Biomedical papers    | Uses NCBI E-utilities API      |
| **EuropePMC**        | Life sciences papers | Filters by publication date    |
| **Semantic Scholar** | AI + general papers  | Supports citation count filter |
| **arXiv**            | Preprints            | Useful for AI/ML preprints     |
| **HuggingFace**      | Datasets & models    | Search via API                 |
| **OpenML**           | Datasets             | API search                     |
| **Kaggle**           | Datasets             | Web scraping fallback          |

---

## ⚙️ Configuration

* **Model**: LLaMA3-70B (Groq API)
* **Timeouts**: All network requests default to 30–60 seconds.
* **Retries**: Network/LLM calls use exponential backoff.
* **Caching**: Results cached with `st.cache_data` for up to 1 hour.

---

## 📁 Project Structure

```
app.py                 # Main Streamlit application
requirements.txt       # Python dependencies
.env                   # Environment variables
README.md              # This file
```

---

## 🛠 Dependencies

Key Python packages:

* `streamlit` – Web app framework
* `requests` – API calls
* `beautifulsoup4` – HTML parsing
* `pydantic` – JSON schema validation
* `plotly` – Interactive plots
* `sentence-transformers` – Text embeddings (optional clustering)
* `scikit-learn` – Clustering (KMeans)
* `python-dotenv` – Env var loading

---

## 🧪 Example Workflow

1. Go to **🎯 Generate Project Ideas**.
2. Enter `"epilepsy EEG"`, select PubMed + EuropePMC.
3. Apply **year filter** `2020–2024`.
4. Enable **clustering** with `3 clusters`.
5. Click **Generate Project Ideas**.
6. Review:

   * Top ideas with scores
   * Thematic synthesis
   * Suggested datasets
7. Export to Markdown or create a study plan.

---

## 🔒 Notes on API Keys

* **Groq API key** is required for AI features.
* API keys are loaded from `.env` file.
* Keep `.env` **private** and **never commit it** to version control.

---


If you want, I can also **add a quick-start GIF** showing the tool in action so the README looks more engaging. Would you like me to prepare that?

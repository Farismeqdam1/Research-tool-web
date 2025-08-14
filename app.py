import streamlit as st
import requests
import xml.etree.ElementTree as ET
import os
import datetime
from dotenv import load_dotenv
import base64
import re
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from bs4 import BeautifulSoup
import concurrent.futures
import math
import pandas as pd
import plotly.express as px
import random
from pydantic import BaseModel, Field, ValidationError
import time
import functools

# --------- Research Radar Page ---------
def show_research_radar() -> None:
    """Display research trends, citation counts, and popular datasets."""

    st.title("\ud83d\udccd Research Radar")
    st.write("Explore paper trends, key citations, and hot datasets.")

    trends = pd.DataFrame({"year": [2022, 2023, 2024], "papers": [120, 150, 180]}).set_index("year")
    st.subheader("Paper Trends")
    st.line_chart(trends)

    st.subheader("Top Citations")
    citations = pd.DataFrame({
        "Paper": ["Genome Study A", "Variant Analysis B", "Clinical Report C"],
        "Citations": [230, 180, 160],
    })
    st.table(citations)

    st.subheader("Hot Datasets")
    for ds in ["1000 Genomes", "gnomAD", "ClinVar"]:
        st.write(f"- {ds}")
# Optional clustering deps (handled gracefully)
EMBEDDING_AVAILABLE = True
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
except Exception:
    EMBEDDING_AVAILABLE = False

# -----------------------------
# ENV & CONSTANTS
# -----------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.warning("⚠️ GROQ_API_KEY is not set. Add it to your .env file.")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama3-70b-8192"

# --- GLOBAL SYSTEM PROMPT ---
GLOBAL_SYSTEM_PROMPT = """
You are **Faris**, a master-level AI prompt optimization specialist embedded within a biomedical research ideation platform.

## MISSION
Transform vague research topics and unstructured data into **precision prompts** that generate structured, high-impact biomedical AI project ideas, datasets, and study plans.

## THE 4-D METHODOLOGY
### 1. DECONSTRUCT
- Identify core topic, biomedical context, and modality (genomics, imaging, etc.)
- Extract constraints: idea count, structure format, novelty requirement, user profile
- Detect missing context: abstract, paper metadata, study objectives

### 2. DIAGNOSE
- Flag ambiguity or missing fields (e.g. no abstract, unclear modality)
- Check clarity, structure, and schema compliance
- Enforce reproducibility and citation integrity (use `paper_id`)

### 3. DEVELOP
- *Technical tasks:* Use constraint-based prompt logic + JSON schema templates
- *Research idea generation:* Chain-of-thought planning, synthesis, novelty filtering
- *Educational outputs:* Few-shot expansion using user profile + tools
- *Data exploration:* Modality-specific dataset/tool model retrieval

### 4. DELIVER
- Return output ONLY in the required JSON schema
- Fill all required fields, truncate where needed
- Do not hallucinate links or citations
- Use smart defaults for edge cases

## OPTIMIZATION TECHNIQUES
*Core:* role assignment, schema enforcement, modular prompt construction
*Advanced:* chain-of-thought synthesis, thematic clustering, few-shot learning for study plans, fallback repair

## PLATFORM NOTES
- *Groq (LLaMA3):* Optimize for structure precision, hallucination reduction, temperature control
- *Streamlit UI:* Designed for interactive refinement of AI output; short input → schema-rich output

## OPERATING MODES
*INFER MODE:* When only title + abstract are available, deduce context and constraints
*STRUCTURE MODE:* When JSON schema is given, enforce it strictly
*IDEATION MODE:* When prompted for 3 ideas, ensure distinct innovation vectors

## WELCOME MESSAGE
"Hello! I'm Faris, your embedded AI prompt optimizer. I transform your topic into structured, impactful biomedical AI outputs. Just enter a topic or paste a paper — I’ll do the rest
"""

GROQ_HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

REQUEST_TIMEOUT = 30  # seconds


# -----------------------------
# UTILITIES
# -----------------------------


# --- Exponential backoff helper for network/LLM calls ---
def with_backoff(max_tries=3, base=0.8, factor=2.0):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            delay = base
            for attempt in range(1, max_tries + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    if attempt == max_tries:
                        raise
                    time.sleep(delay)
                    delay *= factor
        return wrapper
    return deco

    

# --- Simple relevance score: topic-keyword overlap (name + desc) ---
def relevance_score(item_text: str, topic: str) -> float:
    tks = set(re.findall(r"[a-z0-9]+", topic.lower()))
    it = set(re.findall(r"[a-z0-9]+", (item_text or "").lower()))
    if not tks:
        return 0.0
    overlap = len(tks & it)
    return overlap / math.sqrt(len(tks) + 1)

# --- Modality classifier (heuristic) ---
def classify_modality(name_or_desc: str) -> str:
    s = (name_or_desc or "").lower()
    if any(k in s for k in ["eeg", "mri", "x-ray", "ct", "histo", "microscopy", "radiology", "image"]):
        return "imaging"
    if any(k in s for k in ["genome", "genomic", "rna", "rna-seq", "single-cell", "proteo", "omics", "variant", "vcf", "fasta"]):
        return "genomics"
    if any(k in s for k in ["clinical", "ehr", "emr", "mimic", "uk biobank", "patient", "cohort"]):
        return "clinical"
    if any(k in s for k in ["text", "nlp", "abstracts", "notes", "reports"]):
        return "text"
    return "other"

# --- Accessibility classifier (heuristic by host) ---
def classify_accessibility(link: str) -> str:
    url = (link or "").lower()
    if any(h in url for h in ["huggingface.co", "openml.org", "kaggle.com"]):
        return "public"
    if any(h in url for h in ["gdc.cancer.gov", "ega-archive.org", "dbgap"]):
        return "restricted"
    return "unknown"

# --- Normalize dataset/tool/model item with tags + score ---
def normalize_item(name: str, link: str, desc: str, topic: str) -> dict:
    return {
        "name": name or "Untitled",
        "link": link or "",
        "description": desc or "",
        "modality": classify_modality(f"{name} {desc}"),
        "accessibility": classify_accessibility(link),
        "score": round(relevance_score(f"{name} {desc}", topic), 3)
    }

@with_backoff(max_tries=3, base=0.8, factor=2.0)
def groq_chat(user_content: str, temperature: float = 0.7) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": GLOBAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        "temperature": temperature,
    }
    r = requests.post(GROQ_URL, json=payload, headers=GROQ_HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content']


def try_parse_json(s: str) -> Optional[Any]:
    # Fast path
    try:
        return json.loads(s)
    except Exception:
        pass
    # Extract from fenced block
    m = re.search(r"```json\s*(\{.*?\})\s*```", s, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    # Fallback: first {...}
    m = re.search(r"(\{.*\})", s, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    return None

def infer_paper_id(paper: Dict[str, Any]) -> str:
    link = (paper.get("link") or "").lower()
    m = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", link)
    if m:
        return f"PMID:{m.group(1)}"
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9\.]+)", link)
    if m:
        return f"arXiv:{m.group(1)}"
    title = paper.get("title", "")
    year = str(paper.get("year", ""))
    digest = hashlib.sha1(f"{title}|{year}".encode("utf-8")).hexdigest()[:10]
    return f"HASH:{digest}"

def short_abstract(text: str, max_sentences: int = 3) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    short = ' '.join(sentences[:max_sentences]).strip()
    return short

def to_markdown_download_link(text: str, filename: str, label: str = "📥 Download") -> str:
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:text/markdown;base64,{b64}" download="{filename}">{label}</a>'

def score_sum(idea: Dict[str, Any]) -> float:
    return float(idea.get("impact_score", 0)) + float(idea.get("feasibility_score", 0)) + float(idea.get("novelty_score", 0))
# -----------------------------
# JSON SCHEMAS (Pydantic validation)
# -----------------------------
class IdeaModel(BaseModel):
    title: str
    innovation: str
    methods: List[str] = Field(default_factory=list)
    expected_impact: str
    category: str  # validated downstream; allow "other"
    datasets_or_tools: List[str] = Field(default_factory=list)
    risk_factors: List[str] = Field(default_factory=list)
    feasibility_score: int = Field(ge=1, le=5)
    impact_score: int = Field(ge=1, le=5)
    novelty_score: int = Field(ge=1, le=5)

class IdeasResponse(BaseModel):
    paper_id: str
    ideas: List[IdeaModel]

def validate_ideas_json(data: dict) -> IdeasResponse:
    """
    Validate and normalize the '3 ideas' response from the LLM.
    Raises ValidationError if invalid (we show it nicely in Streamlit).
    """
    try:
        resp = IdeasResponse(**data)
        # Optional: coerce unexpected categories to 'other'
        allowed = {"imaging", "diagnostics", "drug discovery", "bioinformatics", "other"}
        for i in resp.ideas:
            if i.category not in allowed:
                i.category = "other"
        return resp
    except ValidationError as e:
        st.error("Ideas JSON failed validation. See details below.")
        st.code(str(e))
        raise

# -----------------------------
# FETCHERS (cached, robust)
# -----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_tools_with_groq(topic: str) -> List[Dict[str, str]]:
    prompt = f"""
List 5-8 widely used **tools, APIs, or libraries** relevant to the research topic: "{topic}".
Return as JSON:
[
  {{"name":"<tool name>","description":"<what it does>","link":"<url>"}}
]
If you don't know any, return [].
"""
    raw = groq_chat(prompt)
    tools = try_parse_json(raw)
    return tools if tools else []
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_pretrained_models_with_groq(topic: str) -> List[Dict[str, str]]:
    prompt = f"""
List 3-5 popular **pre-trained AI models** (with links) relevant to the topic: "{topic}".
Return as JSON:
[
  {{"name":"<model name>","description":"<short info>","link":"<url>"}}
]
If none found, return [].
"""
    raw = groq_chat(prompt)
    models = try_parse_json(raw)
    return models if models else []

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_pubmed(query: str) -> List[Dict[str, Any]]:
    try:
        search = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": query, "retmode": "json", "retmax": 6},
            timeout=REQUEST_TIMEOUT
        ).json()
        ids = search.get('esearchresult', {}).get('idlist', [])
        if not ids:
            return []
        fetch = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
            timeout=REQUEST_TIMEOUT
        )
        root = ET.fromstring(fetch.text)
        results = []
        for i, article in enumerate(root.findall(".//PubmedArticle")):
            title = article.findtext(".//ArticleTitle", "Untitled")
            abstract = ' '.join([a.text or "" for a in article.findall(".//Abstract/AbstractText")]).strip()
            link = f"https://pubmed.ncbi.nlm.nih.gov/{ids[i]}/"
            year = article.findtext(".//PubDate/Year", "")
            authors = [a.findtext("LastName", "") for a in article.findall(".//Author")]
            results.append({
               "title": title,
               "abstract": abstract,
               "link": link,
               "source": "PubMed",
               "year": int(year) if year.isdigit() else None,
               "authors": ", ".join([a for a in authors if a])
           })
        return results
    except Exception:
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_europe_pmc(query: str, year_from: Optional[int] = None, year_to: Optional[int] = None) -> List[Dict[str, Any]]:
    try:
        base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        date_filter = f" AND FIRST_PDATE:[{year_from} TO {year_to}]" if year_from and year_to else ""
        r = requests.get(base_url, params={"query": query + date_filter, "format": "json", "pageSize": 6}, timeout=REQUEST_TIMEOUT)
        items = r.json().get("resultList", {}).get("result", [])
        return [{
            "title": item.get("title", "Untitled"),
            "abstract": item.get("abstractText", "") or "",
            "link": f"https://europepmc.org/article/{item.get('source', '')}/{item.get('id', '')}",
            "source": "EuropePMC",
            "year": int(item.get("pubYear", 0)) if item.get("pubYear") else None,
            "authors": item.get("authorString", ""),
            "citations": int(item.get("citedByCount", 0)) if item.get("citedByCount") else 0,
        } for item in items]
    except Exception:
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_semantic_scholar(query: str, year_from: Optional[int] = None, citation_min: int = 0) -> List[Dict[str, Any]]:
    try:
        r = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": query, "limit": 20, "fields": "title,abstract,year,url,citationCount,authors"},
            timeout=REQUEST_TIMEOUT
        )
        items = r.json().get("data", [])
        out = []
        for item in items:
            c = item.get("citationCount", 0) or 0
            y = item.get("year", 0) or 0
            if c >= citation_min and (not year_from or y >= year_from):
                out.append({
                    "title": item.get("title", "Untitled"),
                    "abstract": item.get("abstract", "") or "",
                    "link": item.get("url", "") or "",
                    "source": "SemanticScholar",
                    "citations": c,
                    "year": y,
                    "authors": ", ".join(a.get("name", "") for a in item.get("authors", []))
                })
        return out
    except Exception:
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_arxiv(query: str) -> List[Dict[str, Any]]:
    try:
        r = requests.get("http://export.arxiv.org/api/query", params={"search_query": query, "start": 0, "max_results": 6}, timeout=REQUEST_TIMEOUT)
        soup = BeautifulSoup(r.text, "xml")
        entries = soup.find_all("entry")
        return [{
            "title": e.title.text.strip(),
            "abstract": e.summary.text.strip(),
            "link": e.id.text.strip(),
            "source": "arXiv",
            "year": None,
            "authors": ""
        } for e in entries]
    except Exception:
        return []

def source_priority_rank(s: str) -> int:
    order = {"PubMed": 0, "EuropePMC": 1, "SemanticScholar": 2, "arXiv": 3}
    return order.get(s, 99)

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_combined_papers(query: str, sources: Dict[str, bool], year_from: Optional[int] = None, year_to: Optional[int] = None, citation_min: int = 0) -> List[Dict[str, Any]]:
    papers: List[Dict[str, Any]] = []
    if sources.get("PubMed"): papers += fetch_pubmed(query)
    if sources.get("EuropePMC"): papers += fetch_europe_pmc(query, year_from, year_to)
    if sources.get("SemanticScholar"): papers += fetch_semantic_scholar(query, year_from, citation_min)
    if sources.get("arXiv"): papers += fetch_arxiv(query)

    # Deduplicate by title+year or link
    seen = set()
    deduped = []
    for p in papers:
        key = (p.get("title","").lower()[:200], str(p.get("year","")), p.get("link",""))
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    def sort_key(p):
        citations = p.get("citations") or 0
        year = p.get("year") or 0
        priority = source_priority_rank(p.get("source",""))
        return (-citations, -year, priority)

    deduped.sort(key=sort_key)
    return deduped[:9]  # fetch more upfront; we’ll cluster then show top 3 clusters or papers

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_huggingface_datasets(query: str) -> List[Dict[str, str]]:
    try:
        r = requests.get("https://huggingface.co/api/datasets", params={"search": query, "limit": 5}, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return []
        return [{"name": d.get("id", ""), "link": f"https://huggingface.co/datasets/{d.get('id','')}"} for d in r.json()]
    except Exception:
        return []
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_hf_datasets_tagged(topic: str) -> list[dict]:
    try:
        r = requests.get("https://huggingface.co/api/datasets", params={"search": topic, "limit": 12}, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        items = r.json()
        out = []
        for d in items:
            name = d.get("id","")
            link = f"https://huggingface.co/datasets/{name}"
            desc = "HuggingFace dataset"
            out.append(normalize_item(name, link, desc, topic))
        return out
    except Exception:
        return []

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_openml_datasets(query: str) -> List[Dict[str, str]]:
    try:
        r = requests.get(f"https://www.openml.org/api/v1/json/data/list/limit/5/data_name/{query}", timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return []
        data = r.json().get("data", {}).get("dataset", [])
        return [{"name": d.get("name"), "link": f"https://www.openml.org/d/{d.get('did')}"} for d in data]
    except Exception:
        return []
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_openml_datasets_tagged(topic: str) -> list[dict]:
    try:
        r = requests.get(f"https://www.openml.org/api/v1/json/data/list/limit/10/data_name/{topic}", timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return []
        data = r.json().get("data", {}).get("dataset", [])
        out = []
        for d in data:
            name = d.get("name","")
            link = f"https://www.openml.org/d/{d.get('did')}"
            desc = "OpenML dataset"
            out.append(normalize_item(name, link, desc, topic))
        return out
    except Exception:
        return []

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_kaggle_datasets(query: str) -> List[Dict[str, str]]:
    try:
        r = requests.get("https://www.kaggle.com/search", params={"q": query}, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        links = soup.select("a.sc-fqkvVR")
        results = []
        for link in links[:5]:
            href = link.get("href")
            if href and href.startswith("/datasets"):
                name = link.text.strip()
                results.append({"name": name, "link": f"https://www.kaggle.com{href}"})
        return results
    except Exception:
        return []
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_kaggle_datasets_tagged(topic: str) -> list[dict]:
    try:
        r = requests.get("https://www.kaggle.com/search", params={"q": topic}, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        # selector can change; keep robust
        links = soup.select("a[href^='/datasets']")
        out = []
        for a in links[:12]:
            href = a.get("href","")
            text = a.get_text(strip=True)
            link = f"https://www.kaggle.com{href}"
            out.append(normalize_item(text, link, "Kaggle dataset", topic))
        return out
    except Exception:
        return []
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_hf_models_tagged(topic: str) -> list[dict]:
    try:
        r = requests.get("https://huggingface.co/api/models", params={"search": topic, "limit": 12}, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        models = r.json()
        out = []
        for m in models:
            name = m.get("id","")
            # Try to derive a short description
            card = m.get("cardData") or {}
            desc = card.get("summary") or card.get("language") or "HuggingFace model"
            link = f"https://huggingface.co/{name}"
            out.append(normalize_item(name, link, desc, topic))
        return out
    except Exception:
        return []
    

def fetch_tools_llm_json(topic: str) -> list[dict]:
    prompt = f"""
List 6–10 commonly used **tools, APIs, or libraries** relevant to the research topic: "{topic}".
Return a JSON array of objects with EXACTLY these fields:
[
  {{"name":"<tool name>","description":"<what it does>","link":"<url>"}}
]
If unsure, return [] and do not add extra fields or text.
"""
    raw = groq_chat(prompt, temperature=0.5)
    data = try_parse_json(raw) or []
    # Tag + score
    out = []
    for t in data[:12]:
        out.append(normalize_item(t.get("name",""), t.get("link",""), t.get("description",""), topic))
    return out
def search_datasets_tools_models(topic: str) -> tuple[list[dict], list[dict], list[dict]]:
    funcs = [
        ("hf_datasets", fetch_hf_datasets_tagged),
        ("openml", fetch_openml_datasets_tagged),
        ("kaggle", fetch_kaggle_datasets_tagged),
        ("hf_models", fetch_hf_models_tagged),
    ]
    datasets, models = [], []

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(f, topic): name for name, f in funcs}
        for fut in concurrent.futures.as_completed(futs):
            name = futs[fut]
            try:
                res = fut.result() or []
                if name in ("hf_models",):
                    models += res
                else:
                    datasets += res
            except Exception:
                pass

    tools = fetch_tools_llm_json(topic)  # LLM call separate (fast enough)
    # Deduplicate by name+link
    def dedupe(items):
        seen, out = set(), []
        for it in items:
            key = (it.get("name","").lower(), it.get("link","").lower())
            if key not in seen:
                seen.add(key)
                out.append(it)
        return out

    datasets = dedupe(datasets)
    tools = dedupe(tools)
    models = dedupe(models)

    # Sort by score desc, then alphabetically
    datasets.sort(key=lambda x: (x.get("score",0), x.get("name","").lower()), reverse=True)
    tools.sort(key=lambda x: (x.get("score",0), x.get("name","").lower()), reverse=True)
    models.sort(key=lambda x: (x.get("score",0), x.get("name","").lower()), reverse=True)

    return datasets, tools, models

def fetch_datasets(query: str) -> List[Dict[str, str]]:
    datasets = []
    datasets += fetch_huggingface_datasets(query)
    datasets += fetch_openml_datasets(query)
    datasets += fetch_kaggle_datasets(query)
    return datasets

def build_graphviz(papers: List[Dict[str, str]], query: str) -> str:
    lines = ["digraph G {", "rankdir=LR;"]
    query_node = query.replace('"', '')
    lines.append(f'"{query_node}" [shape=box, style=filled, color=lightblue];')
    for paper in papers[:10]:
        title = paper.get("title", "").replace('"', '')
        lines.append(f'"{query_node}" -> "{title}";')
    lines.append("}")
    return "\n".join(lines)

# -----------------------------
# LLM TASKS (JSON-based)
# -----------------------------
def ideas_from_paper_json(paper: Dict[str, Any]) -> Dict[str, Any]:
    paper_id = infer_paper_id(paper)
    constraints = {
        "max_words_per_idea": 160,
        "prioritize_categories": ["imaging", "diagnostics", "drug discovery", "bioinformatics"],
        "novelty_requirement": "distinct from the paper’s contributions; propose a new angle or integration"
    }
    user_prompt = f"""
Task: Create **3 novel and feasible research project ideas** from the given paper.

Inputs:
- paper: {{
    "paper_id": "{paper_id}",
    "title": {json.dumps(paper.get("title", ""))},
    "abstract": {json.dumps(paper.get("abstract", ""))},
    "year": {json.dumps(paper.get("year", None))},
    "link": {json.dumps(paper.get("link", ""))},
    "authors": {json.dumps(paper.get("authors", ""))}
}}
- constraints: {json.dumps(constraints)}

Output JSON schema:
{{
  "paper_id": "<string>",
  "ideas": [
    {{
      "title": "<string>",
      "innovation": "<1-2 sentences>",
      "methods": ["<AI/biomedical method>", "..."],
      "expected_impact": "<1-2 sentences>",
      "category": "<one of: imaging|drug discovery|diagnostics|bioinformatics|other>",
      "datasets_or_tools": ["<dataset/tool>", "..."],
      "risk_factors": ["<biggest risks>", "..."],
      "feasibility_score": <1-5>,
      "impact_score": <1-5>,
      "novelty_score": <1-5>
    }},
    ...
  ]
}}

Rules:
- Return exactly 3 ideas.
- Each idea must be **materially different**.
- If datasets/tools are uncertain, return an empty array.
- Keep each idea under constraints.max_words_per_idea.
- Use the given paper_id in the top-level output.

Return JSON ONLY. No prose.
""".strip()
    raw = groq_chat(user_prompt, temperature=0.7)
    parsed = try_parse_json(raw)
    if parsed is None:
        raw = groq_chat(user_prompt + "\n\nReminder: return ONLY valid JSON.", temperature=0.6)
        parsed = try_parse_json(raw)
    if parsed is None:
        raise ValueError("Ideas: invalid JSON")

    if parsed.get("paper_id") is None:
        parsed["paper_id"] = paper_id
    ideas = parsed.get("ideas", [])
    if not isinstance(ideas, list) or len(ideas) != 3:
        ideas = (ideas or [])[:3]
        while len(ideas) < 3:
            ideas.append({
                "title": "Placeholder idea",
                "innovation": "",
                "methods": [],
                "expected_impact": "",
                "category": "other",
                "datasets_or_tools": [],
                "risk_factors": [],
                "feasibility_score": 3,
                "impact_score": 3,
                "novelty_score": 3
            })
        parsed["ideas"] = ideas

    # ✅ ADD THESE TWO LINES:
    validated = validate_ideas_json(parsed)        # will raise + show details if invalid
    return validated.model_dump()                  # return as plain dict for the rest of your app

def cross_paper_synthesis_json(papers: List[Dict[str, Any]]) -> Dict[str, Any]:
    minimal = [{"paper_id": infer_paper_id(p), "title": p.get("title",""), "year": p.get("year", None), "abstract": p.get("abstract",""), "link": p.get("link","")} for p in papers]
    user_prompt = f"""
Task: Synthesize insights across multiple papers.

Inputs:
- papers: {json.dumps(minimal)}

Output JSON schema:
{{
  "themes": [{{"name":"<string>","evidence_from":["<paper_id>", "..."]}}],
  "contradictions": [{{"description":"<string>","papers_in_conflict":["<paper_id>","<paper_id>"]}}],
  "unified_direction": "<2-4 sentences proposing an integrated path forward>",
  "methods_convergence": ["<method>","<method>"],
  "open_questions": ["<question>","<question>"]
}}

Rules:
- Cite papers only via `paper_id`.
- If a section has no content, return an empty array.
- Return JSON only.
""".strip()
    raw = groq_chat(user_prompt, temperature=0.5)
    parsed = try_parse_json(raw)
    if parsed is None:
        raw = groq_chat(user_prompt + "\n\nReminder: return ONLY valid JSON.", temperature=0.4)
        parsed = try_parse_json(raw)
    if parsed is None:
        raise ValueError("Synthesis: invalid JSON")
    return parsed

def research_gaps_json(papers: List[Dict[str, Any]]) -> Dict[str, Any]:
    minimal = [{"paper_id": infer_paper_id(p), "title": p.get("title",""), "year": p.get("year", None), "abstract": p.get("abstract",""), "link": p.get("link","")} for p in papers]
    user_prompt = f"""
Task: Identify research gaps across these papers.

Inputs:
- papers: {json.dumps(minimal)}

Output JSON schema:
{{
  "neglected_populations": ["<group>", "..."],
  "unstudied_conditions": ["<condition>", "..."],
  "underused_methods": ["<method>", "..."],
  "data_limitations": ["<issue>", "..."],
  "ethical_or_bias_risks": ["<risk>", "..."],
  "top_5_actionable_gap_ideas": [
    {{
      "title":"<string>",
      "why_it_matters":"<1 sentence>",
      "minimal_viable_study":"<2-3 steps>",
      "paper_support":["<paper_id>", "..."]
    }}
  ]
}}

Rules:
- Max 5 actionable gap ideas.
- Use `paper_id` for support.
- Return JSON only.
""".strip()
    raw = groq_chat(user_prompt, temperature=0.5)
    parsed = try_parse_json(raw)
    if parsed is None:
        raw = groq_chat(user_prompt + "\n\nReminder: return ONLY valid JSON.", temperature=0.4)
        parsed = try_parse_json(raw)
    if parsed is None:
        raise ValueError("Gaps: invalid JSON")
    return parsed

def expand_idea_json(idea: Dict[str, Any], papers: List[Dict[str, Any]], target_compute="mixed", expected_scale="pilot") -> Dict[str, Any]:
    ctx_papers = [{"paper_id": infer_paper_id(p), "title": p.get("title",""), "year": p.get("year", None), "link": p.get("link","")} for p in papers]
    user_prompt = f"""
Task: Expand a seed idea with technical depth.

Inputs:
- idea: {json.dumps(idea)}
- context: {{
    "papers": {json.dumps(ctx_papers)},
    "target_compute": "{target_compute}",
    "expected_scale": "{expected_scale}"
}}

Output JSON schema:
{{
  "refined_objectives": ["<objective>", "..."],
  "method_pipeline_steps": ["<step 1>","<step 2>","<step 3>"],
  "evaluation_metrics": ["<metric>","<metric>"],
  "baseline_vs_sota": {{"baseline":"<string>","sota":"<string>"}},
  "resources": {{"datasets":["<name>"],"tools":["<tool>"]}},
  "risks_and_mitigations": [{{"risk":"<string>","mitigation":"<string>"}}],
  "timeline_weeks": [{{"week":"<1-2>","milestone":"<string>"}}],
  "citations":["<paper_id>","..."]
}}

Rules:
- Tailor pipeline to `target_compute` and `expected_scale`.
- Return JSON only.
""".strip()
    raw = groq_chat(user_prompt, temperature=0.6)
    parsed = try_parse_json(raw)
    if parsed is None:
        raw = groq_chat(user_prompt + "\n\nReminder: return ONLY valid JSON.", temperature=0.5)
        parsed = try_parse_json(raw)
    if parsed is None:
        raise ValueError("Expand idea: invalid JSON")
    return parsed

def study_plan_json(idea_title: str, student_profile: Dict[str, Any], weeks: int = 14) -> Dict[str, Any]:
    user_prompt = f"""
Task: Produce a {weeks}-week independent study plan.

Inputs:
- idea_title: {json.dumps(idea_title)}
- student_profile: {json.dumps(student_profile)}
- constraints: {{"weeks": {weeks}, "deliverables_required": true}}

Output JSON schema:
{{
  "title":"<string>",
  "duration_weeks": <int>,
  "supervisor_profile":"<1-2 lines>",
  "background_and_rationale":"<short>",
  "objectives":["<objective>", "..."],
  "deliverables":[{{"name":"<string>","due_week":<int>}}],
  "tools_and_tech":["<tool>", "..."],
  "weekly_timeline":[{{"week":<int>,"focus":"<string>","outputs":["<o1>","<o2>"]}}],
  "assessment_rubric":{{"technical":40,"writing":30,"presentation":30}}
}}

Rules:
- Calibrate to student_profile.skills.
- At least 3 concrete deliverables.
- Return JSON only.
""".strip()
    raw = groq_chat(user_prompt, temperature=0.6)
    parsed = try_parse_json(raw)
    if parsed is None:
        raw = groq_chat(user_prompt + "\n\nReminder: return ONLY valid JSON.", temperature=0.5)
        parsed = try_parse_json(raw)
    if parsed is None:
        raise ValueError("Study plan: invalid JSON")
    return parsed

def study_plan_json_to_markdown(plan: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"# {plan.get('title','Independent Study Plan')}")
    lines.append(f"**Duration:** {plan.get('duration_weeks','?')} weeks")
    if sp := plan.get("supervisor_profile"):
        lines.append(f"**Supervisor:** {sp}")
    if bg := plan.get("background_and_rationale"):
        lines.append(f"\n## Background and Rationale\n{bg}")
    if objs := plan.get("objectives", []):
        lines.append("\n## Objectives")
        lines += [f"- {o}" for o in objs]
    if tools := plan.get("tools_and_tech", []):
        lines.append("\n## Tools & Technologies")
        lines += [f"- {t}" for t in tools]
    if dels := plan.get("deliverables", []):
        lines.append("\n## Deliverables")
        for d in dels:
            lines.append(f"- **{d.get('name','Deliverable')}** — Due week {d.get('due_week','?')}")
    if weeks := plan.get("weekly_timeline", []):
        lines.append("\n## Weekly Timeline")
        for w in weeks:
            outs = ", ".join(w.get("outputs", []))
            lines.append(f"- **Week {w.get('week','?')}:** {w.get('focus','')} (Outputs: {outs})")
    if rubric := plan.get("assessment_rubric"):
        lines.append("\n## Assessment Rubric")
        for k, v in rubric.items():
            lines.append(f"- {k.capitalize()}: {v}%")
    return "\n".join(lines)

def deep_study_summary_json(study_text: str) -> Dict[str, Any]:
    # Prevent payload overflow
    if len(study_text) > 12000:
        study_text = study_text[:12000]
    user_prompt = f"""
Task: Produce a deep structured summary from full study content.

Inputs:
- study_text: {json.dumps(study_text)}

Output JSON schema:
{{
  "title":"<string>",
  "background":"<2-4 sentences>",
  "objective":"<1-2 sentences>",
  "methodology":["<step>","<step>","<step>"],
  "figures_and_results":["<result>","<result>"],
  "core_findings":["<finding>","<finding>"],
  "limitations":["<limitation>","<limitation>"],
  "next_steps":["<experiment or analysis>","<...>"]
}}

Return JSON only.
""".strip()
    raw = groq_chat(user_prompt, temperature=0.5)
    parsed = try_parse_json(raw)
    if parsed is None:
        raw = groq_chat(user_prompt + "\n\nReminder: return ONLY valid JSON.", temperature=0.4)
        parsed = try_parse_json(raw)
    if parsed is None:
        raise ValueError("Deep study summary: invalid JSON")
    return parsed

def chat_answer_json(context: Dict[str, Any], question: str) -> Dict[str, Any]:
    user_prompt = f"""
Role: Answer user questions grounded only in the provided content.

Inputs:
- context: {json.dumps(context)}
- question: {json.dumps(question)}

Output JSON schema:
{{
  "answer":"<concise, helpful answer>",
  "supporting_points":[{{"quote_or_fact":"<string>","origin":"<paper_id or section>"}}],
  "uncertainties":["<what is unknown>"]
}}

Return JSON only.
""".strip()
    raw = groq_chat(user_prompt, temperature=0.5)
    parsed = try_parse_json(raw)
    if parsed is None:
        raw = groq_chat(user_prompt + "\n\nReminder: return ONLY valid JSON.", temperature=0.4)
        parsed = try_parse_json(raw)
    if parsed is None:
        raise ValueError("Chat answer: invalid JSON")
    return parsed

# -----------------------------
# CONTENT FETCH (single study)
# -----------------------------
def fetch_full_paper_content(url: str) -> str:
    """Fetch as much textual content as possible from a paper link."""
    if not url:
        raise ValueError("Empty URL.")
    if "pubmed" in url:
        match = re.search(r"(\d+)", url)
        if not match:
            raise ValueError("Invalid PubMed link")
        pmid = match.group(1)
        r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                         params={"db": "pubmed", "id": pmid, "retmode": "xml"}, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        title = root.findtext(".//ArticleTitle", "")
        abstract = " ".join(a.text or "" for a in root.findall(".//AbstractText"))
        return f"Title: {title}\n\nAbstract: {abstract}"
    if "arxiv" in url:
        match = re.search(r"arxiv.org/(?:abs|pdf)/([^?/]+)", url)
        if not match:
            raise ValueError("Invalid arXiv link")
        arxiv_id = match.group(1).replace(".pdf", "")
        r = requests.get("http://export.arxiv.org/api/query", params={"id_list": arxiv_id}, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "xml")
        entry = soup.find("entry")
        if not entry:
            raise ValueError("arXiv entry not found")
        title = entry.title.text.strip()
        abstract = entry.summary.text.strip()
        return f"Title: {title}\n\nAbstract: {abstract}"
    # Generic HTML fetch
    r = requests.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    title = soup.title.text.strip() if soup.title else ""
    paragraphs = " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p")[:15])  # Only first 15 paragraphs
    captions = " ".join(c.get_text(" ", strip=True) for c in soup.find_all("figcaption"))
    return f"Title: {title}\n\n{paragraphs}\n\n{captions}"

# -----------------------------
# CLUSTERING (optional, with fallback)
# -----------------------------
def embed_texts(texts: List[str]) -> Optional[List[List[float]]]:
    if not EMBEDDING_AVAILABLE:
        return None
    try:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embs = model.encode(texts, normalize_embeddings=True).tolist()
        return embs
    except Exception:
        return None

def cluster_papers(papers: List[Dict[str, Any]], desired_k: int = 3) -> List[List[Dict[str, Any]]]:
    """
    Returns clusters of papers (list of lists). If embeddings not available,
    falls back to first N buckets.
    """
    if len(papers) <= desired_k or not EMBEDDING_AVAILABLE:
        # simple buckets
        buckets = [[] for _ in range(min(desired_k, len(papers)) or 1)]
        for idx, p in enumerate(papers):
            buckets[idx % len(buckets)].append(p)
        return [b for b in buckets if b]

    abs_texts = [p.get("abstract","")[:3000] for p in papers]
    embs = embed_texts(abs_texts)
    if embs is None:
        # fallback
        buckets = [[] for _ in range(min(desired_k, len(papers)) or 1)]
        for idx, p in enumerate(papers):
            buckets[idx % len(buckets)].append(p)
        return [b for b in buckets if b]

    k = min(desired_k, max(1, len(papers)//2))  # conservative default
    try:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(embs)
        clusters: Dict[int, List[Dict[str, Any]]] = {}
        for lab, p in zip(labels, papers):
            clusters.setdefault(int(lab), []).append(p)
        return list(clusters.values())
    except Exception:
        # fallback simple split
        buckets = [[] for _ in range(min(desired_k, len(papers)) or 1)]
        for idx, p in enumerate(papers):
            buckets[idx % len(buckets)].append(p)
        return [b for b in buckets if b]

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="AI Research Discovery Tool", layout="wide")
st.title("🌐 AI Research Discovery Tool (Multi-Source)")

page_opts = ["🎯 Generate Project Ideas", "📚 Deep Study Analyzer", "🔍 Dataset & Tool Finder"]

if "section" not in st.session_state:
    st.session_state.section = page_opts[0]

st.session_state.section = st.sidebar.radio("Navigate", page_opts, index=page_opts.index(st.session_state.section))

# State init
if "results" not in st.session_state:
    st.session_state.results = []
if "graph" not in st.session_state:
    st.session_state.graph = ""
if "datasets" not in st.session_state:
    st.session_state.datasets = []
if "cross_synthesis" not in st.session_state:
    st.session_state.cross_synthesis = {}
if "research_gaps" not in st.session_state:
    st.session_state.research_gaps = {}
if "study_plans" not in st.session_state:
    st.session_state.study_plans = {}
if "current_study_plan" not in st.session_state:
    st.session_state.current_study_plan = ""
if "ideas_chat_history" not in st.session_state:
    st.session_state.ideas_chat_history = []
if "study_summary_json" not in st.session_state:
    st.session_state.study_summary_json = {}
if "study_chat_history" not in st.session_state:
    st.session_state.study_chat_history = []

# -----------------------------
# Generate Project Ideas Page
# -----------------------------

if st.session_state.section == "🎯 Generate Project Ideas":
    query = st.text_input("Enter a biomedical or AI-related research topic:", "epilepsy EEG")
    sources = {
        "PubMed": st.checkbox("PubMed", value=True),
        "EuropePMC": st.checkbox("EuropePMC", value=True),
        "SemanticScholar": st.checkbox("SemanticScholar", value=True),
        "arXiv": st.checkbox("arXiv", value=True)
    }

    with st.expander("🔧 Advanced Filters"):
        year_from = st.number_input("From Year", min_value=2000, max_value=datetime.datetime.now().year, value=2020)
        year_to = st.number_input("To Year", min_value=2000, max_value=datetime.datetime.now().year, value=datetime.datetime.now().year)
        citation_min = st.slider("Minimum Citation Count (Semantic Scholar only)", 0, 1000, 10)
        clustering_on = st.checkbox("Cluster similar studies (recommended)", value=True and EMBEDDING_AVAILABLE)
        desired_clusters = st.slider("Desired clusters", 2, 5, 3)

    def generate_clustered_results():
        with st.spinner("🔎 Fetching, clustering, and generating ideas..."):
            try:
                all_papers = fetch_combined_papers(query, sources, year_from, year_to, citation_min)
                if not all_papers:
                    st.warning("No results found.")
                    st.session_state.results = []
                    st.session_state.graph = ""
                    st.session_state.datasets = []
                    return

                # Cluster (or bucket) papers
                clusters = cluster_papers(all_papers, desired_k=desired_clusters) if clustering_on else [all_papers[:3]]

                results = []
                for idx, cluster in enumerate(clusters):
                    # Synthesis per cluster
                    try:
                        synth = cross_paper_synthesis_json(cluster)
                    except Exception:
                        synth = {}

                    # Build a pseudo paper representing the cluster (for idea generation)
                    joined_titles = "; ".join([p.get("title","") for p in cluster[:3]])
                    joined_abstracts = "\n\n".join([p.get("abstract","") for p in cluster])
                    pseudo_paper = {
                        "title": f"Cluster {idx+1}: {joined_titles[:180]}",
                        "abstract": joined_abstracts[:6000],
                        "link": "",
                        "source": "Cluster",
                        "year": None,
                        "authors": ""
                    }

                    # Ideas per cluster (from combined abstracts)
                    try:
                        ideas_json = ideas_from_paper_json(pseudo_paper)
                    except Exception as e:
                        ideas_json = {"paper_id": f"CLUSTER:{idx+1}", "ideas": []}

                    results.append({
                        "cluster_index": idx + 1,
                        "cluster_papers": cluster,
                        "synthesis_json": synth,
                        "ideas_json": ideas_json
                    })

                # Rank clusters by best idea score
                for r in results:
                    r["best_score"] = max([score_sum(i) for i in r["ideas_json"].get("ideas", [])] + [0])
                results.sort(key=lambda r: r["best_score"], reverse=True)

                # Keep top 3 clusters for UI
                st.session_state.results = results[:3]
                st.session_state.graph = build_graphviz(all_papers, query)
                st.session_state.datasets = fetch_datasets(query)

            except Exception as e:
                st.error(f"Error: {e}")

    if st.button("🎯 Generate Project Ideas"):
        generate_clustered_results()

    # Render results
    if st.session_state.results:
        md_export_chunks = []

        for r in st.session_state.results:
            cidx = r["cluster_index"]
            cluster_papers = r["cluster_papers"]
            synth = r.get("synthesis_json", {})
            ideas_json = r.get("ideas_json", {})
            # Try to infer a theme from the synthesis JSON (if available)
            theme_name = None
            if r.get("synthesis_json", {}).get("themes"):
                theme_name = r["synthesis_json"]["themes"][0].get("name")

            title_label = f"## 🔷 Research Theme {cidx}"
            if theme_name:
                title_label = f"## 🔷 {theme_name}"

            st.markdown(title_label)

            # Papers toggle: rename the label
            with st.expander("📄 Supporting Papers"):

                for p in cluster_papers:
                    y = f" ({p.get('year')})" if p.get("year") else ""
                    st.markdown(f"- **{p.get('title','Untitled')}**{y} — [Link]({p.get('link','')})")

            if synth:
                st.markdown("### 🧠 Cross-Paper Synthesis")
                themes = synth.get("themes", [])
                if themes:
                    st.markdown("**Themes:**")
                    for t in themes:
                        st.markdown(f"- {t.get('name','')}")
                contr = synth.get("contradictions", [])
                if contr:
                    st.markdown("**Contradictions:**")
                    for c in contr:
                        st.markdown(f"- {c.get('description','')}")
                if ud := synth.get("unified_direction"):
                    st.markdown(f"**Unified Direction:** {ud}")
                if mc := synth.get("methods_convergence", []):
                    st.markdown(f"**Methods convergence:** {', '.join(mc)}")
                if oq := synth.get("open_questions", []):
                    st.markdown("**Open questions:** " + "; ".join(oq))

            st.markdown("### 💡 Top Ideas for this Research")
            ideas_list = ideas_json.get("ideas", [])
            ideas_list = sorted(ideas_list, key=score_sum, reverse=True)

            for j, idea in enumerate(ideas_list[:3]):
                st.markdown(f"**{j+1}. {idea.get('title','Untitled')}**  \n*Innovation:* {idea.get('innovation','')}")
                st.markdown(f"*Category:* `{idea.get('category','other')}` • *Impact:* {idea.get('expected_impact','')}")
                methods = ", ".join(idea.get("methods", []) or [])
                if methods: st.markdown(f"*Methods:* {methods}")
                ds = idea.get("datasets_or_tools", [])
                if ds: st.markdown(f"*Datasets/Tools:* {', '.join(ds)}")
                risks = idea.get("risk_factors", [])
                if risks: st.markdown(f"*Risks:* {'; '.join(risks)}")
                st.markdown(f"**Scores:** Feasibility {idea.get('feasibility_score',0)}/5 • Impact {idea.get('impact_score',0)}/5 • Novelty {idea.get('novelty_score',0)}/5")

                if st.button("📘 Generate Independent Study Plan", key=f"plan_{cidx}_{j}"):
                    # Minimal idea dict passed to plan
                    idea_md = f"Title: {idea.get('title','')}\nInnovation: {idea.get('innovation','')}\nMethods: {methods}\nExpected impact: {idea.get('expected_impact','')}\nCategory: {idea.get('category','other')}"
                    student_profile = {
                        "skills": ["Python", "MATLAB", "C++"],  # adjust as needed or add a sidebar later
                        "background": "Biomedical Engineering"
                    }
                    try:
                        plan_json = study_plan_json(idea_title=idea.get('title',''), student_profile=student_profile, weeks=14)
                        plan_md = study_plan_json_to_markdown(plan_json)
                    except Exception:
                        # Fallback: simple text if JSON fails
                        plan_md = idea_md + "\n\n(Study plan generation failed.)"
                    st.session_state.current_study_plan = plan_md
                    st.session_state.page = "study_plan_view"
                    st.session_state.section = "🎯 Generate Project Ideas"
                    st.rerun()

            # Markdown export chunk
            theme_name = None
            if r.get("synthesis_json", {}).get("themes"):
                theme_name = r["synthesis_json"]["themes"][0].get("name")
            chunk_title = f"## Research Theme {cidx}" if not theme_name else f"## {theme_name}"
            chunk = [chunk_title, "### Papers"]

            for p in cluster_papers:
                y = f" ({p.get('year')})" if p.get("year") else ""
                chunk.append(f"- {p.get('title','Untitled')}{y} — {p.get('link','')}")
            chunk.append("\n### Top Ideas")
            for idea in ideas_list[:3]:
                chunk.append(f"- **{idea.get('title','Untitled')}** — {idea.get('innovation','')}")
            md_export_chunks.append("\n".join(chunk))

        # Export link
        md_export = "\n\n---\n\n".join(md_export_chunks)
        st.markdown(to_markdown_download_link(md_export, "project_ideas_clusters.md", "📥 Download Results as Markdown"), unsafe_allow_html=True)

        # Graphviz
        if st.session_state.graph:
            st.graphviz_chart(st.session_state.graph)

        # Datasets
        with st.expander("📊 Suggested Datasets"):
            if st.session_state.datasets:
                for d in st.session_state.datasets:
                    st.markdown(f"- [{d['name']}]({d['link']})")
            else:
                st.write("No datasets found.")

        # Gaps button
        if st.button("🕳️ Find the Research Gaps (from reasearches)"):
            all_cluster_papers: List[Dict[str, Any]] = []
            for r in st.session_state.results:
                all_cluster_papers += r["cluster_papers"]
            try:
                st.session_state.research_gaps = research_gaps_json(all_cluster_papers)
            except Exception as e:
                st.error(f"Error generating gaps: {e}")

        if st.session_state.research_gaps:
            gaps = st.session_state.research_gaps
            st.subheader("🕳️ Research Gaps and Niche Opportunities")
            if gp := gaps.get("neglected_populations", []):
                st.markdown("**Neglected populations:** " + ", ".join(gp))
            if uc := gaps.get("unstudied_conditions", []):
                st.markdown("**Unstudied conditions:** " + ", ".join(uc))
            if um := gaps.get("underused_methods", []):
                st.markdown("**Underused methods:** " + ", ".join(um))
            if dl := gaps.get("data_limitations", []):
                st.markdown("**Data limitations:** " + "; ".join(dl))
            if eb := gaps.get("ethical_or_bias_risks", []):
                st.markdown("**Ethical/bias risks:** " + "; ".join(eb))
            if tops := gaps.get("top_5_actionable_gap_ideas", []):
                st.markdown("### Top Actionable Gap Ideas")
                for g in tops:
                    st.markdown(f"- **{g.get('title','')}** — {g.get('why_it_matters','')}")
                    st.markdown(f"  - Minimal viable study: {g.get('minimal_viable_study','')}")
                    if g.get("paper_support"):
                        st.markdown(f"  - Support: {', '.join(g.get('paper_support'))}")

        # Ideas chat
        st.subheader("💬 Ask about these ideas")
        user_msg = st.chat_input("Type your question about the ideas")
        if user_msg:
            # Build context from ideas shown
            ctx = {"ideas": []}
            for r in st.session_state.results:
                for idea in r.get("ideas_json", {}).get("ideas", [])[:3]:
                    ctx["ideas"].append(idea)
            try:
                ans_json = chat_answer_json(context={"ideas": ctx["ideas"]}, question=user_msg)
                st.session_state.ideas_chat_history.append({"q": user_msg, "a": ans_json})
            except Exception as e:
                st.error(f"Error: {e}")

        for ch in st.session_state.ideas_chat_history:
            st.markdown(f"**Q:** {ch['q']}")
            a = ch.get("a", {})
            st.markdown(f"**A:** {a.get('answer','')}")
            sp = a.get("supporting_points", [])
            if sp:
                with st.expander("Supporting points"):
                    for pt in sp:
                        st.markdown(f"- {pt.get('quote_or_fact','')} *(from {pt.get('origin','')})*")
            un = a.get("uncertainties", [])
            if un:
                st.caption("Uncertainties: " + "; ".join(un))

# -----------------------------
# Study Plan Viewer (from Ideas page)
# -----------------------------
if "page" in st.session_state and st.session_state.page == "study_plan_view":
    st.title("📘 Independent Study Plan")
    st.markdown(st.session_state.get("current_study_plan", ""))
    plan_text = st.session_state.get("current_study_plan", "")
    if plan_text:
        st.markdown(to_markdown_download_link(plan_text, "study_plan.md", "📥 Download Study Plan"), unsafe_allow_html=True)
    if st.button("⬅️ Back"):
        st.session_state.page = None
        st.rerun()

# -----------------------------
# Deep Study Analyzer Page
# -----------------------------
if st.session_state.section == "📚 Deep Study Analyzer":
    st.sidebar.title("📂 Pages")

    if "study_input_url" not in st.session_state:
        st.session_state.study_input_url = ""
    st.session_state.study_input_url = st.text_input("Paste a paper link (PubMed, arXiv, etc.)", value=st.session_state.study_input_url)

    if st.button("📚 Deep Summarize"):
        with st.spinner("📚 Summarizing study..."):
            try:
                content = fetch_full_paper_content(st.session_state.study_input_url)
                st.session_state.study_summary_json = deep_study_summary_json(content)
                st.session_state.study_chat_history = []
            except Exception as e:
                st.session_state.study_summary_json = {}
                st.error(f"Could not retrieve or summarize content. {e}")

    if st.session_state.study_summary_json:
        s = st.session_state.study_summary_json
        st.subheader("📚 Deep Study Summary")
        st.markdown(f"### {s.get('title','')}")
        if bg := s.get("background"): st.markdown(f"**Background:** {bg}")
        if ob := s.get("objective"): st.markdown(f"**Objective:** {ob}")
        if m := s.get("methodology", []):
            st.markdown("**Methodology:**")
            for step in m: st.markdown(f"- {step}")
        if fr := s.get("figures_and_results", []):
            st.markdown("**Key Figures/Results:**")
            for r in fr: st.markdown(f"- {r}")
        if cf := s.get("core_findings", []):
            st.markdown("**Core Findings:**")
            for r in cf: st.markdown(f"- {r}")
        if lm := s.get("limitations", []):
            st.markdown("**Limitations:**")
            for r in lm: st.markdown(f"- {r}")
        if nx := s.get("next_steps", []):
            st.markdown("**Next Steps:**")
            for r in nx: st.markdown(f"- {r}")

        st.subheader("💬 Chat About This Study")
        user_q = st.text_input("Ask about this study", key="study_chat_input")
        if st.button("Send Question"):
            if user_q:
                with st.spinner("💬 Generating answer..."):
                    try:
                        # Provide the JSON summary as context
                        ans = chat_answer_json(context={"study_summary": st.session_state.study_summary_json}, question=user_q)
                        st.session_state.study_chat_history.append({"q": user_q, "a": ans})
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please enter a question.")

        for chat in st.session_state.study_chat_history:
            st.markdown(f"**Q:** {chat['q']}")
            a = chat.get("a", {})
            st.markdown(f"**A:** {a.get('answer','')}")
            sp = a.get("supporting_points", [])
            if sp:
                with st.expander("Supporting points"):
                    for pt in sp:
                        st.markdown(f"- {pt.get('quote_or_fact','')} *(from {pt.get('origin','')})*")
            un = a.get("uncertainties", [])
            if un:
                st.caption("Uncertainties: " + "; ".join(un))
                
if st.session_state.section == "🔍 Dataset & Tool Finder":
    st.subheader("🔍 Dataset & Tool Finder")
    topic = st.text_input("Enter a research topic:", "cancer genomics")

    with st.expander("🔧 Filters"):
        modal_filter = st.multiselect("Modality", ["genomics","imaging","clinical","text","other"], default=[])
        access_filter = st.multiselect("Accessibility", ["public","restricted","unknown"], default=[])
        min_score = st.slider("Min relevance score", 0.0, 1.0, 0.15, 0.05)

    if st.button("🔎 Find Datasets, Tools & Models"):
        with st.spinner("Searching across dataset hubs, tools and model repos..."):
            ds, tools, models = search_datasets_tools_models(topic)
            st.session_state.dtf_results = {"datasets": ds, "tools": tools, "models": models, "topic": topic}

    res = st.session_state.get("dtf_results", {})
    if res:
        topic = res.get("topic","")
        datasets = res.get("datasets", [])
        tools = res.get("tools", [])
        models = res.get("models", [])

        def apply_filters(items):
            out = []
            for it in items:
                if modal_filter and it.get("modality") not in modal_filter:
                    continue
                if access_filter and it.get("accessibility") not in access_filter:
                    continue
                if float(it.get("score",0)) < min_score:
                    continue
                out.append(it)
            return out

        datasets_f = apply_filters(datasets)
        tools_f = apply_filters(tools)
        models_f = apply_filters(models)

        # --- Datasets ---
        st.markdown("## 📊 Datasets")
        if datasets_f:
            for d in datasets_f:
                st.markdown(
                    f"- **[{d['name']}]({d['link']})** "
                    f"· *{d['modality']}* · *{d['accessibility']}* · score **{d['score']}**  \n"
                    f"  {d['description']}"
                )
        else:
            st.info("No datasets matched your filters.")

        # --- Tools/APIs ---
        st.markdown("## 🛠️ Tools & APIs")
        if tools_f:
            for t in tools_f:
                st.markdown(
                    f"- **[{t['name']}]({t['link']})** "
                    f"· *{t['modality']}* · *{t['accessibility']}* · score **{t['score']}**  \n"
                    f"  {t['description']}"
                )
        else:
            st.info("No tools/APIs matched your filters.")

        # --- Pre-trained Models ---
        st.markdown("## 🤖 Pre-trained Models")
        if models_f:
            for m in models_f:
                st.markdown(
                    f"- **[{m['name']}]({m['link']})** "
                    f"· *{m['modality']}* · *{m['accessibility']}* · score **{m['score']}**  \n"
                    f"  {m['description']}"
                )
        else:
            st.info("No models matched your filters.")

        # --- Export ---
        if st.button("📥 Export as Markdown"):
            parts = ["# Dataset & Tool Finder Results", f"**Topic:** {topic}"]
            parts.append("\n## Datasets")
            for d in datasets_f:
                parts.append(f"- **{d['name']}** ({d['modality']}, {d['accessibility']}, score {d['score']}) — {d['link']}")
            parts.append("\n## Tools & APIs")
            for t in tools_f:
                parts.append(f"- **{t['name']}** ({t['modality']}, {t['accessibility']}, score {t['score']}) — {t['link']}")
            parts.append("\n## Pre-trained Models")
            for m in models_f:
                parts.append(f"- **{m['name']}** ({m['modality']}, {m['accessibility']}, score {m['score']}) — {m['link']}")
            md = "\n".join(parts)
            st.markdown(to_markdown_download_link(md, "datasets_tools_models.md"), unsafe_allow_html=True)


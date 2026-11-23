# ArabicLegalTextMining

### Intelligent Arabic Legal Document Analysis Platform

**ArabicLegalTextMining** is a full-stack AI platform for searching, classifying, and analyzing Arabic legal documents using **Machine Learning, Natural Language Processing (NLP), and Retrieval-Augmented Generation (RAG)**.

The platform integrates four AI services through a modern **Angular + FastAPI** application.

## Features

| Service | Technologies | Description |
|---|---|---|
| **Lexical Search** | Chi², TF-IDF, Cosine Similarity | Retrieves relevant Arabic legal documents using statistical information retrieval. |
| **Legal Classification** | Multinomial Naive Bayes | Classifies legal texts using a probabilistic model implemented from scratch. |
| **Entity Search (NER)** | AraBERT, BIO Tagging | Extracts named entities and retrieves documents mentioning people, courts, and organizations. |
| **AI Legal Assistant** | AraBERT, Qdrant, Gemini Flash | Answers legal questions using retrieved document passages and RAG. |

## Tech Stack

- **Frontend:** Angular, TypeScript, HTML, SCSS
- **Backend:** Python, FastAPI, Pydantic
- **Machine Learning:** NumPy, Scikit-learn, Naive Bayes
- **Deep Learning / NLP:** PyTorch, Hugging Face Transformers, AraBERT
- **Vector Database:** Qdrant Local
- **Generative AI:** Google Gemini Flash
- **Testing:** Pytest, Ruff

## Dataset

The platform processes Arabic legal documents from two categories:

| Legal Category | Documents |
|---|---:|
| Family and Inheritance — غرفة الأحوال الشخصية والميراث | 1,883 |
| Real Estate — الغرفة العقارية | 365 |
| **Total** | **2,248** |

The raw legal corpus is kept private and excluded from GitHub.

## Project Architecture

```text
ArabicLegalTextMining/
├── data/                   # Private legal corpus
├── notebooks/              # ML/DL experiments
├── src/
│   ├── common/             # Shared preprocessing
│   ├── lexical_search/     # Chi² + TF-IDF + Cosine
│   ├── classification/     # Naive Bayes
│   ├── ner/                # AraBERT NER
│   └── rag/                # AraBERT + Qdrant + Gemini
├── backend/                # FastAPI REST API
├── frontend/angular-app/   # Angular web interface
├── scripts/                # Data and index preparation
├── artifacts/              # Local models and indexes
├── tests/                  # Automated tests
└── README.md
```

## Installation

**Requirements:** Python 3.11, Node.js, npm, Git, and a Gemini API key for RAG.

```powershell
git clone <REPOSITORY_URL>
cd ArabicLegalTextMining

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -e ".[dev,nlp]"
Copy-Item .env.example .env
```

Configure `.env` with your API credentials and model settings. The required datasets, trained models, and indexes must also be prepared locally.

## Run the Application

**Backend:**

```powershell
python -m uvicorn backend.app.main:app --reload
```

API documentation: `http://127.0.0.1:8000/docs`

**Frontend (separate terminal):**

```powershell
cd frontend/angular-app
npm install
npm start
```

Application: `http://localhost:4200`

## Testing

```powershell
python -m pytest tests/ -v --basetemp=data/pytest_tmp
python -m ruff check src/ backend/ scripts/ tests/
```

Frontend build:

```powershell
cd frontend/angular-app
npm run build
```
---

*Academic and research project. AI-generated legal information should be verified against authoritative sources and does not constitute legal advice.*

# ScamShield 🛡️

An AI-powered fraud detection assistant that helps users identify and understand Indian cyber scams — including UPI fraud, phishing emails, SMS/WhatsApp scams, fake job offers, and digital arrest scams.

Built using Retrieval-Augmented Generation (RAG) to provide accurate, context-aware responses grounded in a curated knowledge base.

---

## Features

- Detects and explains common Indian cyber scams
- RAG pipeline for accurate, knowledge-grounded responses
- Covers 4 fraud categories: phishing emails, SMS/WhatsApp scams, URL analysis, fake job offers
- Simple conversational UI built with Streamlit

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq LLaMA 3.3 70B |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Database | ChromaDB (214 vectors) |
| Framework | LangChain |
| UI | Streamlit |

---

## Project Structure

```
ScamShield/
├── data/
│   ├── phishing_emails.txt
│   ├── sms_whatsapp_scams.txt
│   ├── url_analysis.txt
│   └── fake_job_offers.txt
├── ingest.py          # Embeds knowledge base into ChromaDB
├── rag_pipeline.py    # RAG retrieval and response logic
├── app.py             # Streamlit UI
├── requirements.txt
└── .env               # API keys (not included)
```

---

## Setup & Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/pratham040545/ScamShield.git
cd ScamShield
```

**2. Create a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your API key**

Create a `.env` file in the root folder:
```
GROQ_API_KEY=your_groq_api_key_here
```

**5. Ingest the knowledge base**
```bash
python ingest.py
```

**6. Run the app**
```bash
streamlit run app.py
```

---

## Author

**Pratham Kalan**  
[LinkedIn](https://linkedin.com/in/pratham-kalan-a954b4287) | [GitHub](https://github.com/pratham040545)

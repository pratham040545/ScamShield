import os
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "scamshield_kb"

SOURCE_MAP = {
    "Phishing Email": "phishing_emails.txt.txt",
    "SMS / WhatsApp Message": "sms_whatsapp_scams.txt.txt",
    "Suspicious URL": "url_analysis.txt.txt",
    "Fake Job Offer": "fake_job_offers.txt.txt"
}

SYSTEM_PROMPT = """You are ScamShield, an expert AI fraud and phishing detection assistant with deep knowledge of Indian cyber fraud patterns.

You specialize in detecting:
- Phishing emails and social engineering attacks
- SMS and WhatsApp scams (UPI fraud, OTP scams, task-based fraud)
- Suspicious URLs and domain spoofing
- Fake job offers and advance fee fraud

When analyzing content always respond in this exact format:

## 🛡️ ScamShield Analysis

**Verdict:** [🔴 HIGH RISK — Likely Scam | 🟡 SUSPICIOUS — Proceed with Caution | 🟢 LIKELY SAFE]
**Risk Score:** X/10

---

### 🚩 Red Flags Detected
- [List each red flag found]

### 🎭 Scam Technique
[Explain what type of scam this is and how it works]

### ✅ What You Should Do
[Clear actionable steps]

### 💡 How to Protect Yourself
[One specific tip related to this scam type]

For follow-up questions be conversational and reference earlier analysis.
If content is clearly safe say so — don't force a scam verdict.
Be empathetic — never blame the victim."""


class ScamShieldRAG:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("❌ GROQ_API_KEY not found in .env")

        self.client = Groq(api_key=api_key)
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        db = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = db.get_collection(name=COLLECTION_NAME)
        print(f"✅ ChromaDB loaded — {self.collection.count()} vectors")

    def retrieve(self, query: str, detection_type: str, n_results: int = 6):
        query_embedding = self.embedding_model.encode(query).tolist()

        # Filter by source file if specific mode selected
        source_file = SOURCE_MAP.get(detection_type)
        where_filter = {"source": source_file} if source_file else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter
        )
        return results["documents"][0]

    def analyze(self, user_input: str, detection_type: str, chat_history: list) -> str:
        docs = self.retrieve(user_input, detection_type)
        context = "\n\n".join(docs)

        history_str = ""
        for msg in chat_history[-6:]:
            role = "User" if msg["role"] == "user" else "ScamShield"
            history_str += f"{role}: {msg['content']}\n\n"

        is_followup = len(chat_history) > 1

        if is_followup:
            instruction = f"""The user is asking a follow-up question.
Conversation so far:
{history_str}
Answer conversationally. Detection mode: {detection_type}"""
        else:
            instruction = f"""Analyze the following content for fraud indicators.
Detection mode: {detection_type}
Use the structured analysis format."""

        full_prompt = f"""Retrieved Knowledge:
{context}

{instruction}

User Input:
\"\"\"{user_input}\"\"\"

ScamShield:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.1,
            max_tokens=1500
        )
        return response.choices[0].message.content

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="ScamShield — AI Fraud Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #e74c3c;
        margin-bottom: 0;
    }
    .sub-header {
        color: #7f8c8d;
        font-size: 0.95rem;
        margin-top: 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="🔧 Loading ScamShield AI engine...")
def load_rag():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

    if not os.path.exists(CHROMA_PATH):
        return None
    from rag_pipeline import ScamShieldRAG
    return ScamShieldRAG()

with st.sidebar:
    st.markdown("## 🛡️ ScamShield")
    st.markdown("*AI-Powered Fraud & Phishing Detector*")
    st.divider()

    st.markdown("### 🔍 Detection Mode")
    detection_type = st.radio(
        label="Detection Mode",
        options=[
            "📧 Phishing Email",
            "📱 SMS / WhatsApp Message",
            "🔗 Suspicious URL",
            "💼 Fake Job Offer",
            "🔄 Auto Detect"
        ],
        index=4,
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### 📋 Quick Examples")

    examples = {
        "UPI Scam": "Congratulations! You have won Rs 50,000 in PhonePe lucky draw. To claim your prize click here: http://phonep3-rewards.xyz and enter your UPI PIN to verify.",
        "Fake Job": "Hi, we found your profile on Naukri. Amazon WFH is hiring Data Entry Operators. Salary Rs 35,000/month, no experience needed. Pay Rs 500 registration fee. Send Aadhaar on WhatsApp.",
        "Phishing URL": "http://sbi.co.in.secure-banking-login.xyz/account/verify",
        "Phishing Email": "Dear Customer, Your SBI account has been suspended. Verify immediately or account will be closed: http://sbi-secure-verify.com/login"
    }

    for label, text in examples.items():
        if st.button(f"▶ {label}", key=f"ex_{label}", use_container_width=True):
            st.session_state["prefill"] = text

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("prefill", None)
        st.rerun()

    st.divider()
    
    st.markdown("""
    <small>
    <b>Stack:</b> Groq LLaMA 3.3 + ChromaDB + Sentence Transformers + Streamlit<br>
    <b>KB:</b> Phishing · SMS Scams · URLs · Fake Jobs
    </small>
    """, unsafe_allow_html=True)


st.markdown('<p class="main-header">🛡️ ScamShield</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Fraud & Phishing Detection — Paste anything suspicious and ask me anything.</p>', unsafe_allow_html=True)

rag = load_rag()

if rag is None:
    st.error("""
    ⚠️ **Knowledge base not found!**
    Please run first:
    ```
    python ingest.py
    ```
    Then refresh this page.
    """)
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": """👋 **Welcome to ScamShield!**

I can analyze:
- 📧 **Phishing emails**
- 📱 **SMS / WhatsApp scams** — UPI fraud, OTP scams
- 🔗 **Suspicious URLs** — fake domains, spoofing
- 💼 **Fake job offers** — advance fee fraud

**Paste any suspicious content below** and I'll give you:
- Risk Score + Verdict
- Red flags detected
- Scam technique explained
- Actionable advice

➡️ *Try a Quick Example from the sidebar!*"""
    })

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🛡️" if message["role"] == "assistant" else "👤"):
        st.markdown(message["content"])

prefill_text = st.session_state.pop("prefill", None)
prompt = st.chat_input("Paste suspicious email, SMS, URL, or job offer here...")

if prefill_text and not prompt:
    prompt = prefill_text

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🛡️"):
        with st.spinner("🔍 Analyzing for fraud patterns..."):
            mode = detection_type.split(" ", 1)[1] if " " in detection_type else detection_type
            response = rag.analyze(
                user_input=prompt,
                detection_type=mode,
                chat_history=st.session_state.messages[:-1]
            )
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

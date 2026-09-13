# 🎓 ScholarReady AI

AI-powered scholarship discovery, eligibility checking, and live
verification platform — built for HEC-NCEAC & PEC Generative &
Agentic AI Training, Cohort 11 Hackathon.

## 🔗 Live App
https://scholarready.streamlit.app/

## 🎥 Demo Video
[Insert your video link]

## 📄 PRD
https://drive.google.com/file/d/1sldWgcWiRlyr7b26iZ1CSvp-V7WaMJVE/view?usp=sharing

## 🚀 Features
- CV parsing (PDF/DOCX) with full manual fallback
- Semantic scholarship matching (any field, not just STEM)
- Deterministic 4-category eligibility engine with transparent
  checklists (Eligible / Needs Review / Not Eligible / Suspicious)
- Live verification via Tavily + Firecrawl with hallucination
  guardrails for deadlines and funding amounts
- Conversational Scholarship Coach
- Analytics dashboard (Plotly)

## 🧱 Tech Stack
Streamlit · Groq (GPT-OSS 120B/20B) · Gemini · Sentence Transformers ·
Tavily · Firecrawl · Plotly

## ⚙️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py

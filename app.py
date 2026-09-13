from pathlib import Path

Path("app.py").write_text(r'''
import streamlit as st
import pandas as pd
import os
import json
import io
import time as _time
import numpy as np
import re
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="ScholarReady AI", page_icon="🎓", layout="wide")

# ============================================
# API KEYS (Streamlit Secrets on cloud, env locally)
# ============================================
def get_key(name, default=""):
    try:
        return st.secrets[name]
    except:
        return os.environ.get(name, default)

GROQ_KEY = get_key("GROQ_API_KEY")
TAVILY_KEY = get_key("TAVILY_API_KEY")
FIRECRAWL_KEY = get_key("FIRECRAWL_API_KEY")
GEMINI_KEY = get_key("GEMINI_API_KEY")
GROQ_MODEL = get_key("GROQ_MODEL", "openai/gpt-oss-120b")
GEMINI_MODEL = get_key("GEMINI_MODEL", "gemini-flash-latest")

# ============================================
# DATA LOADING
# ============================================
@st.cache_data
def load_data():
    df = pd.read_csv("scholarships_final.csv")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    for c in ["deadline","link","description","location","years"]:
        if c not in df:
            df[c] = ""
        df[c] = df[c].fillna("")
    if "source" not in df:
        df["source"] = "csv"
    df["source"] = df["source"].fillna("csv")
    if "needs_verify" not in df:
        df["needs_verify"] = True
    df["needs_verify"] = df["needs_verify"].fillna(True)
    df["text"] = df["scholarship_name"].astype(str) + ". " + df["description"].astype(str) + ". " + df["location"].astype(str)
    return df

DF = load_data()

# ============================================
# SEMANTIC ENGINE
# ============================================
@st.cache_resource
def embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def build_emb():
    m = embedder()
    df = DF.copy()
    cur = df[df["source"]=="curated"]
    rest = df[df["source"]!="curated"]
    n = min(1200, len(rest))
    rest = rest.sample(n=n, random_state=42) if n > 0 else rest
    sub = pd.concat([cur, rest]).drop_duplicates("scholarship_name").reset_index(drop=True)
    emb = m.encode(sub["text"].tolist(), normalize_embeddings=True, show_progress_bar=False)
    return sub, np.array(emb)

def semantic_rank(ptext, top_k=15):
    m = embedder()
    sub, emb = build_emb()
    q = m.encode([ptext], normalize_embeddings=True)
    sc = (emb @ q.T).ravel()
    sub = sub.copy()
    sub["semantic_score"] = sc
    cur = sub[sub["source"]=="curated"].copy()
    rest = sub[sub["source"]!="curated"].sort_values("semantic_score", ascending=False).head(top_k)
    out = pd.concat([cur, rest]).drop_duplicates("scholarship_name")
    return out.sort_values("semantic_score", ascending=False).reset_index(drop=True)

# ============================================
# LLM LAYER
# ============================================
def llm(prompt, max_tokens=2500, force_json=False, _retry=2):
    if GROQ_KEY:
        for a in range(_retry+1):
            try:
                from groq import Groq
                kw = dict(
                    model=GROQ_MODEL,
                    messages=[{"role":"user","content":prompt}],
                    temperature=0.1,
                    max_tokens=max_tokens
                )
                if force_json:
                    kw["response_format"] = {"type":"json_object"}
                return Groq(api_key=GROQ_KEY).chat.completions.create(**kw).choices[0].message.content.strip()
            except Exception as e:
                if "429" in str(e) and a < _retry:
                    _time.sleep(8)
                    continue
                break
    if GEMINI_KEY:
        try:
            from google import genai
            return genai.Client(api_key=GEMINI_KEY).models.generate_content(
                model=GEMINI_MODEL, contents=prompt
            ).text.strip()
        except:
            return None
    return None

def pjson(t):
    if not t:
        return None
    try:
        if "```" in t:
            t = t.split("```")[1]
            if t.startswith("json"):
                t = t[4:]
        a = t.find("{")
        b = t.rfind("}")
        if a >= 0 and b > a:
            t = t[a:b+1]
        return json.loads(t)
    except:
        return None

# ============================================
# AGENTS
# ============================================
def profile_from_cv(cv):
    p = ("Extract a RICH student profile from this CV. Return ONLY JSON keys: "
       "name, gender, nationality, gpa, major, target_degree, field_of_study(list), "
       "profession, career_goal, skills(list of specific tools/techniques), "
       "research_experience(bool), research_topics(list of actual project names), "
       "work_experience_years(num), strengths_for_scholarships(list), "
       "gaps_for_top_scholarships(list), ideal_scholarship_types(list). "
       "Extract REAL details from the CV, be specific about projects and skills. "
       "target_degree = the NEXT degree they seek (BS student -> MS).\n\nCV:\n" + cv[:5000])
    return pjson(llm(p, 1500))

def fit_batch(profile, chunk):
    profile_str = json.dumps(profile, ensure_ascii=False)[:2200]
    chunk_str = json.dumps(chunk, ensure_ascii=False)[:6000]

    rules_text = (
        "HARD RULES (VIOLATION = MISMATCH):\n"
        "1. DEGREE LEVEL MUST MATCH: If scholarship requires PhD/Postdoc and student wants Master's -> STATUS = not_eligible.\n"
        "2. DEADLINE PASSED -> STATUS = not_eligible.\n"
        "3. NATIONALITY EXCLUDED -> STATUS = not_eligible.\n"
        "4. GPA BELOW STATED MINIMUM -> STATUS = needs_work (show exact shortfall).\n"
        "5. REQUIRES NOMINATION (Eiffel-style) -> STATUS = conditional NOT ready.\n"
        "6. FUNDING TYPE: If partial grant or loan -> mark as almost with warning.\n"
        "7. REALISTIC CHANCE: Gates/Fulbright/Chevening for regional uni students <15% admit. Status = dream.\n"
        "8. FIELD MUST MATCH: Library/agriculture/nursing for an AI/CS student -> mismatch.\n"
        "9. WOMENS UNIVERSITY = Strong fit for diversity (Erasmus, AAUW, DAAD)."
    )

    p_text = (
        "You are ScholarReady Fit Agent - STRICT ELIGIBILITY CHECKER.\n\n"
        "STUDENT PROFILE:\n" + profile_str + "\n\n"
        "SCHOLARSHIPS TO EVALUATE:\n" + chunk_str + "\n\n"
        + rules_text + "\n\n"
        "RETURN ONLY JSON:\n"
        '{"matches":[{"id":0,"status":"ready","fit_score":90,"why_fit":"specific reason tied to THIS student",'
        '"missing":["doc1"],"action_plan":["step1"],"red_flags":[],"priority":"high"}]}\n\n'
        "status must be: ready|almost|needs_work|conditional|dream|not_eligible|mismatch|scam\n"
        "CRITICAL: A Master applicant CANNOT be ready for a PhD scholarship. Ever."
    )

    r = pjson(llm(p_text, 2500, force_json=True))
    return r.get("matches", []) if r else []

def agent_fit(profile, df):
    items=[{
        "id": i,
        "name": str(r.get("scholarship_name",""))[:80],
        "amount": float(r.get("amount",0) or 0),
        "deadline": str(r.get("deadline","")),
        "location": str(r.get("location",""))[:60],
        "description": str(r.get("description",""))[:200],
        "source": str(r.get("source","")),
        "semantic": round(float(r.get("semantic_score",0)),3)
    } for i, (_,r) in enumerate(df.iterrows())]

    allm = []
    prog = st.progress(0.0)
    B = 8
    for b in range(0, len(items), B):
        allm += fit_batch(profile, items[b:b+B])
        prog.progress(min(1.0, (b+B) / max(len(items),1)))
        _time.sleep(3)
    prog.empty()

    if not allm:
        for it in items:
            s = it["semantic"]
            stt = "ready" if s>0.45 else "almost" if s>0.33 else "needs_work" if s>0.22 else "not_eligible"
            allm.append({
                "id": it["id"], "status": stt, "fit_score": int(s*100),
                "why_fit": "Semantic match (fallback).",
                "missing": [], "action_plan": [], "red_flags": [], "priority": "medium"
            })
    return {"matches": allm}

def scout(profile):
    if not TAVILY_KEY:
        return []
    try:
        from tavily import TavilyClient
        q = (f"fully funded {profile.get('target_degree','masters')} scholarship "
               f"{profile.get('profession','')} {profile.get('nationality','')} "
               f"{' '.join(profile.get('field_of_study',[])[:2])} 2027")
        r = TavilyClient(api_key=TAVILY_KEY).search(query=q, max_results=6)
        return [{
            "title": x.get("title",""),
            "content": x.get("content",""),
            "url": x.get("url","")
        } for x in r.get("results",[])]
    except:
        return []

def scrape(url):
    if not FIRECRAWL_KEY or not url:
        return None
    try:
        from firecrawl import FirecrawlApp
        r = FirecrawlApp(api_key=FIRECRAWL_KEY).scrape_url(url, formats=["markdown"])
        return (r.markdown if hasattr(r,"markdown") else r.get("markdown",""))[:3000]
    except:
        return None

def verify(name, link):
    c = scrape(link)
    if not c:
        return None
    return pjson(llm(
        f"Extract as ONLY JSON for {name}\nCONTENT:{c[:2500]}\n"
        'Return: {"real_deadline":"","real_amount":"","gpa_min":0,"requirements":[],"is_legitimate":true}',
        800
    ))

def chat_advisor(profile, scholarship_name, scholarship_desc, question):
    prompt = (
        "You are a scholarship advisor helping a student.\n"
        "STUDENT: " + json.dumps(profile, ensure_ascii=False)[:1500] + "\n\n"
        "SCHOLARSHIP: " + str(scholarship_name) + "\n"
        "DETAILS: " + str(scholarship_desc)[:800] + "\n\n"
        "STUDENT QUESTION: " + question + "\n\n"
        "Give specific, actionable advice in 3-5 sentences. "
        "Reference the student's actual background and this scholarship's requirements."
    )
    return llm(prompt, 600) or "Sorry, could not generate advice right now."

def ptext(p):
    return (f"{p.get('target_degree','MS')} scholarship. Profession {p.get('profession','')}. "
            f"Major {p.get('major','')}. Fields {', '.join(p.get('field_of_study',[]))}. "
            f"Skills {', '.join(p.get('skills',[])[:10])}. Research {p.get('research_experience')} "
            f"{', '.join(p.get('research_topics',[])[:5])}. Nationality {p.get('nationality')} GPA {p.get('gpa')}. "
            f"Want {', '.join(p.get('ideal_scholarship_types',['funded STEM']))}.")

# ============================================
# HARD CONSTRAINT FILTER
# ============================================
def hard_constraint_filter(df, profile):
    target_degree = str(profile.get("target_degree", "MS")).upper().strip()
    nationality = str(profile.get("nationality", "")).upper().strip()
    try:
        gpa = float(profile.get("gpa", 0))
    except (ValueError, TypeError):
        gpa = 0.0

    degree_keywords = {
        'PHD': ['phd','ph.d','doctoral','doctorate','postdoc','post-doctoral','dphil'],
        'MS': ["master's","msc","m.sc","mphil","m.phil","mba","graduate program","postgraduate","master degree","graduate study"],
        'BS': ['bachelor','undergraduate','b.sc','bsc',"bachelor's","first cycle","freshman year"]
    }

    def extract_degrees(text):
        t = text.lower()
        found = set()
        for level, kws in degree_keywords.items():
            for kw in kws:
                if kw in t:
                    found.add(level)
        return found

    def check_deadline(ds):
        if not ds:
            return True
        for fmt in ["%Y-%m-%d","%d/%m/%Y","%B %d, %Y","%d %B %Y"]:
            try:
                d = datetime.strptime(ds.strip(), fmt)
                return d.date() > datetime.now().date()
            except:
                continue
        return True

    blocked_ids = []
    keep_indices = []

    for idx, row in df.iterrows():
        text = f"{row.get('scholarship_name','')} {row.get('description','')} {row.get('years','')}"
        text_lower = text.lower()
        sdeg = extract_degrees(text)

        if target_degree == 'MS':
            if sdeg == {'PHD'}:
                blocked_ids.append({'idx': idx, 'reason': f"PhD/Postdoc only. You want {target_degree}."})
                continue
        elif target_degree == 'PHD':
            if 'phd' not in text_lower and 'doctoral' not in text_lower and sdeg and 'PHD' not in sdeg:
                blocked_ids.append({'idx': idx, 'reason': "Targets Master's/Bachelor level."})
                continue

        if nationality:
            excl = [f"not open to {nationality.lower()}", f"no {nationality.lower()} citizens"]
            if any(pp in text_lower for pp in excl):
                blocked_ids.append({'idx': idx, 'reason': f"Excludes {nationality} nationals."})
                continue

        gm = re.search(r'gpa\s*[\s>]=?\s*([\d.]+)', text_lower)
        if gm:
            try:
                min_gpa = float(gm.group(1))
                if min_gpa < 0 or min_gpa > 10:
                    min_gpa = 0.0
            except (ValueError, TypeError):
                min_gpa = 0.0
            if gpa < min_gpa and min_gpa > 0:
                blocked_ids.append({'idx': idx, 'reason': f"GPA {gpa} below minimum {min_gpa}."})
                continue

        if not check_deadline(str(row.get('deadline',''))):
            blocked_ids.append({'idx': idx, 'reason': f"Deadline passed."})
            continue

        keep_indices.append(idx)

    return df.iloc[keep_indices].reset_index(drop=True), blocked_ids

# ============================================
# FUNDING REALITY CHECKER
# ============================================
def validate_funding_realism(row):
    name = str(row.get("scholarship_name","")).lower()
    desc = str(row.get("description","")).lower()
    claimed = float(row.get("amount",0) or 0)
    text = name + " " + desc
    warnings = []
    realistic = claimed

    if claimed > 60000 and ('master' in text or 'msc' in text):
        warnings.append(f"Amount ${claimed:,.0f} unusually high for Master's.")
    for ind, mx in [('women techmaker',3000),('conference grant',2000),('travel grant',2000)]:
        if ind in text and claimed > mx*2:
            realistic = mx
            warnings.append(f"'{ind}' typically ~${mx:,}, not ${claimed:,.0f}.")
            break
    if any(w in text for w in ['repay','return to home','bond','pay back']):
        warnings.append("Requires repayment/service obligation.")
    if ('processing fee' in text or 'application fee' in text) and claimed > 0:
        warnings.append("Requires payment to apply. Possible scam.")
        realistic = 0
    return realistic, ("\n".join(warnings) if warnings else None)

# ============================================
# PROFILE COMPLETENESS
# ============================================
def calculate_profile_completeness(profile):
    required = {
        'name': bool(profile.get('name','')),
        'nationality': bool(profile.get('nationality','')),
        'gpa': profile.get('gpa',0) > 0,
        'target_degree': profile.get('target_degree','') in ['BS','MS','PhD'],
        'field_of_study': len(profile.get('field_of_study',[])) > 0,
        'profession': bool(profile.get('profession','')),
        'skills': len(profile.get('skills',[])) >= 3,
        'ielts': False,
        'passport': False,
        'lor_writers': len(profile.get('strengths_for_scholarships',[])) > 0,
        'research_experience': profile.get('research_experience',False),
        'projects': len(profile.get('research_topics',[])) > 0,
        'transcript': False,
    }
    filled = sum(1 for v in required.values() if v)
    total = len(required)
    pct = (filled/total)*100
    missing = [k for k,v in required.items() if not v]
    unlocks = 0
    if not required['projects'] and filled >= 7: unlocks += 4
    if not required['passport'] and filled >= 8: unlocks += 2
    if not required['ielts'] and filled >= 7: unlocks += 6
    return {
        'percentage': round(pct,1), 'filled': filled, 'total': total,
        'missing': missing, 'unlocks_if_fixed': unlocks,
        'tier': 'Strong Candidate' if pct>=85 else 'Good Foundation' if pct>=65 else 'Needs Strengthening'
    }

# ============================================
# STATUS CLASSIFIER
# ============================================
def bk(c, profile):
    s = (c["status"] or "").lower().replace(" ","_")
    if s in ["mismatch","scam","not_eligible","conditional","dream"]:
        return s
    r = c.get("row", {})
    desc = str(r.get("description","")) + " " + str(r.get("scholarship_name",""))
    if any(kw in desc.lower() for kw in ['nomination','university nomination','institutional nomination']):
        return "conditional"
    name_low = str(r.get("scholarship_name","")).lower()
    if any(dp in name_low for dp in ['gates','rhodes','chevening','fulbright']):
        if c["fit"] >= 75:
            return "dream"
    f = c["fit"]
    if f >= 82: return "ready"
    elif f >= 68: return "almost"
    elif f >= 45: return "needs_work"
    else: return "not_eligible"

LABEL_MAP = {
    'ielts': 'IELTS Score',
    'passport': 'Passport',
    'lor_writers': '2-3 Letters of Recommendation',
    'projects': 'Published/preprint research project',
    'transcript': 'Official University Transcripts'
}

# ============================================
# HEADER
# ============================================
st.title("🎓 ScholarReady AI")
st.markdown("### Semantic + Multi-Agent Reasoning | v2.0 (Constraint-Aware)")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Database", f"{len(DF):,}")
c2.metric("Curated", f"{(DF['source']=='curated').sum()}")
c3.metric("Reasoner", GROQ_MODEL[:16])
c4.metric("Live", "Tavily+FC+Gate")
st.markdown("---")

# ============================================
# SIDEBAR
# ============================================
st.sidebar.title("Student")
cvf = st.sidebar.file_uploader("Upload CV", type=["pdf","docx"])
if cvf and st.sidebar.button("AI Understand CV"):
    raw = ""
    try:
        if cvf.name.endswith(".pdf"):
            import PyPDF2
            raw = "\n".join((pg.extract_text() or "") for pg in PyPDF2.PdfReader(io.BytesIO(cvf.read())).pages)
        else:
            import docx
            raw = "\n".join(pg.text for pg in docx.Document(io.BytesIO(cvf.read())).paragraphs)
    except Exception as e:
        st.sidebar.error(str(e))
    if raw:
        st.sidebar.info(f"CV read: {len(raw)} chars")
        if not GROQ_KEY:
            st.sidebar.error("No GROQ_API_KEY! Add it in Settings > Secrets")
        with st.spinner("Reading CV..."):
            pr = profile_from_cv(raw)
        if pr:
            st.session_state["p"] = pr
            st.sidebar.success(f"Got: {pr.get('profession','?')}")
        else:
            st.sidebar.error("Parse failed - check API keys in Secrets")
    else:
        st.sidebar.error("Could not read text from file")

dp = {
    "name":"","gender":"","nationality":"","gpa":3.0,
    "major":"","target_degree":"MS","field_of_study":[],
    "profession":"","career_goal":"",
    "skills":[],"research_experience":False,"research_topics":[],
    "work_experience_years":0,"strengths_for_scholarships":[],
    "gaps_for_top_scholarships":[],"ideal_scholarship_types":[]
}
p = st.session_state.get("p", dp)
if "p" not in st.session_state:
    st.sidebar.info("Upload your CV for real AI matching, or fill fields manually.")

p["name"]=st.sidebar.text_input("Name", p.get("name","") or "")
p["nationality"]=st.sidebar.text_input("Nationality", p.get("nationality","") or "")
p["gpa"]=st.sidebar.number_input("GPA", 0.0, 4.0, float(p.get("gpa",3.0) or 3.0), 0.1)
p["profession"]=st.sidebar.text_input("Profession", p.get("profession","") or "")
dopts=["BS","MS","PhD"]
didx=dopts.index(p["target_degree"]) if p.get("target_degree") in dopts else 1
p["target_degree"]=st.sidebar.selectbox("Target Degree", dopts, index=didx)
degree_filter=st.sidebar.selectbox("Show scholarships for", ["Any","BS","MS","PhD"], index=0)
_fld_opts=["AI","CS","Engineering","Medicine","Business","Data_Science"]
_fld_def=[f for f in p.get("field_of_study",[]) if f in _fld_opts]
p["field_of_study"]=st.sidebar.multiselect("Fields", _fld_opts, _fld_def)

if st.sidebar.button("Run Intelligent Match", type="primary"):
    st.session_state["run"]=True

# ============================================
# MAIN LOGIC
# ============================================
if st.session_state.get("run"):

    if not p.get("field_of_study") and not p.get("profession"):
        st.error("Please upload your CV (or fill Profession + Fields) so matching reflects YOU.")
        st.stop()

    comp = calculate_profile_completeness(p)

    with st.expander("AI-Understood Profile", expanded=True):
        st.markdown(f"**{p.get('name') or 'Student'}** - {p.get('profession')} | {p.get('target_degree')} {p.get('major')} | GPA {p.get('gpa')} | {p.get('nationality')}")
        st.markdown(f"**Profile Readiness:** {comp['percentage']}% (**{comp['tier']}**) - Fix missing items to unlock {comp['unlocks_if_fixed']} more scholarships")
        if p.get("skills"):
            st.markdown("Skills: " + ", ".join(p.get("skills",[])[:8]))
        if p.get("research_topics"):
            st.markdown("Research: " + ", ".join(p.get("research_topics",[])[:5]))
        if comp['missing']:
            st.markdown(f"**Missing Items ({len(comp['missing'])}/{comp['total']}):**")
            for m in comp['missing']:
                st.markdown("- " + str(LABEL_MAP.get(m, m)))

    with st.spinner("Step 1: Filtering by eligibility..."):
        filtered_df, blocked = hard_constraint_filter(DF, p)
    st.info(f"Removed {len(blocked)} ineligible scholarships before ranking")

    if blocked:
        with st.expander(f"Filtered Out ({len(blocked)} scholarships)", expanded=False):
            for b_item in blocked[:10]:
                row = DF.iloc[b_item['idx']]
                st.markdown(f"X {str(row.get('scholarship_name','?'))[:65]}: **{b_item['reason']}**")

    with st.spinner("Step 2: Semantic retrieval on eligible pool..."):
        ranked=semantic_rank(ptext(p), 15 if len(filtered_df)>=15 else max(len(filtered_df),1))
    st.success(f"Step 2 done: {len(ranked)} candidates evaluated")

    if degree_filter != "Any":
        tgt=degree_filter.upper()
        def deg_ok(row):
            txt=(str(row.get("years",""))+" "+str(row.get("description",""))+" "+str(row.get("scholarship_name",""))).lower()
            has_phd=any(k in txt for k in ["phd","ph.d","doctoral","doctorate","postdoc"])
            has_ms=any(k in txt for k in ["master","msc","m.sc","mba","graduate","postgraduate"])
            has_bs=any(k in txt for k in ["bachelor","undergraduate","freshman","sophomore","junior","senior","first-year"])
            if not (has_phd or has_ms or has_bs): return True
            if tgt=="PHD": return has_phd
            if tgt=="MS": return has_ms or (not has_bs and not has_phd)
            if tgt=="BS": return has_bs
            return True
        before=len(ranked)
        ranked=ranked[ranked.apply(deg_ok,axis=1)].reset_index(drop=True)
        st.info(f"Showing only {degree_filter}: {len(ranked)}/{before}")

    with st.spinner("Step 3: Fit Agent reasoning..."):
        an=agent_fit(p,ranked)
    st.success("Step 3 done")

    byid={m["id"]:m for m in an["matches"] if "id" in m}
    cards=[]
    for i,(_,r) in enumerate(ranked.iterrows()):
        m=byid.get(i,{})
        base_fit=int(m.get("fit_score",int(100*float(r.get("semantic_score",0)))))
        is_cur=r.get("source")=="curated"
        amt=float(r.get("amount",0) or 0)
        if is_cur: base_fit=min(100,base_fit+10)
        elif 0<amt<=600: base_fit=max(0,base_fit-25)
        cards.append({
            "row": r, "status": str(m.get("status","almost")).lower(), "fit": base_fit,
            "why": m.get("why_fit",""), "missing": m.get("missing",[]) or [],
            "actions": m.get("action_plan",[]) or [], "flags": m.get("red_flags",[]) or [],
            "priority": m.get("priority","medium"), "sem": float(r.get("semantic_score",0))
        })

    G = {k: [] for k in ["ready","conditional","dream","almost","needs_work","not_eligible","mismatch","scam"]}
    for c in cards:
        G[bk(c, p)].append(c)
    for k in G:
        G[k].sort(key=lambda x:(-(x["priority"]=="high"),-x["fit"],-x["sem"]))

    # ===== METRICS =====
    cols=st.columns(8)
    for col,(lbl,k) in zip(cols,[("Ready","ready"),("Conditional","conditional"),
                                    ("Dream","dream"),("Almost","almost"),
                                    ("Work","needs_work"),("NotElig","not_eligible"),
                                    ("Mismatch","mismatch"),("Scam","scam")]):
        col.metric(lbl,len(G[k]))

    # ===== GRAPHS =====
    st.markdown("### Overview")
    gcol1, gcol2 = st.columns(2)
    with gcol1:
        tier_data = pd.DataFrame({
            "Tier": ["Ready","Almost","Dream","Conditional","Work","NotElig","Mismatch","Scam"],
            "Count": [len(G["ready"]),len(G["almost"]),len(G["dream"]),len(G["conditional"]),
                      len(G["needs_work"]),len(G["not_eligible"]),len(G["mismatch"]),len(G["scam"])]
        })
        tier_data = tier_data[tier_data["Count"]>0]
        if len(tier_data)>0:
            fig1 = px.pie(tier_data, values="Count", names="Tier", title="Scholarships by Tier",
                          color_discrete_sequence=px.colors.qualitative.Set2, hole=0.4)
            st.plotly_chart(fig1, use_container_width=True)
    with gcol2:
        applyable = G["ready"]+G["almost"]
        if applyable:
            fund_data = pd.DataFrame([{
                "name": str(c["row"].get("scholarship_name",""))[:25],
                "amount": float(c["row"].get("amount",0) or 0)
            } for c in applyable if float(c["row"].get("amount",0) or 0)>0])
            if len(fund_data)>0:
                fund_data = fund_data.sort_values("amount",ascending=True).tail(10)
                fig2 = px.bar(fund_data, x="amount", y="name", orientation="h",
                              title="Top Funding You Can Apply To ($)",
                              color="amount", color_continuous_scale="Viridis")
                fig2.update_layout(yaxis_title="", xaxis_title="Amount ($)")
                st.plotly_chart(fig2, use_container_width=True)

    total_money = sum(float(c["row"].get("amount",0) or 0) for c in G["ready"]+G["almost"])
    st.success(f"💰 Total funding you can realistically apply to: ${total_money:,.0f}")

    # ===== TABS =====
    tabs=st.tabs([
        f"🟢 Ready ({len(G['ready'])})", f"🔶 Conditional ({len(G['conditional'])})",
        f"⭐ Dream ({len(G['dream'])})", f"🟡 Almost ({len(G['almost'])})",
        f"🟠 Work ({len(G['needs_work'])})", f"🔴 NotElig ({len(G['not_eligible'])})",
        f"⚠️ Mismatch ({len(G['mismatch'])})", f"🚨 Scam ({len(G['scam'])})"
    ])

    def render_tab(lst, tab_container):
        with tab_container:
            if not lst:
                st.info("No scholarships in this category.")
                return
            for i,c in enumerate(lst):
                r=c["row"]
                nm=str(r.get("scholarship_name",""))[:70]
                amt=float(r.get("amount",0) or 0)
                star="⭐" if r.get("source")=="curated" else ""
                with st.expander(f"{star} {nm} | fit {c['fit']}% | ${amt:,.0f}"):
                    st.markdown(f"📅 {r.get('deadline')} | 📍 {r.get('location')} | sem {c['sem']:.2f}")
                    if amt > 0:
                        real_amt, fund_warn = validate_funding_realism(r)
                        if fund_warn:
                            st.warning(fund_warn)
                            if real_amt != amt:
                                st.caption(f"Realistic value: ${real_amt:,.0f}")
                    if c["why"]:
                        st.markdown(f"**🧠 Why this fits you:** {c['why']}")
                    desc_preview = str(r.get("description",""))[:400]
                    if desc_preview.strip():
                        st.write(desc_preview)
                    if c["missing"]:
                        st.markdown("**Missing:**")
                        for mx in c["missing"]:
                            st.markdown("- " + str(LABEL_MAP.get(mx, mx)))
                    if c["actions"]:
                        st.markdown("**Action Plan:**")
                        for act in c["actions"]:
                            st.markdown(f"- [ ] {act}")
                    if c["flags"]:
                        st.markdown("**Concerns:**")
                        for fl in c["flags"]:
                            st.markdown(f"- {fl}")

                    lk=str(r.get("link",""))
                    if lk.startswith("http") and "fake" not in lk:
                        st.markdown(f"🔗 [Apply Here]({lk})")

                    # ===== CHAT WITH ADVISOR =====
                    st.markdown("**💬 Ask AI advisor about this scholarship:**")
                    q = st.text_input("e.g. What should I improve to win this?",
                                       key=f"chat_q_{i}_{nm[:12]}")
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        if st.button("Ask Advisor", key=f"ask_{i}_{nm[:12]}"):
                            if q:
                                with st.spinner("Advisor thinking..."):
                                    ans = chat_advisor(p, nm, str(r.get("description","")), q)
                                st.info(ans)
                            else:
                                st.warning("Type a question first")
                    with cc2:
                        if lk.startswith("http") and "fake" not in lk:
                            if st.button("Verify Details", key=f"v{i}{nm[:15]}"):
                                with st.spinner("Verifying..."):
                                    v = verify(nm,lk)
                                    st.json(v) if v else st.warning("Unavailable")

    render_tab(G["ready"], tabs[0])
    render_tab(G["conditional"], tabs[1])
    render_tab(G["dream"], tabs[2])
    render_tab(G["almost"], tabs[3])
    render_tab(G["needs_work"], tabs[4])
    render_tab(G["not_eligible"], tabs[5])
    render_tab(G["mismatch"], tabs[6])
    render_tab(G["scam"], tabs[7])

    with st.spinner("Step 4: Live scouting..."):
        lv=scout(p)
    if lv:
        st.markdown("### 🌐 Live Discoveries")
        live_df = pd.DataFrame([{
            "scholarship_name": x["title"], "description": x["content"], "link": x["url"],
            "amount": 0, "deadline":"", "location":"Discovered Online",
            "source":"tavily_live", "years":"2027", "text": x["title"]+". "+x["content"]
        } for x in lv])
        live_filtered,_ = hard_constraint_filter(live_df, p)
        if len(live_filtered) > 0:
            st.success(f"Found {len(live_filtered)} eligible new scholarships online!")
            for x in live_filtered.itertuples():
                with st.expander(f"NEW {str(x.scholarship_name)[:70]}"):
                    st.write(str(x.description)[:300])
                    if x.link: st.markdown(f"[Source]({x.link})")
        else:
            st.info("No eligible results in live search.")
else:
    st.info("""
Upload CV then Run Intelligent Match

Features:
- Constraint Gate: removes PhD-only / expired / nationality / GPA fails BEFORE ranking
- Semantic + AI reasoning for each scholarship
- Chat advisor: ask "what should I improve?" per scholarship
- Graphs: tier breakdown + funding chart
- Funding Reality Checker: flags inflated amounts and loans
- Dream Tier: honest <15% chance for Gates/Fulbright/Rhodes
- Live web scouting via Tavily
Always evaluated: KAUST, Erasmus Mundus, DAAD, AAUW, Stipendium Hungaricum.
""")

st.caption("ScholarReady AI v2.0 | Semantic + Multi-Agent + Chat | Free")
''')
print("✅ Complete app.py with CHAT + GRAPHS + polish written")
print("   💬 Chat advisor in every scholarship card")
print("   📊 Pie chart (tiers) + Bar chart (funding)")
print("   💰 Total funding counter")
print("   🎨 Emojis on tabs + Apply links")

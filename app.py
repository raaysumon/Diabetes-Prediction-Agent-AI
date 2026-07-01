import os
import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
from groq import Groq
import re
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io
import time

# --- Page Config ---
st.set_page_config(
    page_title="Early Diabetes Chatbot AI",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- API Key (use st.secrets in production) ---
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "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")

# --- 📚 RAG Knowledge Base Setup with References ---
@st.cache_resource
def setup_rag_knowledge_base():
    """Returns a list of guideline dicts with text and source."""
    return [
        {
            "text": "Polyuria (frequent urination) and Polydipsia (excessive thirst) are key indicators of high blood glucose. Immediate tests required: HbA1c (>6.5% indicates diabetes) and Fasting Blood Sugar (FBS >126 mg/dL).",
            "source": "WHO Diabetes Guidelines 2023 – https://www.who.int/diabetes/guidelines"
        },
        {
            "text": "Delayed healing of wounds or cuts indicates microvascular complications often related to prolonged hyperglycemia. Patients must be screened for peripheral neuropathy and HbA1c.",
            "source": "ADA Standards of Medical Care in Diabetes – https://care.diabetesjournals.org/"
        },
        {
            "text": "Diabetes management for high risk includes lifestyle changes: reducing carbohydrate intake to less than 45% of daily calories, engaging in 150 minutes of moderate exercise per week, and weight monitoring.",
            "source": "NICE Guideline NG28 – https://www.nice.org.uk/guidance/ng28"
        },
        {
            "text": "For low risk or negative diabetes risk screen, routine wellness checkup including annual HbA1c and fasting glucose is recommended, especially for adults above 35 years old.",
            "source": "USPSTF Recommendation – https://www.uspreventiveservicestaskforce.org/"
        },
        {
            "text": "Symptoms like Irritability, Alopecia (hair loss), and skin Itching can be secondary systemic signs of metabolic changes or poor circulation linked with early insulin resistance.",
            "source": "Endocrine Society Clinical Practice Guidelines – https://www.endocrine.org/guidelines"
        }
    ]

# Load guidelines with sources
guidelines_data = setup_rag_knowledge_base()
guidelines_texts = [item["text"] for item in guidelines_data]
guidelines_sources = [item["source"] for item in guidelines_data]

def get_rag_agent_response(patient_context, language):
    """Returns (agent_report, sources_list)"""
    context_source = "\n".join(guidelines_texts)
    try:
        client = Groq(api_key=GROQ_API_KEY)
        system_content = (
            f"You are an expert Medical AI Agent acting as a supportive AI Doctor named DECat-AI. Analyze the patient strictly based on the provided Clinical Guidelines. "
            f"CRITICAL RULE: You MUST write your entire response strictly in {language}. If the language is বাংলা, use simple and clear Bengali words. "
            f"Format your response as a beautiful, professional, and clear clinical report. Use clear headers like 'Diagnostic Advice', "
            f"'Dietary Modifications', and 'Lifestyle Protocol' (translate these headers properly if the language is Bengali, e.g., 'ডায়াগনস্টিক পরামর্শ', 'খাদ্যতালিকাগত পরিবর্তন', 'জীবনধারা প্রোটোকল'). "
            f"STRICT RULE: Never use mathematical symbols like greater than, less than, percentage signs, dollar signs, or brackets. "
            f"Write them in plain text words if necessary (e.g., in English write 'greater than 6.5 percent', or in Bengali write '৬.৫ শতাংশের বেশি')."
        )
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"Clinical Guidelines Reference:\n{context_source}\n\nPatient Case Profile:\n{patient_context}"}
            ],
            temperature=0.2,
            max_tokens=1000,
        )
        return completion.choices[0].message.content, guidelines_sources
    except Exception as e:
        return f"Error generating Agent insights: {e}", guidelines_sources

def get_english_prescription_insights(patient_context):
    context_source = "\n".join(guidelines_texts)
    try:
        client = Groq(api_key=GROQ_API_KEY)
        system_content = (
            "You are an expert Medical AI Agent. Generate a highly concise, professional, point-by-point prescription plan in English. "
            "Format the output strictly as a clean, structured bulleted list with these exact section headers: "
            "'DIAGNOSTIC ADVICE:', 'DIETARY MODIFICATIONS:', and 'LIFESTYLE PROTOCOL:'. "
            "Keep each point short, precise, and practical. Do not use formatting markdown symbols like asterisks or hashtags. "
            "STRICT RULE: Never use mathematical symbols like greater than, less than, percentage signs, dollar signs, or brackets. Write them as plain words (e.g., 'percent', 'greater than')."
        )
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"Clinical Guidelines Reference:\n{context_source}\n\nPatient Case Profile:\n{patient_context}"}
            ],
            temperature=0.1,
            max_tokens=500,
        )
        return completion.choices[0].message.content
    except Exception:
        return "DIAGNOSTIC ADVICE:\n- Order HbA1c and Fasting Blood Sugar tests.\nDIETARY MODIFICATIONS:\n- Restrict daily carbohydrates intake.\nLIFESTYLE PROTOCOL:\n- Engage in 150 minutes of moderate exercise weekly."

# --- CatBoost Model Loading ---
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, "final_catboost_modol.cbm")
    try:
        if os.path.exists(model_path):
            model.load_model(model_path)
            return model
        return None
    except Exception:
        return None

model = load_model()

# --- 📄 PDF Prescription with References ---
def generate_prescription_pdf(patient_name, patient_data, result_text, confidence, english_agent_report, sources_list):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    styles = getSampleStyleSheet()
    
    header_text = "AI DOCTOR SCREENING SYSTEM"
    sub_header_text = "Automated Clinical Decision Support & Screening"
    section1_text = "PATIENT CLINICAL CASE HISTORY"
    col1_title = "Symptom / Metric"
    col2_title = "Patient Status"
    section2_text = "STATISTICAL RISK ASSESSMENT (CATBOOST ENGINE)"
    risk_label = "Risk Evaluation:"
    conf_label = "Algorithmic Confidence Level:"
    section3_text = "Rx — CLINICAL GUIDELINE & ACTION PLAN"
    section4_text = "REFERENCES (Guidelines Consulted)"
    warning_text = "Warning: This AI Doctor decision is not final. It is a preliminary screening report. Please consult a registered medical practitioner for formal diagnosis and treatment."

    header_style = ParagraphStyle('HeaderStyle', parent=styles['Heading1'], fontSize=22, textColor=colors.HexColor('#dc3545'), alignment=1, spaceAfter=5, fontName='Helvetica-Bold')
    sub_header_style = ParagraphStyle('SubHeaderStyle', parent=styles['Normal'], fontSize=10, textColor=colors.HexColor('#555555'), alignment=1, spaceAfter=15, fontName='Helvetica')
    section_style = ParagraphStyle('SectionStyle', parent=styles['Heading2'], fontSize=13, textColor=colors.HexColor('#0056b3'), spaceBefore=12, spaceAfter=6, fontName='Helvetica-Bold')
    body_style = ParagraphStyle('BodyStyle', parent=styles['Normal'], fontSize=10, leading=15, textColor=colors.HexColor('#222222'), fontName='Helvetica')
    ref_style = ParagraphStyle('RefStyle', parent=styles['Normal'], fontSize=9, leading=12, textColor=colors.HexColor('#555555'), fontName='Helvetica')
    alert_style = ParagraphStyle('AlertStyle', parent=styles['Normal'], fontSize=9, leading=14, textColor=colors.HexColor('#bd2130'), fontName='Helvetica-Bold', alignment=1)
    
    story.append(Paragraph(header_text, header_style))
    story.append(Paragraph(sub_header_text, sub_header_style))
    story.append(Table([[""]], colWidths=[530], rowHeights=[2], style=TableStyle([('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#dc3545'))])))
    story.append(Spacer(1, 15))
    story.append(Paragraph(section1_text, section_style))
    
    data = [[col1_title, col2_title], ["Patient Name", str(patient_name)]]
    for k, v in patient_data.items():
        v_final = "Present (Yes)" if str(v) == "Yes" else ("Absent (No)" if str(v) == "No" else str(v))
        data.append([str(k), v_final])
        
    t = Table(data, colWidths=[265, 265])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (1,0), colors.HexColor('#f8f9fa')),
        ('TEXTCOLOR', (0,0), (1,0), colors.HexColor('#111111')),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTNAME', (0,0), (1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dddddd')),
    ]))
    story.append(t)
    story.append(Spacer(1, 15))
    story.append(Paragraph(section2_text, section_style))
    pdf_verdict = "DIABETES RISK DETECTED" if ("DETECTED" in result_text or "সনাক্ত" in result_text) else "NO IMMEDIATE RISK DETECTED"
    verdict_color = '#dc3545' if "DETECTED" in pdf_verdict else '#28a745'
    verdict_html = f"<font color='{verdict_color}'><b>{pdf_verdict}</b></font>"
    story.append(Paragraph(f"<b>{risk_label}</b> {verdict_html}", body_style))
    story.append(Paragraph(f"<b>{conf_label}</b> {confidence}", body_style))
    story.append(Spacer(1, 15))
    story.append(Paragraph(section3_text, section_style))
    
    clean_report = english_agent_report.replace("**", "").replace("###", "").replace("*", "-")
    clean_report = clean_report.replace(">", " greater than ").replace("<", " less than ")
    clean_report = clean_report.replace("%", " percent ").replace("$", "")
    
    for para in clean_report.split("\n"):
        if para.strip():
            story.append(Paragraph(para.strip(), body_style))
            story.append(Spacer(1, 4))
    
    # --- References Section ---
    story.append(Spacer(1, 20))
    story.append(Paragraph(section4_text, section_style))
    for idx, src in enumerate(sources_list, 1):
        story.append(Paragraph(f"{idx}. {src}", ref_style))
        story.append(Spacer(1, 4))
            
    story.append(Spacer(1, 25))
    story.append(Table([[""]], colWidths=[530], rowHeights=[1], style=TableStyle([('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#cccccc'))])))
    story.append(Spacer(1, 15))
    story.append(Paragraph(warning_text, alert_style))
    doc.build(story)
    buffer.seek(0)
    return buffer

# --- 🎨 Enhanced CSS with Animations ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,600;0,700;1,400&display=swap');

    html, body, .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(145deg, #f8faff 0%, #eef4fa 100%);
        min-height: 100vh;
    }

    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulseGlow {
        0% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0.4); }
        70% { box-shadow: 0 0 0 12px rgba(220, 53, 69, 0); }
        100% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0); }
    }
    @keyframes slideInLeft {
        0% { opacity: 0; transform: translateX(-20px); }
        100% { opacity: 1; transform: translateX(0); }
    }
    @keyframes slideInRight {
        0% { opacity: 0; transform: translateX(20px); }
        100% { opacity: 1; transform: translateX(0); }
    }

    .fade-in { animation: fadeInUp 0.5s ease-out forwards; }
    .slide-left { animation: slideInLeft 0.4s ease-out forwards; }
    .slide-right { animation: slideInRight 0.4s ease-out forwards; }
    .pulse-glow { animation: pulseGlow 2s infinite; }

    .main-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 1rem;
    }

    .app-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #0b3954, #1e6f9f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        font-size: 1rem;
        color: #2c3e50;
        opacity: 0.8;
        font-weight: 400;
    }

    .chat-bubble-ai {
        background: #ffffff;
        padding: 14px 20px;
        border-radius: 18px 18px 18px 4px;
        border-left: 5px solid #dc3545;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
        margin-bottom: 16px;
        font-size: calc(15px + 0.1vw);
        line-height: 1.6;
        max-width: 100%;
        transition: all 0.2s ease;
        animation: fadeInUp 0.4s ease-out;
    }
    .chat-bubble-ai:hover {
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    }

    .chat-bubble-user {
        background: linear-gradient(135deg, #dc3545, #c82333);
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        text-align: left;
        margin-bottom: 16px;
        display: inline-block;
        float: right;
        clear: both;
        font-size: calc(15px + 0.1vw);
        max-width: 85%;
        box-shadow: 0 4px 12px rgba(220, 53, 69, 0.25);
        animation: slideInRight 0.4s ease-out;
        transition: all 0.2s ease;
    }

    .stForm {
        background: rgba(255,255,255,0.7);
        backdrop-filter: blur(10px);
        padding: 1.5rem !important;
        border-radius: 24px !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.04);
        border: 1px solid rgba(255,255,255,0.5);
        transition: all 0.3s ease;
    }
    .stForm:hover {
        box-shadow: 0 12px 36px rgba(0,0,0,0.06);
    }

    .stButton > button {
        background: linear-gradient(135deg, #1e6f9f, #0b3954);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.25s ease;
        box-shadow: 0 4px 12px rgba(27, 94, 140, 0.3);
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(27, 94, 140, 0.4);
        background: linear-gradient(135deg, #1a5f85, #082b3f);
    }
    .stButton > button:active {
        transform: scale(0.97);
    }

    .stRadio > div {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        justify-content: center;
    }
    .stRadio label {
        background: #f0f4f9;
        padding: 0.6rem 1.2rem;
        border-radius: 40px;
        font-weight: 500;
        color: #1e2a3a;
        transition: all 0.2s ease;
        border: 2px solid transparent;
        cursor: pointer;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    }
    .stRadio label:hover {
        background: #e2eaf2;
        border-color: #b0c4d9;
    }
    .stRadio [data-testid="stRadio"] > label[data-selected="true"] {
        background: linear-gradient(135deg, #1e6f9f, #0b3954);
        color: white;
        border-color: #0b3954;
        box-shadow: 0 4px 14px rgba(27, 94, 140, 0.35);
    }

    .stNumberInput input {
        border-radius: 30px !important;
        border: 2px solid #dce5ed !important;
        padding: 0.6rem 1rem !important;
        font-size: 1rem !important;
        transition: border-color 0.3s ease;
    }
    .stNumberInput input:focus {
        border-color: #1e6f9f !important;
        box-shadow: 0 0 0 3px rgba(30, 111, 159, 0.2);
    }

    .css-1d391kg {
        background: #ffffffd9;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(0,0,0,0.05);
    }
    .sidebar-content {
        padding: 1rem 0.5rem;
    }

    .stProgress > div > div {
        background: linear-gradient(90deg, #0b3954, #1e6f9f) !important;
        border-radius: 30px;
    }

    .metric-card {
        background: white;
        border-radius: 20px;
        padding: 1rem 1.2rem;
        box-shadow: 0 6px 18px rgba(0,0,0,0.03);
        transition: all 0.3s ease;
        border: 1px solid rgba(0,0,0,0.02);
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 28px rgba(0,0,0,0.06);
    }

    .report-box {
        background: white;
        padding: 1.8rem;
        border-radius: 24px;
        border-left: 6px solid #1e6f9f;
        box-shadow: 0 8px 24px rgba(0,0,0,0.03);
        font-size: 0.95rem;
        line-height: 1.7;
        animation: fadeInUp 0.6s ease-out;
    }

    .ref-box {
        background: #f8faff;
        padding: 1.2rem 1.5rem;
        border-radius: 16px;
        border-left: 4px solid #6c757d;
        margin-top: 1.5rem;
        font-size: 0.9rem;
        line-height: 1.6;
        animation: fadeInUp 0.6s ease-out;
    }
    .ref-box li {
        list-style-type: decimal;
        margin-left: 1.5rem;
        color: #2c3e50;
    }
    .ref-box a {
        color: #1e6f9f;
        text-decoration: none;
    }

    .warning-box {
        background: #fff9e6;
        color: #8a6d3b;
        padding: 1rem 1.5rem;
        border-radius: 16px;
        border-left: 6px solid #ffc107;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(255, 193, 7, 0.15);
        animation: pulseGlow 2.5s infinite;
    }

    .rag-source {
        background: #f0f3f8;
        padding: 0.8rem 1.2rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-family: 'Courier New', monospace;
        color: #2c3e50;
        border: 1px solid #dce5ed;
        overflow-x: auto;
    }

    .stDownloadButton button {
        background: linear-gradient(135deg, #28a745, #1e7e34) !important;
        box-shadow: 0 4px 14px rgba(40, 167, 69, 0.3);
        border-radius: 50px;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        transition: all 0.25s ease;
    }
    .stDownloadButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(40, 167, 69, 0.4);
        background: linear-gradient(135deg, #218838, #155d27) !important;
    }

    @media (max-width: 640px) {
        .app-title { font-size: 1.6rem; }
        .chat-bubble-ai, .chat-bubble-user { font-size: 14px; padding: 10px 14px; }
        .stForm { padding: 1rem !important; }
        .stButton > button { font-size: 0.9rem; padding: 0.5rem 1.2rem; }
        .main-container { padding: 0.5rem; }
        .stRadio label { padding: 0.4rem 1rem; font-size: 0.9rem; }
    }

    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #eef2f7; border-radius: 10px; }
    ::-webkit-scrollbar-thumb { background: #b0c4d9; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: #8aa0b9; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# --- Language Selection Sidebar ---
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.image("https://img.icons8.com/fluency/96/000000/doctor.png", width=80)
    st.markdown("## 🌐 Language / ভাষা")
    lang = st.radio("Select Chat Language", ["English", "বাংলা"], index=0, label_visibility="collapsed")
    st.markdown("---")
    st.caption("🩸 Early Diabetes Screening Assistant")
    st.caption("Powered by AI • Clinical Guidelines • CatBoost")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Questions Definition ---
questions = [
    {"field": "Age", "en": "What is your Age?", "bn": "আপনার বয়স কত?"},
    {"field": "Gender", "en": "What is your Gender?", "bn": "আপনার লিঙ্গ কী?", "options": ["Male", "Female"]},
    {"field": "Polyuria", "en": "Do you experience excessive urination (Polyuria)?", "bn": "আপনার কি অতিরিক্ত প্রস্রাবের সমস্যা (Polyuria) হচ্ছে?", "options": ["Yes", "No"]},
    {"field": "Polydipsia", "en": "Do you feel excessively thirsty (Polydipsia)?", "bn": "আপনার কি অতিরিক্ত তৃষ্ণা (Polydipsia) পায়?", "options": ["Yes", "No"]},
    {"field": "Irritability", "en": "Have you been feeling unusually irritable lately?", "bn": "আপনি কি ইদানীং খিটখিটে মেজাজ অনুভব করছেন?", "options": ["Yes", "No"]},
    {"field": "Itching", "en": "Do you have frequent skin itching?", "bn": "আপনার শরীরে কি ঘন ঘন চুলকানির সমস্যা হচ্ছে?", "options": ["Yes", "No"]},
    {"field": "delayed healing", "en": "Do your wounds or cuts take a long time to heal?", "bn": "আপনার শরীরে কোনো ক্ষত বা কাটা শুকাতে কি স্বাভাবিকের চেয়ে বেশি সময় লাগে?", "options": ["Yes", "No"]},
    {"field": "Alopecia", "en": "Are you experiencing significant hair loss (Alopecia)?", "bn": "আপনার কি অতিরিক্ত চুল পড়ে যাওয়ার (Alopecia) সমস্যা হচ্ছে?", "options": ["Yes", "No"]}
]

# --- Session State Initializations ---
if "step" not in st.session_state:
    st.session_state.step = -2
if "patient_name" not in st.session_state:
    st.session_state.patient_name = ""
if "user_responses" not in st.session_state:
    st.session_state.user_responses = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Helper functions ---
def add_chat(role, text):
    st.session_state.chat_history.append({"role": role, "text": text})

def transition_to(new_step):
    st.session_state.step = new_step
    st.rerun()

# --- Main App Render ---
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Title
st.markdown("""
    <div class="fade-in">
        <span class="app-title">🩸 DECat‑AI</span>
        <p class="app-subtitle">Early Diabetes Screening • Intelligent Clinical Support</p>
    </div>
""", unsafe_allow_html=True)
st.markdown("---")

# Chat history
chat_container = st.container()
with chat_container:
    for chat in st.session_state.chat_history:
        if chat.get("role") == "ai":
            st.markdown(f'<div class="chat-bubble-ai">🤖 <b>DECat-AI:</b> {chat.get("text", "")}</div>', unsafe_allow_html=True)
        elif chat.get("role") == "user":
            st.markdown(f'<div style="width:100%; overflow:auto;"><div class="chat-bubble-user">👤 {chat.get("text", "")}</div></div>', unsafe_allow_html=True)

# --- STEP -2: ASK FOR PATIENT NAME ---
if st.session_state.step == -2:
    welcome_init = (
        "Hello! Welcome to our Early Diabetes Screening Desk. I am your AI Doctor, DECat-AI. Before we begin, may I know your name please?"
        if lang == "English" else
        "হ্যালো! আমাদের আর্লি ডায়াবেটিস স্ক্রিনিং ডেস্কে আপনাকে স্বাগত। আমি আপনার এআই ডাক্তার, DECat-AI। আমাদের পরীক্ষা শুরু করার আগে, আমি কি আপনার নামটা জানতে পারি?"
    )
    st.markdown(f'<div class="chat-bubble-ai slide-left">🤖 <b>DECat-AI:</b> {welcome_init}</div>', unsafe_allow_html=True)
    
    # ===== FORM 1: NAME =====
    with st.form(key="form_name_step", clear_on_submit=False):
        name_input = st.text_input("Enter your name..." if lang == "English" else "আপনার নাম লিখুন...", key="name_input_field")
        # ✅ Submit button for name form
        submit_name = st.form_submit_button("Next ➡️" if lang == "English" else "পরবর্তী ➡️")
        
        if submit_name and name_input.strip():
            st.session_state.patient_name = name_input.strip()
            add_chat("ai", welcome_init)
            add_chat("user", name_input.strip())
            
            welcome_back = (
                f"Nice to meet you, {st.session_state.patient_name}! I am DECat-AI. How are you doing today? Do you have any health concerns? "
                f"By the way, can I start your early diabetes screening test now?"
                if lang == "English" else
                f"আপনার সাথে পরিচিত হয়ে ভালো লাগলো, {st.session_state.patient_name}! আমি DECat-AI। আজ আপনি কেমন আছেন? আপনার কি কোনো স্বাস্থ্য সমস্যা হচ্ছে? "
                f"ভালো কথা, আমি কি আপনার ডায়াবেটিস স্ক্রিনিং টেস্টটি এখন শুরু করতে পারি?"
            )
            add_chat("ai", welcome_back)
            transition_to(-1)

# --- STEP -1: NATURAL CHAT & INTENT TRIGGER ---
elif st.session_state.step == -1:
    # ===== FORM 2: CHAT =====
    with st.form(key="form_chat_step", clear_on_submit=True):
        user_msg = st.text_input("Ask me anything or say something..." if lang == "English" else "আমাকে যেকোনো প্রশ্ন করুন বা কিছু বলুন...", key="chat_input_field")
        # ✅ Submit button for chat form
        submit_chat = st.form_submit_button("Send 💬" if lang == "English" else "পাঠান 💬")
        
        if submit_chat and user_msg.strip():
            add_chat("user", user_msg.strip())
            
            positive_keywords = [
                "ha", "haa", "hay", "hoy", "yes", "y", "ok", "okay", "sure", "start", "go", "test", 
                "হ্যাঁ", "হ্যা", "হুম", "করুন", "করো", "শুরু", "ঠিক আছে", "চলুন", "হবে"
            ]
            if any(kw in user_msg.strip().lower() for kw in positive_keywords):
                ack = (
                    "Great! Let's begin the screening. I'll ask you a few questions about your health."
                    if lang == "English" else
                    "চমৎকার! তাহলে স্ক্রিনিং শুরু করা যাক। আমি আপনাকে আপনার স্বাস্থ্য সম্পর্কে কয়েকটি প্রশ্ন করব।"
                )
                add_chat("ai", ack)
                transition_to(0)
            else:
                with st.spinner("🤔 Thinking..."):
                    try:
                        client = Groq(api_key=GROQ_API_KEY)
                        chat_context = [
                            {
                                "role": "system", 
                                "content": (
                                    f"You are DECat-AI, a warm, natural and empathetic AI Doctor talking to {st.session_state.patient_name}. "
                                    f"Answer their questions or chats conversationally and concisely in {lang}. "
                                    f"At the end of your response, you MUST always ask them elegantly whether you can start the diabetes test now."
                                )
                            }
                        ]
                        for h in st.session_state.chat_history[-6:]:
                            chat_context.append({"role": "user" if h["role"] == "user" else "assistant", "content": h["text"]})
                            
                        reply = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=chat_context,
                            temperature=0.6,
                            max_tokens=250
                        )
                        ai_reply = reply.choices[0].message.content
                    except Exception:
                        ai_reply = "I see. Shall we start your early diabetes risk test now?" if lang == "English" else "বুঝতে পারলাম। আমরা কি এখন আপনার ডায়াবেটিস পরীক্ষাটি শুরু করতে পারি?"
                
                add_chat("ai", ai_reply)
                st.rerun()

# --- 📋 STEP 0 to N: MEDICAL QUESTIONNAIRE ---
elif 0 <= st.session_state.step < len(questions):
    current_q = questions[st.session_state.step]
    q_text = current_q["bn"] if lang == "বাংলা" else current_q["en"]
    
    st.markdown(f'<div class="chat-bubble-ai slide-left">🤖 <b>DECat-AI:</b> {q_text}</div>', unsafe_allow_html=True)
    
    # ===== FORM 3: MEDICAL QUESTIONS (each step is a separate form) =====
    with st.form(key=f"form_medical_step_{st.session_state.step}"):
        if "options" in current_q:
            opt_mapping = {"Male": "পুরুষ" if lang == "বাংলা" else "Male", "Female": "নারী" if lang == "বাংলা" else "Female", "Yes": "হ্যাঁ" if lang == "বাংলা" else "Yes", "No": "না" if lang == "বাংলা" else "No"}
            rev_mapping = {v: k for k, v in opt_mapping.items()}
            
            user_choice = st.radio("Choose one:", [opt_mapping[o] for o in current_q["options"]], index=None, label_visibility="collapsed", key=f"med_radio_{st.session_state.step}")
            # ✅ Submit button for medical radio form
            submit_btn = st.form_submit_button("Next ➡️" if lang == "English" else "পরবর্তী ➡️")
            
            if submit_btn:
                if user_choice is None:
                    st.error("Please select an option!" if lang == "English" else "দয়া করে একটি অপশন সিলেক্ট করুন!")
                else:
                    st.session_state.user_responses[current_q["field"]] = rev_mapping[user_choice]
                    add_chat("ai", q_text)
                    add_chat("user", user_choice)
                    st.session_state.step += 1
                    st.rerun()
        else:
            user_val = st.number_input("Enter your age:", min_value=1, max_value=120, value=None, placeholder="e.g. 35", label_visibility="collapsed", key=f"med_age_{st.session_state.step}")
            # ✅ Submit button for age number form
            submit_btn = st.form_submit_button("Next ➡️" if lang == "English" else "পরবর্তী ➡️")
            
            if submit_btn:
                if user_val is None:
                    st.error("Please enter your age!" if lang == "English" else "দয়া করে আপনার বয়স লিখুন!")
                else:
                    st.session_state.user_responses[current_q["field"]] = int(user_val)
                    add_chat("ai", q_text)
                    add_chat("user", str(int(user_val)))
                    st.session_state.step += 1
                    st.rerun()

# --- 📊 FINAL EVALUATION & REPORT ---
else:
    st.write("---")
    if model is None:
        st.error("❌ Model file (.cbm) missing. Cannot proceed with risk assessment.")
    else:
        res = st.session_state.user_responses
        input_df = pd.DataFrame([res])
        for col in input_df.columns:
            if col != 'Age':
                input_df[col] = input_df[col].astype('category')
                
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        is_positive = str(prediction) == "1" or prediction == 1 or str(prediction).lower() == "positive"
        score = probability[1] if is_positive else probability[0]
        
        verdict_str = ("ডায়াবেটিসের ঝুঁকি সনাক্ত হয়েছে" if is_positive else "কোনো তাত্ক্ষণিক ঝুঁকি পাওয়া যায়নি") if lang == "বাংলা" else ("DIABETES RISK DETECTED" if is_positive else "NO IMMEDIATE RISK DETECTED")
        confidence_str = f"{score * 100:.2f}%"

        st.markdown("## 📊 Analytics Summary")
        
        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            if is_positive:
                st.markdown(f'<div class="metric-card" style="border-left: 6px solid #dc3545;"><h3 style="color:#dc3545;">🚨 {verdict_str}</h3></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="metric-card" style="border-left: 6px solid #28a745;"><h3 style="color:#28a745;">✅ {verdict_str}</h3></div>', unsafe_allow_html=True)
            st.metric(label="Model Confidence", value=confidence_str)
        with col_res2:
            st.write("**Risk Probability Meter**")
            st.progress(float(score))
            
        active_symptoms = [k for k, v in res.items() if v == 'Yes']
        patient_case_context = f"Patient Name: {st.session_state.patient_name}\nAge: {res['Age']}, Gender: {res['Gender']}\nSymptoms: {', '.join(active_symptoms) if active_symptoms else 'None'}\nVerdict: {verdict_str} ({confidence_str})"
        
        with st.spinner("🩺 Consulting clinical guidelines..."):
            agent_report, sources = get_rag_agent_response(patient_case_context, lang)
        with st.spinner("📄 Preparing your prescription document..."):
            english_prescription_report = get_english_prescription_insights(patient_case_context)
            
        st.markdown("## 🤖 AI Doctor Assessment Report")
        st.markdown(f'<div class="report-box">{agent_report}</div>', unsafe_allow_html=True)
        
        # --- Display References ---
        with st.expander("📚 References (Guidelines Consulted)", expanded=True):
            ref_html = '<div class="ref-box"><ul>'
            for src in sources:
                ref_html += f'<li>{src}</li>'
            ref_html += '</ul></div>'
            st.markdown(ref_html, unsafe_allow_html=True)
        
        warning_text_display = "⚠️ Warning: Preliminary screening report only. Consult a doctor." if lang == "English" else "⚠️ সতর্কবার্তা: প্রাথমিক স্ক্রিনিং রিপোর্ট মাত্র। ডাক্তারের পরামর্শ নিন।"
        st.markdown(f'<div class="warning-box">{warning_text_display}</div>', unsafe_allow_html=True)
        
        st.write(" ")
        prescription_pdf = generate_prescription_pdf(
            st.session_state.patient_name, 
            res, 
            verdict_str, 
            confidence_str, 
            english_prescription_report,
            sources
        )
        st.download_button(
            label="📥 Download Prescription PDF",
            data=prescription_pdf,
            file_name=f"AI_Prescription_{st.session_state.patient_name}.pdf",
            mime="application/pdf"
        )

    st.write(" ")
    if st.button("🔄 Restart Assessment"):
        st.session_state.step = -2
        st.session_state.patient_name = ""
        st.session_state.user_responses = {}
        st.session_state.chat_history = []
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

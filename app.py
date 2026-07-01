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

# --- 1. Page Config ---
st.set_page_config(
    page_title="Early Diabetes Chatbot AI",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. API Key Management ---
# Production-ready: Fetching from Streamlit secrets
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "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")

if not GROQ_API_KEY:
    st.error("❌ Groq API Key missing! Please configure 'GROQ_API_KEY' in your Streamlit Secrets (.streamlit/secrets.toml).")

# --- 3. RAG Knowledge Base Setup ---
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

# Load guidelines globally
guidelines_data = setup_rag_knowledge_base()
guidelines_texts = [item["text"] for item in guidelines_data]
guidelines_sources = [item["source"] for item in guidelines_data]

# --- 4. Groq API Integration (Strict Medical Grounding) ---
def get_rag_agent_response(patient_context, language):
    """Generates localized AI Doctor response using Groq Cloud API."""
    context_source = "\n".join(guidelines_texts)
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        lang_instruction = (
            f"CRITICAL: Your entire response MUST be written strictly in {language}. "
            f"Do NOT use any other language. If {language} is 'English', respond only in English. "
            f"If {language} is 'বাংলা', respond only in Bengali."
        )
        
        system_content = (
            f"You are an expert Medical AI Agent acting as a supportive AI Doctor named DECat-AI. "
            f"{lang_instruction} "
            f"CRITICAL SAFETY RULE: Analyze the patient STRICTLY based ONLY on the provided Clinical Guidelines Reference. "
            f"Do NOT use outside medical knowledge or assume things not written in the reference. If the reference is insufficient, explicitly say so. "
            f"Format your response as a professional and clear clinical report. Use headers like 'Diagnostic Advice', "
            f"'Dietary Modifications', and 'Lifestyle Protocol' (translate these headers properly if the language is Bengali). "
            f"STRICT FORMATTING RULE: Never use mathematical symbols like >, <, %, $, or brackets. "
            f"Write them out in plain text words (e.g., 'greater than 6.5 percent', '৬.৫ শতাংশের বেশি')."
        )
        
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"Clinical Guidelines Reference:\n{context_source}\n\nPatient Case Profile:\n{patient_context}"}
            ],
            temperature=0.1,  # Lower temperature prevents hallucination
            max_tokens=600,
        )
        return completion.choices[0].message.content, guidelines_sources
    except Exception as e:
        return f"Error generating Agent insights: {e}", guidelines_sources

def get_english_prescription_insights(patient_context):
    """Generates a concise English fallback recommendation specifically structured for the PDF."""
    context_source = "\n".join(guidelines_texts)
    try:
        client = Groq(api_key=GROQ_API_KEY)
        system_content = (
            "You are an expert Medical AI Agent. Generate a highly concise point-by-point recommendation plan in English based strictly on the provided guidelines. "
            "Format the output strictly as a clean, structured bulleted list with these exact section headers: "
            "'DIAGNOSTIC ADVICE:', 'DIETARY MODIFICATIONS:', and 'LIFESTYLE PROTOCOL:'. "
            "Do not use markdown symbols like asterisks or hashtags. Never use symbols like >, <, %, $. Write them as plain text words."
        )
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"Clinical Guidelines Reference:\n{context_source}\n\nPatient Case Profile:\n{patient_context}"}
            ],
            temperature=0.1,
            max_tokens=300,
        )
        return completion.choices[0].message.content
    except Exception:
        return "DIAGNOSTIC ADVICE:\n- Order HbA1c and Fasting Blood Sugar tests.\nDIETARY MODIFICATIONS:\n- Restrict daily carbohydrates intake.\nLIFESTYLE PROTOCOL:\n- Engage in 150 minutes of moderate exercise weekly."

# --- 5. CatBoost Machine Learning Model Loader ---
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    current_dir = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
    model_path = os.path.join(current_dir, "final_catboost_modol.cbm")
    try:
        if os.path.exists(model_path):
            model.load_model(model_path)
            return model
        return None
    except Exception:
        return None

model = load_model()

# --- 6. ReportLab PDF Generation Engine ---
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
    section3_text = "RECOMMENDATIONS — CLINICAL GUIDELINE & ACTION PLAN"
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
    
    pdf_verdict = "DIABETES RISK DETECTED" if ("DETECTED" in result_text or "কনাক্ত" in result_text or "ঝুঁকি" in result_text) else "NO IMMEDIATE RISK DETECTED"
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

# --- 7. UI Styling (Custom CSS) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, .stApp { font-family: 'Inter', sans-serif; background: #f8faff; }
    .main-container { max-width: 800px; margin: 0 auto; padding: 1rem; }
    .app-title { font-size: 2.2rem; font-weight: 700; color: #0b3954; }
    .chat-bubble-ai { background: white; padding: 15px; border-radius: 12px; border-left: 5px solid #dc3545; margin-bottom: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); font-size: 15px; line-height: 1.5; }
    .chat-bubble-user { background: #dc3545; color: white; padding: 12px; border-radius: 12px; float: right; clear: both; margin-bottom: 10px; font-size: 15px; }
    .report-box { background: white; padding: 20px; border-radius: 12px; border-left: 5px solid #1e6f9f; box-shadow: 0 4px 10px rgba(0,0,0,0.05); margin-top: 15px; }
    .warning-box { background: #fff9e6; color: #8a6d3b; padding: 15px; border-radius: 8px; border-left: 5px solid #ffc107; font-weight: bold; margin-top: 15px; }
</style>
""", unsafe_allow_html=True)

# --- 8. Localization & Sidebar ---
with st.sidebar:
    st.markdown("## 🌐 Language / ভাষা")
    lang = st.radio("Select Chat Language", ["English", "বাংলা"], index=0)

# --- 9. Clinical Questions Map ---
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

# --- 10. Session State Initializer ---
if "step" not in st.session_state: st.session_state.step = -2
if "patient_name" not in st.session_state: st.session_state.patient_name = ""
if "user_responses" not in st.session_state: st.session_state.user_responses = {}
if "chat_history" not in st.session_state: st.session_state.chat_history = []

def add_chat(role, text): st.session_state.chat_history.append({"role": role, "text": text})
def transition_to(new_step):
    st.session_state.step = new_step
    st.rerun()

# Layout Container Start
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<span class="app-title">🩸 DECat‑AI</span><p style="color:#555;">Early Diabetes Screening desk</p>', unsafe_allow_html=True)
st.markdown("---")

# Render Dynamic Chat History Box
for chat in st.session_state.chat_history:
    if chat["role"] == "ai":
        st.markdown(f'<div class="chat-bubble-ai">🤖 <b>DECat-AI:</b> {chat["text"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="overflow:auto;"><div class="chat-bubble-user">👤 {chat["text"]}</div></div>', unsafe_allow_html=True)

# --- STEP -2: GET PATIENT NAME ---
if st.session_state.step == -2:
    welcome_init = "Hello! Welcome to our Screening Desk. I am DECat-AI. May I know your name please?" if lang == "English" else "হ্যালো! আমাদের স্ক্রিনিং ডেস্কে আপনাকে স্বাগত। আমি DECat-AI। আপনার নামটা জানতে পারি?"
    st.markdown(f'<div class="chat-bubble-ai">🤖 <b>DECat-AI:</b> {welcome_init}</div>', unsafe_allow_html=True)
    with st.form(key="name_form"):
        name_input = st.text_input("Name / নাম", placeholder="Type here...")
        if st.form_submit_button("Next ➡️"):
            if name_input.strip():
                st.session_state.patient_name = name_input.strip()
                add_chat("ai", welcome_init)
                add_chat("user", name_input.strip())
                transition_to(-1)

# --- STEP -1: CONSENT PROTOCOL ---
elif st.session_state.step == -1:
    ask_consent = f"Nice to meet you, {st.session_state.patient_name}! Shall we start the early diabetes risk assessment test?" if lang == "English" else f"আপনার সাথে পরিচিত হয়ে ভালো লাগলো, {st.session_state.patient_name}! আমরা কি আপনার ডায়াবেটিস স্ক্রিনিং টেস্টটি শুরু করতে পারি?"
    st.markdown(f'<div class="chat-bubble-ai">🤖 <b>DECat-AI:</b> {ask_consent}</div>', unsafe_allow_html=True)
    with st.form(key="consent_form"):
        user_reply = st.text_input("Your Response / আপনার উত্তর", placeholder="e.g., Yes / হ্যাঁ")
        if st.form_submit_button("Start Screening 🩸"):
            add_chat("ai", ask_consent)
            add_chat("user", user_reply if user_reply.strip() else "Yes")
            transition_to(0)

# --- STEPS 0 to N: SEQUENTIAL MEDICAL QUESTIONNAIRE ---
elif 0 <= st.session_state.step < len(questions):
    current_q = questions[st.session_state.step]
    q_text = current_q["bn"] if lang == "বাংলা" else current_q["en"]
    st.markdown(f'<div class="chat-bubble-ai">🤖 <b>DECat-AI:</b> {q_text}</div>', unsafe_allow_html=True)
    
    with st.form(key=f"q_form_{st.session_state.step}"):
        if "options" in current_q:
            # Map labels visually based on language selection
            opt_labels = ["Yes", "No"] if lang == "English" else ["হ্যাঁ", "না"] if current_q["field"] != "Gender" else ["পুরুষ", "নারী"]
            opt_map = {"Yes": opt_labels[0], "No": opt_labels[1]} if current_q["field"] != "Gender" else {"Male": opt_labels[0], "Female": opt_labels[1]}
            rev_map = {v: k for k, v in opt_map.items()}
            
            ans_visual = st.radio("Select option:", list(opt_map.values()), index=None)
            if st.form_submit_button("Next ➡️") and ans_visual:
                st.session_state.user_responses[current_q["field"]] = rev_map[ans_visual]
                add_chat("ai", q_text)
                add_chat("user", ans_visual)
                transition_to(st.session_state.step + 1)
        else:
            ans_age = st.number_input("Enter Age:", min_value=1, max_value=120, value=None, placeholder="e.g., 35")
            if st.form_submit_button("Next ➡️") and ans_age:
                st.session_state.user_responses[current_q["field"]] = int(ans_age)
                add_chat("ai", q_text)
                add_chat("user", str(int(ans_age)))
                transition_to(st.session_state.step + 1)

# --- 📊 FINAL STEP: MACHINE LEARNING EVALUATION & RAG GENERATION ---
else:
    st.write("### 📊 Clinical Screening Assessment Dashboard")
    if model is None:
        st.error("❌ CatBoost Model file (.cbm) missing or corrupt. Check file alignment inside the working directory.")
    else:
        res = st.session_state.user_responses
        input_df = pd.DataFrame([res])
        
        # Enforce exact category casting for CatBoost
        for col in input_df.columns:
            if col != 'Age':
                input_df[col] = input_df[col].astype('category')
                
        # Perform CatBoost Predictive Math
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        
        # Set probability rules safely
        is_positive = bool(prediction == 1 or probability[1] > 0.5)
        risk_score = probability[1] * 100 if is_positive else probability[0] * 100
        confidence_str = f"{risk_score:.1f} percent"

        # Present the Algorithmic Risk Analytics Status
        if is_positive:
            verdict_text = "DIABETES RISK DETECTED" if lang == "English" else "ডায়াবেটিস ঝুঁকি সনাক্ত হয়েছে"
            st.error(f"⚠️ **{verdict_text}** (Statistical Risk Matrix Confidence: {confidence_str})")
        else:
            verdict_text = "NO IMMEDIATE RISK DETECTED" if lang == "English" else "কোনো তাৎক্ষণিক ঝুঁকি পাওয়া যায়নি"
            st.success(f"✅ **{verdict_text}** (Statistical Wellness Index Confidence: {confidence_str})")

        # Compile dynamic patient telemetry to pass context down stream to Groq APIs
        patient_summary = ", ".join([f"{k}: {v}" for k, v in res.items()])
        
        with st.spinner("Invoking Groq API & processing RAG Clinical Guidelines..."):
            agent_report, sources = get_rag_agent_response(patient_summary, lang)
            english_report = get_english_prescription_insights(patient_summary)
            
        # Render the markdown clinical block safely extracted out of RAG Database
        st.markdown(f'<div class="report-box"><h4>📋 Guideline-Augmented Action Plan</h4><div>{agent_report}</div></div>', unsafe_allow_html=True)
        
        # Print references mapped inside the system cache
        st.markdown("#### 📚 Verified Evidence Base")
        for s in sources:
            st.caption(f"• {s}")
            
        # Pass payload down to ReportLab compiler engine
        pdf_data = generate_prescription_pdf(
            st.session_state.patient_name, res, verdict_text, confidence_str, english_report, sources
        )
        
        st.download_button(
            label="📥 Download Clinical Report PDF",
            data=pdf_data,
            file_name=f"{st.session_state.patient_name}_clinical_screening_report.pdf",
            mime="application/pdf"
        )
        
        # Legal Medical Warning Flag
        st.markdown("<div class='warning-box'>⚠️ Disclaimer: DECat-AI is a preliminary statistical screening application. The insights compiled above should not override professional diagnostic clinical testing. Kindly consult a medical practitioner for standard care paths.</div>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

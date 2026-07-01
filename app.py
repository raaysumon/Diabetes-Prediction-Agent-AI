import os
import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
from groq import Groq
import re

# ReportLab Components for Professional Prescription Layout
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io

# --- Page Config ---
st.set_page_config(
    page_title="Early Diabetes Chatbot AI",
    page_icon="🩸",
    layout="wide"
)

# --- API Key ---
GROQ_API_KEY = "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"

# --- 📚 RAG Knowledge Base Setup ---
@st.cache_resource
def setup_rag_knowledge_base():
    clinical_guidelines = [
        "Guideline: Polyuria (frequent urination) and Polydipsia (excessive thirst) are key indicators of high blood glucose. Immediate tests required: HbA1c (>6.5% indicates diabetes) and Fasting Blood Sugar (FBS >126 mg/dL).",
        "Guideline: Delayed healing of wounds or cuts indicates microvascular complications often related to prolonged hyperglycemia. Patients must be screened for peripheral neuropathy and HbA1c.",
        "Guideline: Diabetes management for high risk includes lifestyle changes: reducing carbohydrate intake to less than 45% of daily calories, engaging in 150 minutes of moderate exercise per week, and weight monitoring.",
        "Guideline: For low risk or negative diabetes risk screen, routine wellness checkup including annual HbA1c and fasting glucose is recommended, especially for adults above 35 years old.",
        "Guideline: Symptoms like Irritability, Alopecia (hair loss), and skin Itching can be secondary systemic signs of metabolic changes or poor circulation linked with early insulin resistance."
    ]
    return clinical_guidelines

guidelines_db = setup_rag_knowledge_base()

def get_rag_agent_response(patient_context, language):
    context_source = "\n".join(guidelines_db)
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
        return completion.choices[0].message.content, context_source
    except Exception as e:
        return f"Error generating Agent insights: {e}", ""

def get_english_prescription_insights(patient_context):
    context_source = "\n".join(guidelines_db)
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

# --- 📄 ALWAYS ENGLISH PDF Prescription Pad Generation ---
def generate_prescription_pdf(patient_name, patient_data, result_text, confidence, english_agent_report):
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
    warning_text = "Warning: This AI Doctor decision is not final. It is a preliminary screening report. Please consult a registered medical practitioner for formal diagnosis and treatment."

    header_style = ParagraphStyle('HeaderStyle', parent=styles['Heading1'], fontSize=22, textColor=colors.HexColor('#dc3545'), alignment=1, spaceAfter=5, fontName='Helvetica-Bold')
    sub_header_style = ParagraphStyle('SubHeaderStyle', parent=styles['Normal'], fontSize=10, textColor=colors.HexColor('#555555'), alignment=1, spaceAfter=15, fontName='Helvetica')
    section_style = ParagraphStyle('SectionStyle', parent=styles['Heading2'], fontSize=13, textColor=colors.HexColor('#0056b3'), spaceBefore=12, spaceAfter=6, fontName='Helvetica-Bold')
    body_style = ParagraphStyle('BodyStyle', parent=styles['Normal'], fontSize=10, leading=15, textColor=colors.HexColor('#222222'), fontName='Helvetica')
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
            
    story.append(Spacer(1, 25))
    story.append(Table([[""]], colWidths=[530], rowHeights=[1], style=TableStyle([('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#cccccc'))])))
    story.append(Spacer(1, 15))
    
    story.append(Paragraph(warning_text, alert_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# --- 📱 Custom Responsive CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #f4f6f9; }
    .chat-bubble-ai { 
        background-color: #ffffff; padding: 12px 16px; border-radius: 14px; 
        border-left: 5px solid #dc3545; box-shadow: 0 2px 4px rgba(0,0,0,0.04); 
        margin-bottom: 12px; font-size: calc(14px + 0.15vw); line-height: 1.5; max-width: 100%;
    }
    .chat-bubble-user { 
        background-color: #dc3545; color: white; padding: 10px 16px; border-radius: 14px; 
        text-align: left; margin-bottom: 12px; display: inline-block; float: right; 
        clear: both; font-size: calc(14px + 0.15vw); max-width: 85%;
    }
    .report-box { background-color: #eef2f7; padding: 18px; border-left: 5px solid #0056b3; border-radius: 10px; margin-top: 15px; }
    .warning-box { background-color: #fff3cd; color: #856404; padding: 14px; border-left: 5px solid #ffc107; border-radius: 8px; margin-top: 15px; font-weight: bold;}
    .rag-source { background-color: #e2e3e5; padding: 12px; border-radius: 6px; font-size: 12px; font-family: monospace; }
    div[data-testid="stForm"] { border: none !important; padding: 0 !important; }
    div.stDownloadButton button { width: 100% !important; }
    </style>
    """, unsafe_allow_html=True)

# --- Language Selection Sidebar ---
st.sidebar.title("🌐 Language / ভাষা")
lang = st.sidebar.radio("Select Chat Language", ["English", "বাংলা"], index=0)

# --- Questions Definition ---
questions = [
    {"field": "Age", "en": "What is your Age?", "bn": "আপনার বয়স কত?"},
    {"field": "Gender", "en": "What is your Gender?", "bn": "আপনার লিঙ্গ কী?", "options": ["Male", "Female"]},
    {"field": "Polyuria", "en": "Do you experience excessive urination (Polyuria)?", "bn": "আপনার কি অতিরিক্ত প্রস্রাবের সমস্যা (Polyuria) হচ্ছে?", "options": ["Yes", "No"]},
    {"field": "Polydipsia", "en": "Do you feel excessively thirsty (Polydipsia)?", "bn": "আপনার কি অতিরিক্ত তৃষ্ণা (Polydipsia) পায়?", "options": ["Yes", "No"]},
    {"field": "Irritability", "en": "Have you been feeling unusually irritable lately?", "bn": "আপনি কি ইদানীং খিটখীটে মেজাজ অনুভব করছেন?", "options": ["Yes", "No"]},
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

# --- Global Flow Control Flag ---
should_rerun = False

# --- Main App Render ---
st.title("🩸 Early Diabetes Conversational AI Agent" if lang == "English" else "🩸 ডায়াবেটিস চ্যাটবট এআই充জেন্ট")
st.markdown("---")

with st.container():
    # Render Chat History safely
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
        st.markdown(f'<div class="chat-bubble-ai">🤖 <b>DECat-AI:</b> {welcome_init}</div>', unsafe_allow_html=True)
        
        with st.form(key="form_name_step"):
            name_input = st.text_input("Enter your name..." if lang == "English" else "আপনার নাম লিখুন...", key="name_input_field")
            submit_name = st.form_submit_button("Next ➡️" if lang == "English" else "পরবর্তী ➡️")
            
            if submit_name and name_input.strip() != "":
                st.session_state.patient_name = name_input.strip()
                st.session_state.chat_history.append({"role": "ai", "text": welcome_init})
                st.session_state.chat_history.append({"role": "user", "text": name_input.strip()})
                
                welcome_back = (
                    f"Nice to meet you, {st.session_state.patient_name}! I am DECat-AI. How are you doing today? Do you have any health concerns? "
                    f"By the way, can I start your early diabetes screening test now?"
                    if lang == "English" else
                    f"আপনার সাথে পরিচিত হয়ে ভালো লাগলো, {st.session_state.patient_name}! আমি DECat-AI। আজ আপনি কেমন আছেন? আপনার কি কোনো স্বাস্থ্য সমস্যা হচ্ছে? "
                    f"ভালো কথা, আমি কি আপনার ডায়াবেটিস স্ক্রিনিং টেস্টটি এখন শুরু করতে পারি?"
                )
                st.session_state.chat_history.append({"role": "ai", "text": welcome_back})
                st.session_state.step = -1
                should_rerun = True

    # --- STEP -1: NATURAL CHAT & SEAMLESS AUTOMATIC INTENT TRIGGER ---
    elif st.session_state.step == -1:
        with st.form(key="form_chat_step", clear_on_submit=True):
            user_msg = st.text_input("Ask me anything or say something..." if lang == "English" else "আমাকে যেকোনো প্রশ্ন করুন বা কিছু বলুন...", key="chat_input_field")
            submit_chat = st.form_submit_button("Send 💬" if lang == "English" else "পাঠান 💬")
            
            if submit_chat and user_msg.strip() != "":
                st.session_state.chat_history.append({"role": "user", "text": user_msg})
                
                text_clean = user_msg.strip().lower()
                positive_keywords = [
                    "ha", "haa", "hay", "hoy", "yes", "y", "ok", "okay", "sure", "start", "go", "test", 
                    "হ্যাঁ", "হ্যা", "হুম", "করুন", "করো", "শুরু", "ঠিক আছে", "চলুন", "হবে"
                ]
                
                if any(kw in text_clean for kw in positive_keywords):
                    st.session_state.step = 0
                    should_rerun = True
                else:
                    with st.spinner("Thinking..."):
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
                    
                    st.session_state.chat_history.append({"role": "ai", "text": ai_reply})
                    should_rerun = True

    # --- 📋 STEP 0 to N: MEDICAL QUESTIONNAIRE ---
    elif 0 <= st.session_state.step < len(questions):
        current_q = questions[st.session_state.step]
        q_text = current_q["bn"] if lang == "বাংলা" else current_q["en"]
        
        st.markdown(f'<div class="chat-bubble-ai">🤖 <b>DECat-AI:</b> {q_text}</div>', unsafe_allow_html=True)
        
        with st.form(key=f"form_medical_step_{st.session_state.step}"):
            if "options" in current_q:
                opt_mapping = {"Male": "পুরুষ" if lang == "বাংলা" else "Male", "Female": "নারী" if lang == "বাংলা" else "Female", "Yes": "হ্যাঁ" if lang == "বাংলা" else "Yes", "No": "না" if lang == "বাংলা" else "No"}
                rev_mapping = {v: k for k, v in opt_mapping.items()}
                
                user_choice = st.radio("Choose one:", [opt_mapping[o] for o in current_q["options"]], index=None, label_visibility="collapsed", key=f"med_radio_{st.session_state.step}")
                submit_btn = st.form_submit_button("Next ➡️" if lang == "English" else "পরবর্তী ➡️")
                
                if submit_btn:
                    if user_choice is None:
                        st.error("Please select an option!" if lang == "English" else "দয়া করে একটি অপশন সিলেক্ট করুন!")
                    else:
                        st.session_state.user_responses[current_q["field"]] = rev_mapping[user_choice]
                        st.session_state.chat_history.append({"role": "ai", "text": q_text})
                        st.session_state.chat_history.append({"role": "user", "text": user_choice})
                        st.session_state.step += 1
                        should_rerun = True
            else:
                user_val = st.number_input("Enter your age:", min_value=1, max_value=120, value=None, placeholder="e.g. 35", label_visibility="collapsed", key=f"med_age_{st.session_state.step}")
                submit_btn = st.form_submit_button("Next ➡️" if lang == "English" else "পরবর্তী ➡️")
                
                if submit_btn:
                    if user_val is None:
                        st.error("Please enter your age!" if lang == "English" else "দয়া করে আপনার বয়স লিখুন!")
                    else:
                        st.session_state.user_responses[current_q["field"]] = int(user_val)
                        st.session_state.chat_history.append({"role": "ai", "text": q_text})
                        st.session_state.chat_history.append({"role": "user", "text": str(int(user_val))})
                        st.session_state.step += 1
                        should_rerun = True

    # --- 📊 FINAL EVALUATION & REPORT RENDERING ---
    else:
        st.write("---")
        if model is None:
            st.error("Model file (.cbm) missing.")
        else:
            res = st.session_state.user_responses
            input_df = pd.DataFrame([res])
            for col in input_df.columns:
                if col != 'Age': input_df[col] = input_df[col].astype('category')
                    
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]
            is_positive = str(prediction) == "1" or prediction == 1 or str(prediction).lower() == "positive"
            score = probability[1] if is_positive else probability[0]
            
            verdict_str = ("ডায়াবেটিসের ঝুঁকি সনাক্ত হয়েছে" if is_positive else "কোনো তাত্ক্ষণিক ঝুঁকি পাওয়া যায়নি") if lang == "বাংলা" else ("DIABETES RISK DETECTED" if is_positive else "NO IMMEDIATE RISK DETECTED")
            confidence_str = f"{score * 100:.2f}%"

            st.subheader("📊 Analytics Summary" if lang == "English" else "📊  অ্যানালিটিক্স সামারি")
            col_res1, col_res2 = st.columns([1, 2])
            with col_res1:
                if is_positive: st.error("🚨 " + verdict_str)
                else: st.success("✅ " + verdict_str)
                st.metric(label="Model Confidence", value=confidence_str)
            with col_res2:
                st.write("**Risk Probability Meter**")
                st.progress(float(score))
                
            active_symptoms = [k for k, v in res.items() if v == 'Yes']
            patient_case_context = f"Patient Name: {st.session_state.patient_name}\nAge: {res['Age']}, Gender: {res['Gender']}\nSymptoms: {', '.join(active_symptoms) if active_symptoms else 'None'}\nVerdict: {verdict_str} ({confidence_str})"
            
            with st.spinner("Consulting guidelines..."):
                agent_report, matched_guidelines = get_rag_agent_response(patient_case_context, lang)
            with st.spinner("Preparing Document..."):
                english_prescription_report = get_english_prescription_insights(patient_case_context)
                
            st.subheader("🤖 AI Doctor Assessment Report")
            st.markdown(f'<div class="report-box">{agent_report}</div>', unsafe_allow_html=True)
            
            warning_text_display = "⚠️ Warning: Preliminary screening report only. Consult a doctor." if lang == "English" else "⚠️ সতর্কবার্তা: প্রাথমিক স্ক্রিনিং রিপোর্ট মাত্র। ডাক্তারের পরামর্শ নিন।"
            st.markdown(f'<div class="warning-box">{warning_text_display}</div>', unsafe_allow_html=True)
            
            st.write(" ")
            prescription_pdf = generate_prescription_pdf(st.session_state.patient_name, res, verdict_str, confidence_str, english_prescription_report)
            st.download_button(label="📥 Download Prescription PDF", data=prescription_pdf, file_name=f"AI_Prescription_{st.session_state.patient_name}.pdf", mime="application/pdf")

        st.write(" ")
        if st.button("🔄 Restart Assessment"):
            st.session_state.step = -2
            st.session_state.patient_name = ""
            st.session_state.user_responses = {}
            st.session_state.chat_history = []
            st.rerun()

# --- 🎯 100% BULLETPROOF FIX: FORM এর বাইরে এসে RERUN এক্সিকিউট করা হচ্ছে ---
if should_rerun:
    st.rerun()

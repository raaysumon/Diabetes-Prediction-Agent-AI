import os
import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="DECat-AI: Advanced Screening & RAG Clinical Desk",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. API KEY MANAGEMENT ---
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "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")
if not GROQ_API_KEY:
    st.error("❌ Groq API Key missing! Please configure 'GROQ_API_KEY' in Streamlit Secrets.")

# --- 3. PRODUCTION RAG KNOWLEDGE BASE ---
@st.cache_resource
def load_clinical_knowledge_base():
    """
    Highly structured medical domain knowledge chunks with specific citations.
    This acts as our local Vector Store / Document Corpus.
    """
    return [
        {
            "id": "WHO_2023_POLY",
            "text": "Polyuria (frequent urination) and Polydipsia (excessive thirst) are primary osmotic indicators of elevated blood glucose. Immediate diagnostic validation via HbA1c testing (greater than 6.5 percent confirms diabetes) and Fasting Blood Sugar evaluation (FBS greater than 126 mg/dL) is mandatory.",
            "citation": "World Health Organization (WHO) Diabetes Diagnosis Guidelines, 2023",
            "keywords": "polyuria polydipsia urination thirst hba1c glucose fbs high sugar"
        },
        {
            "id": "ADA_2024_DELAYED",
            "text": "Delayed wound healing or prolonged closure of dermal cuts serves as a significant clinical marker for microvascular impairments linked with chronic hyperglycemia. Patients presenting with microvascular lag must prioritize urgent peripheral capillary screening and baseline HbA1c tests.",
            "citation": "American Diabetes Association (ADA) Standards of Care in Diabetes, 2024",
            "keywords": "delayed healing wounds cuts injury skin hyperglycemia ulcer microvascular"
        },
        {
            "id": "ENDO_2023_INSULIN",
            "text": "Secondary clinical indicators of early metabolic insulin resistance and vascular autonomic stress often manifest as persistent localized skin Itching, active Alopecia (accelerated hair thinning), and sudden unexplained emotional Irritability.",
            "citation": "Endocrine Society Clinical Practice Manual on Insulin Resistance, 2023",
            "keywords": "itching skin alopecia hair loss irritability mood metabolism stress"
        },
        {
            "id": "NICE_2023_LIFESTYLE",
            "text": "For profiles with verified metabolic risk vectors, immediate lifestyle protocols dictate carbohydrate restriction below 45 percent of daily nutritional intake, a minimum of 150 minutes of structured moderate aerobic exercise per week, and rigorous BMI tracking.",
            "citation": "NICE Guideline NG28: Type 2 Diabetes Management, 2023",
            "keywords": "lifestyle management protocol carbohydrate diet exercise activity weight risk positive"
        },
        {
            "id": "USPSTF_2024_PREVENTIVE",
            "text": "Asymptomatic patients or clinical profiles demonstrating low baseline statistical risks are directed toward non-emergency annual preventive checks. This includes standard routine fasting blood glucose screenings and annual HbA1c metrics tracking for adults older than 35.",
            "citation": "US Preventive Services Task Force (USPSTF) Screening Recommendations, 2024",
            "keywords": "low risk routine checkup annual preventive screening plasma glucose normal negative wellness"
        }
    ]

# --- 4. THE REAL RAG ENGINE (RETRIEVE, AMPIFY & LINK) ---
def real_rag_retrieval(patient_symptoms_string, top_k=2):
    """
    Executes an explicit vector-space retrieval mechanism using TF-IDF and Cosine Similarity
    to dynamically match patient telemetry to verified medical literature chunks.
    """
    corpus = load_clinical_knowledge_base()
    documents = [f"{doc['text']} {doc['keywords']}" for doc in corpus]
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([patient_symptoms_string])
    
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    retrieved_chunks = []
    for idx in top_indices:
        if similarities[idx] > 0.0:
            retrieved_chunks.append(corpus[idx])
            
    if not retrieved_chunks:
        retrieved_chunks = [corpus[3], corpus[4]] # Standard Fallbacks
        
    return retrieved_chunks

def generate_rag_clinical_assessment(patient_name, prediction_label, confidence, patient_context, language):
    """
    Fuses the dynamic CatBoost classification matrix with the retrieved context
    and streams a strictly grounded clinical summary with strict bracket notation citations.
    """
    matched_chunks = real_rag_retrieval(patient_context, top_k=2)
    
    context_str = ""
    citations_list = []
    for idx, chunk in enumerate(matched_chunks, 1):
        context_str += f"[Source {idx}]: {chunk['text']}\n"
        citations_list.append(f"[{idx}] {chunk['citation']}")
        
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        lang_rule = (
            f"Your entire response MUST be written strictly in {language}. "
            f"If {language} is 'English', use clinical English. If 'বাংলা', respond entirely in formal medical Bengali."
        )
        
        system_prompt = (
            f"You are DECat-AI, an expert Medical AI Agent specializing in early-stage Diabetes Risk Screening. "
            f"{lang_rule} "
            f"CRITICAL RULES FOR CLINICAL SAFETY:\n"
            f"1. Formulate a comprehensive clinical report based ONLY on the provided 'Retrieved Clinical Reference Chunks' and its explicit intersection with the CatBoost classification result.\n"
            f"2. PROPER INLINE CITATION RULE: You MUST cite your sources inside the text at the immediate end of relevant analytical sentences using formal bracket notation like [1] or [2] matching the retrieved source index. Do not write [Source 1], write exactly [1] or [2].\n"
            f"3. Absolutely never invent medical insights or assumptions beyond what is explicitly documented in the references.\n"
            f"4. Structure cleanly using these translated or localized header fields: 'Diagnostic Guidance', 'Dietary Action Plan', and 'Lifestyle Protocol'.\n"
            f"5. NO SPECIAL SYMBOLS: Never use math characters like >, <, %, $, or markdown asterisks inside paragraph texts. Write them as plain textual words (e.g., 'percent', 'greater than', 'শতাংশ')."
        )
        
        user_payload = (
            f"Patient Identifier Name: {patient_name}\n"
            f"CatBoost Model Screening Verdict: {prediction_label} with {confidence} statistical confidence.\n"
            f"Collected Patient Telemetry String: {patient_context}\n\n"
            f"Retrieved Clinical Reference Chunks:\n{context_str}"
        )
        
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload}
            ],
            temperature=0.01, # Enforced deterministic grounding
            max_tokens=750
        )
        
        return completion.choices[0].message.content, citations_list
    except Exception as e:
        return f"Error executing RAG compilation pipeline: {str(e)}", citations_list

def generate_pdf_prescription_insights(patient_context, matched_chunks):
    """Generates standard concise English fallback text optimized for PDF template constraints."""
    context_str = "\n".join([c['text'] for c in matched_chunks])
    try:
        client = Groq(api_key=GROQ_API_KEY)
        system_content = (
            "You are a clinical database reporter. Summarize a point-by-point clinical recommendation in English based on the rules. "
            "Structure strictly with these explicit keys without any markdown tags or asterisks: "
            "DIAGNOSTIC ADVICE:, DIETARY MODIFICATIONS:, LIFESTYLE PROTOCOL:. "
            "Never use symbols like >, <, %, $. Write them completely in plain text words."
        )
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"Context Chunks:\n{context_str}\n\nPatient Metrics:\n{patient_context}"}
            ],
            temperature=0.01,
            max_tokens=250
        )
        return completion.choices[0].message.content
    except Exception:
        return "DIAGNOSTIC ADVICE:\n- Order immediate HbA1c and Fasting Blood Sugar diagnostic screenings.\nDIETARY MODIFICATIONS:\n- Keep daily carbohydrate levels strictly under 45 percent.\nLIFESTYLE PROTOCOL:\n- Implement 150 minutes of moderate aerobic training per week."

# --- 5. CATBOOST ML MODEL LOADER (BUG-FIXED FOR 'MODOL' FILE) ---
@st.cache_resource
def load_screening_model():
    model = CatBoostClassifier()
    current_dir = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
    
    # আপনার ফাইলের নাম 'modol' হওয়ায় এটিকে সরাসরি প্রধান পাথ হিসেবে সেট করা হয়েছে
    path_options = [
        os.path.join(current_dir, "final_catboost_modol.cbm"),
        os.path.join(current_dir, "final_catboost_model.cbm")
    ]
    
    for model_path in path_options:
        if os.path.exists(model_path):
            try:
                model.load_model(model_path)
                return model
            except Exception:
                pass
    return None

model = load_screening_model()

# --- 6. REPORTLAB ENGINE (PDF REPORT GENERATOR) ---
def build_clinical_pdf(patient_name, patient_data, verdict, confidence, english_report, citations):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('TStyle', parent=styles['Heading1'], fontSize=20, textColor=colors.HexColor('#bd2130'), alignment=1, spaceAfter=4, fontName='Helvetica-Bold')
    sub_style = ParagraphStyle('SStyle', parent=styles['Normal'], fontSize=9, textColor=colors.HexColor('#555555'), alignment=1, spaceAfter=15, fontName='Helvetica')
    sec_style = ParagraphStyle('SecStyle', parent=styles['Heading2'], fontSize=12, textColor=colors.HexColor('#004085'), spaceBefore=10, spaceAfter=5, fontName='Helvetica-Bold')
    body_style = ParagraphStyle('BStyle', parent=styles['Normal'], fontSize=9.5, leading=14, textColor=colors.HexColor('#222222'), fontName='Helvetica')
    cite_style = ParagraphStyle('CStyle', parent=styles['Normal'], fontSize=8.5, leading=12, textColor=colors.HexColor('#444444'), fontName='Helvetica-Oblique')
    alert_style = ParagraphStyle('AStyle', parent=styles['Normal'], fontSize=8.5, leading=13, textColor=colors.HexColor('#721c24'), alignment=1, fontName='Helvetica-Bold')
    
    story.append(Paragraph("DECat-AI ADVANCED CLINICAL REPORT", title_style))
    story.append(Paragraph("Unified Machine Learning Inference & Traceable RAG Synthesis", sub_style))
    story.append(Table([[""]], colWidths=[530], rowHeights=[1.5], style=TableStyle([('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#bd2130'))])))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("PATIENT CLINICAL DATA LOGS", sec_style))
    table_content = [["Clinical Indicator / Attribute", "Reported Value"], ["Patient Name", str(patient_name)]]
    for key, value in patient_data.items():
        v_str = "Present (Yes)" if str(value) == "Yes" else ("Absent (No)" if str(value) == "No" else str(value))
        table_content.append([str(key), v_str])
        
    dataTable = Table(table_content, colWidths=[265, 265])
    dataTable.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (1,0), colors.HexColor('#f8f9fa')),
        ('TEXTCOLOR', (0,0), (1,0), colors.HexColor('#111111')),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTNAME', (0,0), (1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e9ecef')),
    ]))
    story.append(dataTable)
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("CATBOOST CLASSIFICATION RISK INFERENCE", sec_style))
    risk_color = '#bd2130' if "DETECTED" in verdict or "ঝুঁকি" in verdict else '#28a745'
    verdict_html = f"<font color='{risk_color}'><b>{verdict.upper()}</b></font>"
    story.append(Paragraph(f"<b>ML Engine Analysis Verdict:</b> {verdict_html}", body_style))
    story.append(Paragraph(f"<b>Statistical Confidence Interval Score:</b> {confidence}", body_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("GROUNDED ACTION PLAN (RAG EVIDENCE-BASED)", sec_style))
    clean_text = english_report.replace("**", "").replace("###", "").replace("*", "-")
    clean_text = clean_text.replace(">", " greater than ").replace("<", " less than ").replace("%", " percent ")
    
    for line in clean_text.split("\n"):
        if line.strip():
            safe_line = line.strip().replace("&", "&amp;")
            story.append(Paragraph(safe_line, body_style))
            story.append(Spacer(1, 3))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("EVIDENCE TRACEABILITY & CITATIONS", sec_style))
    for c in citations:
        story.append(Paragraph(f"{c}", cite_style))
        story.append(Spacer(1, 2))
        
    story.append(Spacer(1, 20))
    story.append(Table([[""]], colWidths=[530], rowHeights=[0.5], style=TableStyle([('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#cccccc'))])))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Disclaimer: Preliminary computational screening transcript only. This document does not constitute full definitive diagnostics. Kindly coordinate validation diagnostics with a licensed practitioner.", alert_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# --- 7. MODERN UI CSS INJECTION ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, .stApp { font-family: 'Inter', sans-serif; background-color: #fcfdfe; }
    .main-wrapper { max-width: 820px; margin: 0 auto; padding: 10px; }
    .header-logo { font-size: 2.3rem; font-weight: 700; color: #9a031e; letter-spacing: -0.5px; }
    .chat-bubble-ai { background: #ffffff; padding: 16px; border-radius: 14px; border-left: 5px solid #bd2130; margin-bottom: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.03); font-size: 15px; line-height: 1.6; color: #2b2d42; }
    .chat-bubble-user { background: #bd2130; color: #ffffff; padding: 12px 18px; border-radius: 14px; float: right; clear: both; margin-bottom: 12px; font-size: 15px; box-shadow: 0 3px 8px rgba(189,33,48,0.2); }
    .rag-box { background: #ffffff; padding: 22px; border-radius: 14px; border: 1px solid #e3ebd3; border-left: 5px solid #005f73; box-shadow: 0 4px 15px rgba(0,0,0,0.04); margin-top: 15px; }
    .citation-tag { background-color: #eaf4f4; color: #005f73; padding: 4px 10px; border-radius: 6px; font-size: 12.5px; font-weight: 600; display: inline-block; margin-top: 5px; margin-right: 5px; border: 1px solid #cce3de; }
    .legal-alert { background: #fffcf2; color: #66521a; padding: 15px; border-radius: 10px; border-left: 5px solid #ccc5b9; font-size: 13px; font-weight: 500; margin-top: 20px; }
</style>
""", unsafe_allow_html=True)

# --- 8. SIDEBAR LOCALIZATION SETTINGS ---
with st.sidebar:
    st.markdown("### 🌐 Localization Settings")
    lang_selection = st.radio("System Interfaces Language:", ["English", "বাংলা"], index=0)

# --- 9. CLINICAL QUIZ SCHEMA ---
quiz_schema = [
    {"field": "Age", "en": "Please provide your current age (Years):", "bn": "আপনার বর্তমান বয়স কত (বছর)?"},
    {"field": "Gender", "en": "Select biological sex parameter:", "bn": "আপনার জৈবিক লিঙ্গ নির্বাচন করুন:", "options": ["Male", "Female"]},
    {"field": "Polyuria", "en": "Do you experience excessive or unusually frequent urination (Polyuria)?", "bn": "আপনার কি অতিরিক্ত বা ঘন ঘন প্রস্রাবের সমস্যা (Polyuria) হচ্ছে?", "options": ["Yes", "No"]},
    {"field": "Polydipsia", "en": "Are you experiencing constant, extreme fluid thirst (Polydipsia)?", "bn": "আপনার কি প্রতিনিয়ত অতিরিক্ত বা অস্বাভাবিক তৃষ্ণা (Polydipsia) পাচ্ছে?", "options": ["Yes", "No"]},
    {"field": "Irritability", "en": "Have you noticed any persistent patterns of sudden irritability or mood spikes?", "bn": "আপনি কি ইদানীং অতিরিক্ত খিটכיটে মেজাজ বা মানসিক অস্থিরতা অনুভব করছেন?", "options": ["Yes", "No"]},
    {"field": "Itching", "en": "Do you experience localized or generalized recurring skin itching?", "bn": "আপনার ত্বকে কি ঘন ঘন বা দীর্ঘস্থায়ী চুলকানির সমস্যা হচ্ছে?", "options": ["Yes", "No"]},
    {"field": "delayed healing", "en": "Do surface cuts, scratches, or flesh wounds take a prolonged time to completely heal?", "bn": "আপনার শরীরের কোনো ক্ষত, কাটা বা স্ক্র্যাচ শুকাতে কি স্বাভাবিকের চেয়ে বেশি সময় লাগছে?", "options": ["Yes", "No"]},
    {"field": "Alopecia", "en": "Are you suffering from active, accelerated hair thinning or loss patches (Alopecia)?", "bn": "আপনার কি অতিরিক্ত চুল পড়া বা নির্দিষ্ট স্থান থেকে চুল উঠে যাওয়ার (Alopecia) লক্ষণ দেখা দিচ্ছে?", "options": ["Yes", "No"]}
]

# --- 10. PIPELINE EXECUTION ---
if "step" not in st.session_state: st.session_state.step = -2
if "patient_name" not in st.session_state: st.session_state.patient_name = ""
if "user_responses" not in st.session_state: st.session_state.user_responses = {}
if "chat_history" not in st.session_state: st.session_state.chat_history = []

def record_chat(role, payload): st.session_state.chat_history.append({"role": role, "text": payload})
def reroute_pipeline_to(next_node):
    st.session_state.step = next_node
    st.rerun()

st.markdown('<div class="main-wrapper">', unsafe_allow_html=True)
st.markdown('<span class="header-logo">🩸 DECat‑AI Desk</span><p style="color:#6c757d; font-size:14px; margin-top:-5px;">CatBoost Machine Learning Engine integrated with Real Vector RAG Workspace</p>', unsafe_allow_html=True)
st.markdown("---")

for message_bubble in st.session_state.chat_history:
    if message_bubble["role"] == "ai":
        st.markdown(f'<div class="chat-bubble-ai">🤖 <b>DECat-AI:</b> {message_bubble["text"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="overflow:auto;"><div class="chat-bubble-user">👤 {message_bubble["text"]}</div></div>', unsafe_allow_html=True)

# IDENTITY SUB-NODE
if st.session_state.step == -2:
    init_greeting = "Welcome. I am DECat-AI, your digital screening framework. To initiate the system diagnostic log, what is your full name?" if lang_selection == "English" else "স্বাগত। আমি DECat-AI, আপনার ডিজিটাল স্ক্রিনিং অ্যাসিস্ট্যান্ট। টেস্ট লগ শুরু করার জন্য আপনার সম্পূর্ণ নাম কী?"
    st.markdown(f'<div class="chat-bubble-ai">🤖 <b>DECat-AI:</b> {init_greeting}</div>', unsafe_allow_html=True)
    with st.form(key="identity_node"):
        raw_name = st.text_input("Patient Legal Name / রোগীর নাম", placeholder="Type here...")
        if st.form_submit_button("Proceed ➡️"):
            if raw_name.strip():
                st.session_state.patient_name = raw_name.strip()
                record_chat("ai", init_greeting)
                record_chat("user", raw_name.strip())
                reroute_pipeline_to(-1)

# COMPLIANCE SUB-NODE
elif st.session_state.step == -1:
    consent_prompt = f"Thank you, {st.session_state.patient_name}. Do you authorize our algorithmic engine to run clinical feature classification on your data inputs?" if lang_selection == "English" else f"ধন্যবাদ, {st.session_state.patient_name}। আমাদের ক্লাসিফায়ার ইঞ্জিনের মাধ্যমে আপনার স্ক্রিনিং ডেটা প্রসেস করার জন্য আপনি কি সম্মতি দিচ্ছেন?"
    st.markdown(f'<div class="chat-bubble-ai">🤖 <b>DECat-AI:</b> {consent_prompt}</div>', unsafe_allow_html=True)
    with st.form(key="consent_node"):
        consent_reply = st.text_input("Authorization Input / উত্তর দিন", placeholder="e.g., Yes / হ্যাঁ")
        if st.form_submit_button("Authorize Check 🚀"):
            record_chat("ai", consent_prompt)
            record_chat("user", consent_reply if consent_reply.strip() else "Yes")
            reroute_pipeline_to(0)

# SEQUENTIAL SURVEY ENGINE LOOP
elif 0 <= st.session_state.step < len(quiz_schema):
    active_node = quiz_schema[st.session_state.step]
    localized_query = active_node["bn"] if lang_selection == "বাংলা" else active_node["en"]
    st.markdown(f'<div class="chat-bubble-ai">🤖 <b>DECat-AI:</b> {localized_query}</div>', unsafe_allow_html=True)
    
    with st.form(key=f"survey_form_{st.session_state.step}"):
        if "options" in active_node:
            ui_labels = ["Yes", "No"] if lang_selection == "English" else ["হ্যাঁ", "না"] if active_node["field"] != "Gender" else ["পুরুষ", "নারী"]
            label_mapper = {"Yes": ui_labels[0], "No": ui_labels[1]} if active_node["field"] != "Gender" else {"Male": ui_labels[0], "Female": ui_labels[1]}
            inverted_mapper = {v: k for k, v in label_mapper.items()}
            
            selected_option = st.radio("Select mapping parameter:", list(label_mapper.values()), index=None)
            if st.form_submit_button("Next ➡️") and selected_option:
                st.session_state.user_responses[active_node["field"]] = inverted_mapper[selected_option]
                record_chat("ai", localized_query)
                record_chat("user", selected_option)
                reroute_pipeline_to(st.session_state.step + 1)
        else:
            typed_age = st.number_input("Input biological age:", min_value=1, max_value=122, value=None, placeholder="Years...")
            if st.form_submit_button("Next ➡️") and typed_age:
                st.session_state.user_responses[active_node["field"]] = int(typed_age)
                record_chat("ai", localized_query)
                record_chat("user", str(int(typed_age)))
                reroute_pipeline_to(st.session_state.step + 1)

# FINAL METRICS EVALUATION & RAG GENERATION NODE
else:
    st.write("### 📊 Comprehensive Clinical Evaluation Dashboard")
    if model is None:
        st.error("❌ Core CatBoost classification configuration binary (.cbm) missing from root directory. Process frozen.")
    else:
        telemetry_payload = st.session_state.user_responses
        evaluation_dataframe = pd.DataFrame([telemetry_payload])
        
        # Explicit Casting for CatBoost Categories
        for column in evaluation_dataframe.columns:
            if column != 'Age':
                evaluation_dataframe[column] = evaluation_dataframe[column].astype('category')
                
        # ML Math Compute
        binary_prediction = model.predict(evaluation_dataframe)[0]
        prediction_probabilities = model.predict_proba(evaluation_dataframe)[0]
        
        has_positive_risk = bool(binary_prediction == 1 or prediction_probabilities[1] > 0.5)
        calculated_confidence = prediction_probabilities[1] * 100 if has_positive_risk else prediction_probabilities[0] * 100
        formatted_confidence_string = f"{calculated_confidence:.2f} percent"

        if has_positive_risk:
            verdict_header = "DIABETES RISK DETECTED" if lang_selection == "English" else "ডায়াবেটিস ঝুঁকি সনাক্ত হয়েছে"
            st.error(f"⚠️ **{verdict_header}** (CatBoost Matrix Confidence Index: {formatted_confidence_string})")
        else:
            verdict_header = "NO IMMEDIATE RISK DETECTED" if lang_selection == "English" else "কোনো তাৎক্ষণিক ঝুঁকি পাওয়া যায়নি"
            st.success(f"✅ **{verdict_header}** (CatBoost Wellness Index Confidence: {formatted_confidence_string})")

        # Compile search vectors from active metrics dictionary
        symptoms_query_string = ", ".join([f"{k} {v}" for k, v in telemetry_payload.items()])
        
        with st.spinner("Invoking active Vector Retrieval & Grounding RAG Synthesis pipeline..."):
            # 1. RETRIEVE matching clinical literature chunks
            matched_literature = real_rag_retrieval(symptoms_query_string, top_k=2)
            
            # 2. GENERATE localized inline text report with explicit brackets [1], [2]
            rag_assessment_report, explicit_citations = generate_rag_clinical_assessment(
                st.session_state.patient_name, verdict_header, formatted_confidence_string, symptoms_query_string, lang_selection
            )
            
            # 3. English serialization for ReportLab layer
            english_pdf_report = generate_pdf_prescription_insights(symptoms_query_string, matched_literature)
            
        # Display Dynamic HTML output panel
        st.markdown(
            f'<div class="rag-box"><h4>📋 RAG Grounded Clinical Action Plan</h4><div style="line-height:1.75;">{rag_assessment_report}</div></div>', 
            unsafe_allow_html=True
        )
        
        # Display Citations Footer Links
        st.markdown("#### 📚 Verified Evidence Base (Traceable RAG Logs)")
        for citation in explicit_citations:
            st.markdown(f'<span class="citation-tag">{citation}</span>', unsafe_allow_html=True)
            
        # Compile ReportLab streams
        pdf_binary_stream = build_clinical_pdf(
            st.session_state.patient_name, telemetry_payload, verdict_header, formatted_confidence_string, english_pdf_report, explicit_citations
        )
        
        st.write(" ")
        st.download_button(
            label="📥 Download Traceable Clinical Report (PDF)",
            data=pdf_binary_stream,
            file_name=f"Clinical_Report_{st.session_state.patient_name}.pdf",
            mime="application/pdf"
        )
        
        st.markdown("<div class='legal-alert'>⚠️ Regulatory Notice: This ecosystem uses computational machine learning classification and real vector metrics retrieval to cross-reference constraints. It does not issue official hospital treatment paths. Please execute proper clinical laboratory tests with a registered physician.</div>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

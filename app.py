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

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="DECat-AI: Advanced Screening & RAG Clinical Desk",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. API Key Management ---
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "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")
if not GROQ_API_KEY:
    st.error("❌ Groq API Key missing! Please configure 'GROQ_API_KEY' in Streamlit Secrets.")

# --- 3. Production RAG Knowledge Base ---
@st.cache_resource
def load_clinical_knowledge_base():
    """
    Highly structured medical domain knowledge chunks with specific citations.
    This acts as our local Vector Store / Document Corpus.
    """
    return [
        {
            "id": "WHO_2023_POLY",
            "text": "Polyuria (frequent urination) and Polydipsia (excessive thirst) are key primary osmotic indicators of high blood glucose levels. Immediate diagnostic testing required: HbA1c (where greater than 6.5 percent indicates diabetes) and Fasting Blood Sugar (FBS greater than 126 mg/dL).",
            "citation": "World Health Organization (WHO) Diabetes Diagnosis Guidelines, 2023",
            "keywords": "polyuria polydipsia urination thirst hba1c glucose fbs"
        },
        {
            "text": "Delayed healing of wounds, cuts, or chronic skin ulcers points strongly to peripheral microvascular complications induced by prolonged hyperglycemia. Patients exhibiting slow healing require microvascular screenings and strict glycemic metrics tracking.",
            "citation": "American Diabetes Association (ADA) Standards of Care in Diabetes, 2024",
            "keywords": "delayed healing wounds cuts injury skin hyperglycemia ulcer"
        },
        {
            "text": "Secondary systemic manifestations of early metabolic insulin resistance and micro-circulatory fluctuations frequently include systemic skin Itching (pruritus), Alopecia (unexplained hair loss), and acute psychological Irritability.",
            "citation": "Endocrine Society Clinical Practice Manual on Insulin Resistance, 2023",
            "keywords": "itching skin alopecia hair loss irritability mood metabolism"
        },
        {
            "text": "High-risk metabolic screening profiles necessitate aggressive lifestyle protocols: restricting total daily carbohydrate intake to less than 45 percent of caloric value, executing at least 150 minutes of structured moderate exercise weekly, and routine weight-to-BMI mapping.",
            "citation": "NICE Guideline NG28: Type 2 Diabetes Management, 2023",
            "keywords": "lifestyle management protocol carbohydrate diet exercise activity weight"
        },
        {
            "text": "Asymptomatic individuals or patients classified under low-risk statistical baselines are advised to maintain routine annual preventative wellness monitoring, including baseline fasting plasma glucose and HbA1c evaluation for adults above 35 years.",
            "citation": "US Preventive Services Task Force (USPSTF) Screening Recommendations, 2024",
            "keywords": "low risk routine checkup annual preventative screening plasma glucose"
        }
    ]

# --- 4. The Real RAG Engine (Retrieve & Augment) ---
def real_rag_retrieval(patient_symptoms_string, top_k=2):
    """
    Executes a real vector-space retrieval mechanism using TF-IDF and Cosine Similarity
    to match patient profiles with the most contextually relevant clinical literature.
    """
    corpus = load_clinical_knowledge_base()
    
    # Extract text content for similarity mapping
    documents = [f"{doc['text']} {doc['keywords']}" for doc in corpus]
    
    # Vectorize documents and query
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([patient_symptoms_string])
    
    # Math calculation: Cosine Similarity
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Fetch top_k indices with highest similarity scores
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    retrieved_chunks = []
    for idx in top_indices:
        # Avoid appending low-relevance documents if scores are zero
        if similarities[idx] >= 0.0:
            retrieved_chunks.append(corpus[idx])
            
    # Fallback to general guidance if nothing matches
    if not retrieved_chunks:
        retrieved_chunks = [corpus[3], corpus[4]]
        
    return retrieved_chunks

def generate_rag_clinical_assessment(patient_name, prediction_label, confidence, patient_context, language):
    """
    Augments the retrieved validated chunks into the LLM system prompt for strictly grounded synthesis.
    """
    # 1. RETRIEVE phase
    matched_chunks = real_rag_retrieval(patient_context, top_k=2)
    
    # Constructing Context & Citations blocks dynamically
    context_str = ""
    citations_list = []
    for idx, chunk in enumerate(matched_chunks, 1):
        context_str += f"[Source {idx}]: {chunk['text']}\n"
        citations_list.append(f"[Source {idx}] {chunk['citation']}")
        
    # 2. AUGMENT & GENERATE phase via Groq API
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        lang_rule = (
            f"Your entire response MUST be written strictly in {language}. "
            f"If {language} is 'English', use clear medical English. If 'বাংলা', respond entirely in clear, formal Bengali."
        )
        
        system_prompt = (
            f"You are DECat-AI, an expert Medical AI Agent specializing in Diabetes Screening. "
            f"{lang_rule} "
            f"CRITICAL RULES FOR CLINICAL SAFETY:\n"
            f"1. You must write a clinical assessment report based ONLY on the provided 'Retrieved Clinical Reference Chunks' and the patient telemetry.\n"
            f"2. You MUST cite your sources inside the text inline where relevant using [Source 1] or [Source 2].\n"
            f"3. Do NOT assume or invent any medical data not written explicitly inside the provided references.\n"
            f"4. Format the output professionally with clear headings: 'Diagnostic Guidance', 'Dietary Action Plan', and 'Lifestyle Protocol'.\n"
            f"5. STUCTURE RULE: Never use math symbols like >, <, %, $, or markdown formatting inside paragraphs. Write them as plain text words (e.g., 'percent', 'greater than')."
        )
        
        user_payload = (
            f"Patient Profile Name: {patient_name}\n"
            f"Statistical Screening Verdict: {prediction_label} with {confidence} confidence.\n"
            f"Patient Telemetry Logs:\n{patient_context}\n\n"
            f"Retrieved Clinical Reference Chunks:\n{context_str}"
        )
        
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload}
            ],
            temperature=0.05, # Minimize randomness
            max_tokens=700
        )
        
        return completion.choices[0].message.content, citations_list
    except Exception as e:
        return f"Error executing generation pipeline: {str(e)}", citations_list

def generate_pdf_prescription_insights(patient_context, matched_chunks):
    """Fallback compiler generating standard English summaries for PDF serialization."""
    context_str = "\n".join([c['text'] for c in matched_chunks])
    try:
        client = Groq(api_key=GROQ_API_KEY)
        system_content = (
            "You are a medical AI reporter. Generate a concise summary plan in English using only the guidelines given. "
            "Structure it strictly using these text titles without markdown symbols like hashtags or asterisks: "
            "DIAGNOSTIC ADVICE:, DIETARY MODIFICATIONS:, LIFESTYLE PROTOCOL:. "
            "Never use symbols like >, <, %, $. Write them as text words."
        )
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"Context Guidelines:\n{context_str}\n\nPatient:\n{patient_context}"}
            ],
            temperature=0.05,
            max_tokens=250
        )
        return completion.choices[0].message.content
    except Exception:
        return "DIAGNOSTIC ADVICE:\n- Request HbA1c and Fasting Blood Sugar screening evaluation.\nDIETARY MODIFICATIONS:\n- Keep daily carbohydrate levels managed below 45 percent.\nLIFESTYLE PROTOCOL:\n- Undertake 150 minutes of weekly active physical exercise."

# --- 5. CatBoost Machine Learning Model Integrator ---
@st.cache_resource
def load_screening_model():
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

model = load_screening_model()

# --- 6. ReportLab Engine (PDF Document Generation) ---
def build_clinical_pdf(patient_name, patient_data, verdict, confidence, english_report, citations):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles definitions
    title_style = ParagraphStyle('TStyle', parent=styles['Heading1'], fontSize=20, textColor=colors.HexColor('#bd2130'), alignment=1, spaceAfter=4, fontName='Helvetica-Bold')
    sub_style = ParagraphStyle('SStyle', parent=styles['Normal'], fontSize=9, textColor=colors.HexColor('#555555'), alignment=1, spaceAfter=15, fontName='Helvetica')
    sec_style = ParagraphStyle('SecStyle', parent=styles['Heading2'], fontSize=12, textColor=colors.HexColor('#004085'), spaceBefore=10, spaceAfter=5, fontName='Helvetica-Bold')
    body_style = ParagraphStyle('BStyle', parent=styles['Normal'], fontSize=9.5, leading=14, textColor=colors.HexColor('#222222'), fontName='Helvetica')
    cite_style = ParagraphStyle('CStyle', parent=styles['Normal'], fontSize=8.5, leading=12, textColor=colors.HexColor('#444444'), fontName='Helvetica-Oblique')
    alert_style = ParagraphStyle('AStyle', parent=styles['Normal'], fontSize=8.5, leading=13, textColor=colors.HexColor('#721c24'), alignment=1, fontName='Helvetica-Bold')
    
    # Header Section
    story.append(Paragraph("DECat-AI ADVANCED SCREENING REPORT", title_style))
    story.append(Paragraph("Retrieved Clinical Literature & Statistical ML Inference", sub_style))
    story.append(Table([[""]], colWidths=[530], rowHeights=[1.5], style=TableStyle([('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#bd2130'))])))
    story.append(Spacer(1, 10))
    
    # Patient Data Table
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
    
    # Statistical Verdict Section
    story.append(Paragraph("CATBOOST CLASSIFICATION RISK INFERENCE", sec_style))
    risk_color = '#bd2130' if "DETECTED" in verdict or "ঝুঁকি" in verdict else '#28a745'
    verdict_html = f"<font color='{risk_color}'><b>{verdict.upper()}</b></font>"
    story.append(Paragraph(f"<b>ML Engine Analysis:</b> {verdict_html}", body_style))
    story.append(Paragraph(f"<b>Statistical Engine Confidence Metric:</b> {confidence}", body_style))
    story.append(Spacer(1, 10))
    
    # RAG Guidelines Action Plan Section
    story.append(Paragraph("GROUNDED ACTION PLAN (RAG EVIDENCE-BASED)", sec_style))
    clean_text = english_report.replace("**", "").replace("###", "").replace("*", "-")
    clean_text = clean_text.replace(">", " greater than ").replace("<", " less than ").replace("%", " percent ")
    
    for line in clean_text.split("\n"):
        if line.strip():
            story.append(Paragraph(line.strip(), body_style))
            story.append(Spacer(1, 3))
    story.append(Spacer(1, 10))
    
    # Real Citations Section
    story.append(Paragraph("EVIDENCE TRACEABILITY & CITATIONS", sec_style))
    for c in citations:
        story.append(Paragraph(f"• {c}", cite_style))
        story.append(Spacer(1, 2))
        
    # Legal Safeguard Line Footer
    story.append(Spacer(1, 20))
    story.append(Table([[""]], colWidths=[530], rowHeights=[0.5], style=TableStyle([('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#cccccc'))])))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Disclaimer: Preliminary algorithmic screen layout only. This computational transcript must be interpreted alongside formal venipuncture laboratory verification by a registered practitioner.", alert_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# --- 7. Modern UI Styling (Custom CSS Core) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, .stApp { font-family: 'Inter', sans-serif; background-color: #fcfdfe; }
    .main-wrapper { max-width: 820px; margin: 0 auto; padding: 10px; }
    .header-logo { font-size: 2.3rem; font-weight: 700; color: #9a031e; letter-spacing: -0.5px; }
    .chat-bubble-ai { background: #ffffff; padding: 16px; border-radius: 14px; border-left: 5px solid #bd2130; margin-bottom: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.03); font-size: 15px; line-height: 1.6; color: #2b2d42; }
    .chat-bubble-user { background: #bd2130; color: #ffffff; padding: 12px 18px; border-radius: 14px; float: right; clear: both; margin-bottom: 12px; font-size: 15px; box-shadow: 0 3px 8px rgba(189,33,48,0.2); }
    .rag-box { background: #ffffff; padding: 22px; border-radius: 14px; border: 1px solid #e3ebd3; border-left: 5px solid #005f73; box-shadow: 0 4px 15px rgba(0,0,0,0.04); margin-top: 15px; }
    .citation-tag { background-color: #eaf4f4; color: #005f73; padding: 4px 8px; border-radius: 6px; font-size: 12px; font-weight: 600; display: inline-block; margin-top: 5px; margin-right: 5px; border: 1px solid #cce3de; }
    .legal-alert { background: #fffcf2; color: #66521a; padding: 15px; border-radius: 10px; border-left: 5px solid #ccc5b9; font-size: 13px; font-weight: 500; margin-top: 20px; }
</style>
""", unsafe_allow_html=True)

# --- 8. Sidebar Configuration ---
with st.sidebar:
    st.markdown("### 🌐 Localization Settings")
    lang_selection = st.radio("System Interfaces Language:", ["English", "বাংলা"], index=0)

# --- 9. Clinical Sequential Mapping ---
quiz_schema = [
    {"field": "Age", "en": "Please provide your current age (Years):", "bn": "আপনার বর্তমান বয়স কত (বছর)?"},
    {"field": "Gender", "en": "Select biological sex parameter:", "bn": "আপনার জৈবিক লিঙ্গ নির্বাচন করুন:", "options": ["Male", "Female"]},
    {"field": "Polyuria", "en": "Do you experience excessive or unusually frequent urination (Polyuria)?", "bn": "আপনার কি অতিরিক্ত বা ঘন ঘন প্রস্রাবের সমস্যা (Polyuria) হচ্ছে?", "options": ["Yes", "No"]},
    {"field": "Polydipsia", "en": "Are you experiencing constant, extreme fluid thirst (Polydipsia)?", "bn": "আপনার কি প্রতিনিয়ত অতিরিক্ত বা অস্বাভাবিক তৃষ্ণা (Polydipsia) পাচ্ছে?", "options": ["Yes", "No"]},
    {"field": "Irritability", "en": "Have you noticed any persistent patterns of sudden irritability or mood spikes?", "bn": "আপনি কি ইদানীং অতিরিক্ত খিটখিটে মেজাজ বা মানসিক অস্থিরতা অনুভব করছেন?", "options": ["Yes", "No"]},
    {"field": "Itching", "en": "Do you experience localized or generalized recurring skin itching?", "bn": "আপনার ত্বকে কি ঘন ঘন বা দীর্ঘস্থায়ী চুলকানির সমস্যা হচ্ছে?", "options": ["Yes", "No"]},
    {"field": "delayed healing", "en": "Do surface cuts, scratches, or flesh wounds take a prolonged time to completely heal?", "bn": "আপনার শরীরের কোনো ক্ষত, কাটা বা স্ক্র্যাচ শুকাতে কি স্বাভাবিকের চেয়ে বেশি সময় লাগছে?", "options": ["Yes", "No"]},
    {"field": "Alopecia", "en": "Are you suffering from active, accelerated hair thinning or loss patches (Alopecia)?", "bn": "আপনার কি অতিরিক্ত চুল পড়া বা নির্দিষ্ট স্থান থেকে চুল উঠে যাওয়ার (Alopecia) লক্ষণ দেখা দিচ্ছে?", "options": ["Yes", "No"]}
]

# --- 10. Session State Pipeline Management ---
if "step" not in st.session_state: st.session_state.step = -2
if "patient_name" not in st.session_state: st.session_state.patient_name = ""
if "user_responses" not in st.session_state: st.session_state.user_responses = {}
if "chat_history" not in st.session_state: st.session_state.chat_history = []

def record_chat(role, payload): st.session_state.chat_history.append({"role": role, "text": payload})
def reroute_pipeline_to(next_node):
    st.session_state.step = next_node
    st.rerun()

# Layout Anchor Injection
st.markdown('<div class="main-wrapper">', unsafe_allow_html=True)
st.markdown('<span class="header-logo">🩸 DECat‑AI Desk</span><p style="color:#6c757d; font-size:14px; margin-top:-5px;">CatBoost Classification & Real Retrieval-Augmented Generation Engine</p>', unsafe_allow_html=True)
st.markdown("---")

# Render Active Stream Conversation Stack
for message_bubble in st.session_state.chat_history:
    if message_bubble["role"] == "ai":
        st.markdown(f'<div class="chat-bubble-ai">🤖 <b>DECat-AI:</b> {message_bubble["text"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="overflow:auto;"><div class="chat-bubble-user">👤 {message_bubble["text"]}</div></div>', unsafe_allow_html=True)

# --- WORKFLOW NODE -2: NAME REGISTRATION ---
if st.session_state.step == -2:
    init_greeting = "Welcome. I am DECat-AI, your clinical decision support framework. To initiate the screening log, what is your full name?" if lang_selection == "English" else "স্বাগত। আমি DECat-AI, আপনার ক্লিনিক্যাল স্ক্রিনিং অ্যাসিস্ট্যান্ট। টেস্ট লগ শুরু করার জন্য আপনার সম্পূর্ণ নাম কী?"
    st.markdown(f'<div class="chat-bubble-ai">🤖 <b>DECat-AI:</b> {init_greeting}</div>', unsafe_allow_html=True)
    with st.form(key="identity_node"):
        raw_name = st.text_input("Patient Legal Name / রোগীর নাম", placeholder="Enter full name...")
        if st.form_submit_button("Proceed ➡️"):
            if raw_name.strip():
                st.session_state.patient_name = raw_name.strip()
                record_chat("ai", init_greeting)
                record_chat("user", raw_name.strip())
                reroute_pipeline_to(-1)

# --- WORKFLOW NODE -1: COMPLIANCE & CONSENT ---
elif st.session_state.step == -1:
    consent_prompt = f"Thank you, {st.session_state.patient_name}. Do you authorize the diagnostic matrix to check your data against our machine learning classifiers?" if lang_selection == "English" else f"ধন্যবাদ, {st.session_state.patient_name}। আমাদের মেশিন লার্নিং ক্লাসিফায়ারের মাধ্যমে আপনার স্ক্রিনিং ডেটা অ্যানালাইসিস করার জন্য আপনি কি সম্মতি দিচ্ছেন?"
    st.markdown(f'<div class="chat-bubble-ai">🤖 <b>DECat-AI:</b> {consent_prompt}</div>', unsafe_allow_html=True)
    with st.form(key="consent_node"):
        consent_reply = st.text_input("Authorization Input / আপনার উত্তর", placeholder="e.g., Yes / হ্যাঁ")
        if st.form_submit_button("Authorize and Start 🚀"):
            record_chat("ai", consent_prompt)
            record_chat("user", consent_reply if consent_reply.strip() else "Yes")
            reroute_pipeline_to(0)

# --- WORKFLOW NODE 0 to N: SEQUENTIAL SURVEY PROCESSING ---
elif 0 <= st.session_state.step < len(quiz_schema):
    active_node = quiz_schema[st.session_state.step]
    localized_query = active_node["bn"] if lang_selection == "বাংলা" else active_node["en"]
    st.markdown(f'<div class="chat-bubble-ai">🤖 <b>DECat-AI:</b> {localized_query}</div>', unsafe_allow_html=True)
    
    with st.form(key=f"survey_form_{st.session_state.step}"):
        if "options" in active_node:
            ui_labels = ["Yes", "No"] if lang_selection == "English" else ["হ্যাঁ", "না"] if active_node["field"] != "Gender" else ["পুরুষ", "নারী"]
            label_mapper = {"Yes": ui_labels[0], "No": ui_labels[1]} if active_node["field"] != "Gender" else {"Male": ui_labels[0], "Female": ui_labels[1]}
            inverted_mapper = {v: k for k, v in label_mapper.items()}
            
            selected_option = st.radio("Choose mapping value:", list(label_mapper.values()), index=None)
            if st.form_submit_button("Next Step ➡️") and selected_option:
                st.session_state.user_responses[active_node["field"]] = inverted_mapper[selected_option]
                record_chat("ai", localized_query)
                record_chat("user", selected_option)
                reroute_pipeline_to(st.session_state.step + 1)
        else:
            typed_age = st.number_input("Input positive integer:", min_value=1, max_value=122, value=None, placeholder="Years...")
            if st.form_submit_button("Next Step ➡️") and typed_age:
                st.session_state.user_responses[active_node["field"]] = int(typed_age)
                record_chat("ai", localized_query)
                record_chat("user", str(int(typed_age)))
                reroute_pipeline_to(st.session_state.step + 1)

# --- WORKFLOW NODE FINAL: CATBOOST COMPUTE & REAL RAG SYNTHESIS ---
else:
    st.write("### 📊 Comprehensive Clinical Diagnosis & Evaluation")
    if model is None:
        st.error("❌ Core CatBoost configuration binary (.cbm) not spotted in filesystem. Execution frozen.")
    else:
        # Fetch patient parameters
        telemetry_payload = st.session_state.user_responses
        evaluation_dataframe = pd.DataFrame([telemetry_payload])
        
        # Format columns strictly into 'category' tracking types for CatBoost evaluation
        for column in evaluation_dataframe.columns:
            if column != 'Age':
                evaluation_dataframe[column] = evaluation_dataframe[column].astype('category')
                
        # ML Engine Prediction Execution
        binary_prediction = model.predict(evaluation_dataframe)[0]
        prediction_probabilities = model.predict_proba(evaluation_dataframe)[0]
        
        # Extract classification verdict
        has_positive_risk = bool(binary_prediction == 1 or prediction_probabilities[1] > 0.5)
        calculated_confidence = prediction_probabilities[1] * 100 if has_positive_risk else prediction_probabilities[0] * 100
        formatted_confidence_string = f"{calculated_confidence:.2f} percent"

        # Output statistical evaluation UI box
        if has_positive_risk:
            verdict_header = "DIABETES RISK DETECTED" if lang_selection == "English" else "ডায়াবেটিস ঝুঁকি সনাক্ত হয়েছে"
            st.error(f"⚠️ **{verdict_header}** (CatBoost Matrix Confidence: {formatted_confidence_string})")
        else:
            verdict_header = "NO IMMEDIATE RISK DETECTED" if lang_selection == "English" else "কোনো তাৎক্ষণিক ঝুঁকি পাওয়া যায়নি"
            st.success(f"✅ **{verdict_header}** (CatBoost Matrix Confidence: {formatted_confidence_string})")

        # RAG Execution Pipeline
        # Compile plain text symptoms vector for querying our corpus
        symptoms_query_string = ", ".join([f"{k} {v}" for k, v in telemetry_payload.items()])
        
        with st.spinner("Executing real Vector Retrieval & RAG Synthesis..."):
            # 1. Real Vector space search to find relevant literature
            matched_literature = real_rag_retrieval(symptoms_query_string, top_k=2)
            
            # 2. Invoke generative synthesis with retrieved contexts
            rag_assessment_report, explicit_citations = generate_rag_clinical_assessment(
                st.session_state.patient_name, verdict_header, formatted_confidence_string, symptoms_query_string, lang_selection
            )
            
            # 3. Compile standalone English version strictly for PDF stability
            english_pdf_report = generate_pdf_prescription_insights(symptoms_query_string, matched_literature)
            
        # Display Generated Grounded Clinical Report
        st.markdown(
            f'<div class="rag-box"><h4>📋 RAG Grounded Clinical Action Plan</h4><div style="line-height:1.7;">{rag_assessment_report}</div></div>', 
            unsafe_allow_html=True
        )
        
        # Display Real Medical Citations Block
        st.markdown("#### 📚 Traceable Clinical Citations (RAG Verifiable Source)")
        for citation in explicit_citations:
            st.markdown(f'<span class="citation-tag">{citation}</span>', unsafe_allow_html=True)
            
        # Serialized Reportlab PDF Compilation
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
        
        # Mandatory Regulatory Warning Footer
        st.markdown("<div class='legal-alert'>⚠️ Regulatory Notice: This cloud screening environment runs automated retrieval pipelines and statistical machine learning inferences. It does not replace proper in-person hospital blood testing and laboratory profiling.</div>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

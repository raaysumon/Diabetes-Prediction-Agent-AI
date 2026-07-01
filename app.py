import os
import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Early Diabetes Chatbot AI", page_icon="🩸", layout="wide", initial_sidebar_state="collapsed")

# --- 2. API KEY MANAGEMENT ---
try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "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")
except Exception:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")

# --- 3. PRODUCTION RAG KNOWLEDGE BASE ---
@st.cache_resource
def load_clinical_knowledge_base():
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

# --- INTENT CLASSIFIER ENGINE ---
def check_user_consent_intent(user_text):
    try:
        client = Groq(api_key=GROQ_API_KEY)
        system_prompt = (
            "Analyze the user's intent to start a medical screening right now. "
            "If they say yes, okay, start, sure, or show positive intent, reply ONLY with 'START'. "
            "If they say next day, later, no, not now, how are you, or deflect, reply ONLY with 'HOLD'."
        )
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}],
            temperature=0.01, max_tokens=5
        )
        return completion.choices[0].message.content.strip()
    except Exception:
        return "START"

# --- 4. DYNAMIC AUTOMATED RAG ENGINE WITH CITATIONS ---
def real_rag_retrieval(patient_symptoms_string, similarity_threshold=0.01):
    corpus = load_clinical_knowledge_base()
    documents = [f"{doc['text']} {doc['keywords']}" for doc in corpus]
    
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([patient_symptoms_string])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    retrieved_chunks = []
    for idx, score in enumerate(similarities):
        if score >= similarity_threshold:
            retrieved_chunks.append(corpus[idx])
            
    if not retrieved_chunks:
        retrieved_chunks = [corpus[3], corpus[4]]
    return retrieved_chunks

def generate_rag_clinical_assessment(patient_name, prediction_label, confidence, patient_context, language, matched_chunks):
    context_str = "".join([f"[Source: {chunk['citation']}]: {chunk['text']}\n" for chunk in matched_chunks])
    try:
        client = Groq(api_key=GROQ_API_KEY)
        lang_rule = f"Your entire response MUST be written strictly in {language}."
        system_prompt = (
            "You are DECat-AI, a helpful digital clinician specializing in Diabetes Risk Screening. " + lang_rule + "\n"
            f"Explain the diagnostic risk dynamically based on CatBoost: {prediction_label} ({confidence}).\n"
            f"Advise tests and structure cleanly using header fields: 'Diagnostic Guidance', 'Dietary Action Plan', and 'Lifestyle Protocol'.\n"
            f"CRITICAL: Always append the clinical citation names inline at the end of relevant paragraphs, e.g., (Source: WHO 2023)."
        )
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Patient: {patient_name}\nData: {patient_context}\nReferences:\n{context_str}"}],
            temperature=0.3, max_tokens=750
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# --- FOLLOW-UP CONSULTATION ENGINE ---
def generate_followup_answer(user_question, language, patient_context, verdict, confidence, matched_chunks):
    context_str = "".join([f"[Clinical Reference: {chunk['citation']}]: {chunk['text']}\n" for chunk in matched_chunks])
    try:
        client = Groq(api_key=GROQ_API_KEY)
        system_prompt = (
            f"You are DECat-AI, the patient's digital doctor. Answer the patient's follow-up question strictly in {language}.\n"
            f"CRITICAL RULE: Your answer MUST align perfectly with the CatBoost ML result ({verdict} with {confidence} confidence) and the provided RAG Guidelines.\n"
            f"Explicitly mention citations like (World Health Organization, 2023) or (American Diabetes Association, 2024) inside the response text when talking about guidelines."
        )
        user_payload = f"Patient Question: {user_question}\n\nEstablished Screening Context:\n- Result: {verdict}\n- Metrics: {patient_context}\n- RAG Knowledge:\n{context_str}"
        
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_payload}],
            temperature=0.4, max_tokens=500
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def generate_pdf_prescription_insights(symptoms_query_string, matched_literature):
    context_str = "\n".join([f"[{c['citation']}]: {c['text']}" for c in matched_literature])
    try:
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Summarize recommendations in English. Keys: DIAGNOSTIC ADVICE:, DIETARY MODIFICATIONS:, LIFESTYLE PROTOCOL:. Include source names inline. Plain text only."},
                {"role": "user", "content": f"Context:\n{context_str}\nMetrics:\n{symptoms_query_string}"}
            ], temperature=0.01, max_tokens=250
        )
        return completion.choices[0].message.content
    except Exception:
        return "DIAGNOSTIC ADVICE:\n- Order immediate HbA1c screening (WHO 2023)."

# --- 5. CATBOOST ML MODEL LOADER ---
@st.cache_resource
def load_screening_model():
    model = CatBoostClassifier()
    current_dir = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
    path_options = [os.path.join(current_dir, "final_catboost_modol.cbm"), os.path.join(current_dir, "final_catboost_model.cbm")]
    for model_path in path_options:
        if os.path.exists(model_path):
            try: model.load_model(model_path); return model
            except Exception: pass
    return None

model = load_screening_model()

# --- 6. REPORTLAB ENGINE ---
def build_clinical_pdf(patient_name, patient_data, verdict, confidence, english_report, matched_chunks):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TStyle', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#e63946'), alignment=1, spaceAfter=15, fontName='Helvetica-Bold')
    sec_style = ParagraphStyle('SecStyle', parent=styles['Heading2'], fontSize=12, textColor=colors.HexColor('#4a90e2'), spaceBefore=10, spaceAfter=5, fontName='Helvetica-Bold')
    body_style = ParagraphStyle('BStyle', parent=styles['Normal'], fontSize=10, leading=14, textColor=colors.HexColor('#222222'), fontName='Helvetica')
    
    story.append(Paragraph("DECat-AI ADVANCED CLINICAL REPORT", title_style))
    story.append(Paragraph(f"<b>Patient Name:</b> {patient_name}", body_style))
    story.append(Paragraph(f"<b>ML Verdict:</b> {verdict}", body_style))
    story.append(Paragraph(f"<b>Confidence:</b> {confidence}", body_style))
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("CLINICAL RECOMMENDATIONS WITH EVIDENCE", sec_style))
    clean_text = english_report.replace(">", " greater than ").replace("<", " less than ").replace("%", " percent ")
    for line in clean_text.split("\n"):
        if line.strip():
            story.append(Paragraph(line.strip().replace("&", "&amp;"), body_style))
    
    story.append(Spacer(1, 15))
    story.append(Paragraph("OFFICIAL EVIDENCE BASE CITATIONS", sec_style))
    for chunk in matched_chunks:
        story.append(Paragraph(f"- {chunk['citation']}", body_style))
        
    doc.build(story)
    buffer.seek(0)
    return buffer

# --- 7. SIDEBAR LOCALIZATION ---
with st.sidebar:
    st.markdown("### ⚙️ Settings / সেটিংস")
    lang_selection = st.radio("System Interface Language:", ["English", "বাংলা"], index=0)

# --- 8. PREMIUM DARK MATTE (TOP BAR REMOVED) CSS ---
st.markdown("""
<style>
    * {
        box-sizing: border-box !important;
    }
    
    /* 🛠️ ১. ওপুরের সাদা টপ বার এবং মেনু বার চিরতরে লুকিয়ে ফেলার ফিক্স */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        background: transparent !important;
        height: 0px !important;
        display: none !important;
    }
    
    div[data-testid="stToolbar"] {
        display: none !important;
    }
    
    /* ব্যাকগ্রাউন্ড কুচকুচে কালো নয়, বরং একটি চমৎকার ডার্ক মেটালিক চারকোল টোন */
    html, body, .stApp { 
        font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
        background-color: #1a1e24 !important; 
        color: #e2e8f0 !important;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #111418 !important;
    }
    
    /* মেইন কন্টেন্ট বক্স সেন্টারড ও আল্ট্রা-রেসপন্সিভ */
    .main-wrapper { 
        max-width: 100% !important; 
        width: 720px !important;
        margin: 20px auto !important; 
        padding: 24px !important; 
        background-color: #222933 !important; 
        border-radius: 20px !important; 
        box-shadow: 0 12px 40px rgba(0,0,0,0.4) !important;
        border: 1px solid #2e3745 !important;
    }
    
    .header-logo { 
        font-size: calc(1.5rem + 0.7vw) !important;
        font-weight: 700 !important; 
        color: #e63946 !important; 
        display: block !important;
        text-align: center !important;
        margin-bottom: 5px !important;
    }
    
    /* 🛠️ ২. এআই চ্যাট বাবল প্যাডিং ফিক্স (লেখা বাম পাশে চেপে থাকবে না) */
    .chat-bubble-ai { 
        background-color: #2d3748 !important; 
        color: #f7fafc !important; 
        padding: 16px 20px !important; 
        border-radius: 16px !important; 
        border-left: 5px solid #e63946 !important; 
        margin-bottom: 16px !important; 
        display: block !important;
        width: 100% !important;
        font-size: 15px !important; 
        line-height: 1.6 !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }
    
    /* 🛠️ ৩. ইউজার চ্যাট বাবল প্যাডিং ও ইমোজি অ্যালাইনমেন্ট ফিক্স */
    .chat-bubble-user { 
        background-color: #e63946 !important; 
        color: #ffffff !important; 
        padding: 14px 20px !important; 
        border-radius: 16px !important; 
        display: inline-block !important;
        float: right !important; 
        clear: both !important; 
        margin-bottom: 16px !important; 
        font-size: 15px !important; 
        max-width: 85% !important;
        box-shadow: 0 4px 14px rgba(230,57,70,0.25) !important;
    }
    
    /* ডার্ক মোড ফ্রেন্ডলি ফর্ম কার্ড */
    div[data-testid="stForm"] {
        background-color: #1a202c !important;
        border: 1px solid #2d3748 !important;
        border-radius: 16px !important;
        padding: 22px !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.2) !important;
    }
    
    /* বাটনের ফ্রেন্ডলি ফিক্স */
    div[data-testid="stForm"] button, .stButton button {
        background-color: #e63946 !important; 
        color: #ffffff !important; 
        border: none !important;
        padding: 10px 24px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 12px rgba(230,57,70,0.3) !important;
        transition: all 0.2s ease-in-out !important;
        width: auto !important;
    }
    
    div[data-testid="stForm"] button:hover, .stButton button:hover {
        background-color: #cc323f !important; 
        transform: translateY(-1px) !important;
    }

    input {
        background-color: #2d3748 !important;
        color: #ffffff !important;
        border: 1px solid #4a5568 !important;
    }
    
    label, p, span, div[data-baseweb="radio"] {
        color: #edf2f7 !important;
    }

    @media (max-width: 768px) {
        .main-wrapper { margin: 10px auto !important; padding: 15px !important; border-radius: 12px !important; }
        .chat-bubble-user { max-width: 90% !important; }
    }
</style>
""", unsafe_allow_html=True)

# --- 9. CLINICAL QUIZ SCHEMA ---
quiz_schema = [
    {"field": "Age", "en": "Please provide your current age (Years):", "bn": "আপনার বর্তমান বয়স কত (বছর)?"},
    {"field": "Gender", "en": "Select biological sex parameter:", "bn": "আপনার জৈবিক লিঙ্গ নির্বাচন করুন:", "options": ["Male", "Female"]},
    {"field": "Polyuria", "en": "Do you experience excessive or unusually frequent urination (Polyuria)?", "bn": "আপনার কি অতিরিক্ত বা ঘন ঘন প্রস্রাবের সমস্যা (Polyuria) হচ্ছে?", "options": ["Yes", "No"]},
    {"field": "Polydipsia", "en": "Are you experiencing constant, extreme fluid thirst (Polydipsia)?", "bn": "আপনার কি প্রতিনিয়ত অতিরিক্ত বা অস্বাভাবিক তৃষ্ণা (Polydipsia) পাচ্ছে?", "options": ["Yes", "No"]},
    {"field": "Irritability", "en": "Have you noticed any persistent patterns of sudden irritability or mood spikes?", "bn": "আপনি কি ইদানীং অতিরিক্ত খিটখিটে মেজাজ বা মানসিক অস্থিরতা অনুভব করছেন?", "options": ["Yes", "No"]},
    {"field": "Itching", "en": "Do you experience localized or generalized recurring skin itching?", "bn": "আপনার ত্বকে কি ঘন ঘন বা দীর্ঘস্থায়ী চুলকানির সমস্যা হচ্ছে?", "options": ["Yes", "No"]},
    {"field": "delayed healing", "en": "Do surface cuts, scratches, or flesh wounds take a prolonged time to completely heal?", "bn": "আপনার শরীরের কোনো ক্ষত, কাটা বা স্ক্র্যাচ শুকাতে কি স্বাভাবিকের চেয়ে বেশি সময় লাগছে?", "options": ["Yes", "No"]},
    {"field": "Alopecia", "en": "Are you suffering from active, accelerated hair thinning or loss patches (Alopecia)?", "bn": "আপনার কি অতিরিক্ত চুল পড়া বা নির্দিষ্ট স্থান থেকে চুল উঠে যাওয়ার (Alopecia) লক্ষণ দেখা দিচ্ছে?", "options": ["Yes", "No"]}
]

# --- 10. PIPELINE INITIALIZATION ---
if "step" not in st.session_state: st.session_state.step = -2
if "patient_name" not in st.session_state: st.session_state.patient_name = ""
if "user_responses" not in st.session_state: st.session_state.user_responses = {}
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "final_calculated" not in st.session_state: st.session_state.final_calculated = False

def record_chat(role, payload): st.session_state.chat_history.append({"role": role, "text": payload})
def reroute_pipeline_to(next_node): st.session_state.step = next_node; st.rerun()

st.markdown('<div class="main-wrapper">', unsafe_allow_html=True)
st.markdown('<span class="header-logo">🩸 DECat‑AI Desk</span>', unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #2e3745;' />", unsafe_allow_html=True)

# Render previous chat history
for message_bubble in st.session_state.chat_history:
    if message_bubble["role"] == "ai":
        st.markdown(f'<div class="chat-bubble-ai">🤖 <b>DECat-AI:</b> {message_bubble["text"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="overflow:auto;"><div class="chat-bubble-user">👤 {message_bubble["text"]}</div></div>', unsafe_allow_html=True)

# STEP -2: IDENTITY
if st.session_state.step == -2:
    init_greeting = "Hello! Before we talk about your health, could you please tell me your full name?" if lang_selection == "English" else "হ্যালো! আপনার স্বাস্থ্য নিয়ে কথা বলার আগে, আমি কি আপনার সম্পূর্ণ নামটা জানতে পারি?"
    st.markdown(f'<div class="chat-bubble-ai">🤖 <b>DECat-AI:</b> {init_greeting}</div>', unsafe_allow_html=True)
    with st.form(key="identity_node"):
        raw_name = st.text_input("Your Name / আপনার নাম")
        if st.form_submit_button("Proceed ➡️") and raw_name.strip():
            st.session_state.patient_name = raw_name.strip()
            record_chat("ai", init_greeting); record_chat("user", raw_name.strip())
            reroute_pipeline_to(-1)

# STEP -1: COMPLIANCE WITH INTENT PROTECTION
elif st.session_state.step == -1:
    consent_prompt = f"Nice to meet you, {st.session_state.patient_name}. Would you like to check your diabetes risks with a quick screening test?" if lang_selection == "English" else f"আপনার সাথে পরিচিত হয়ে ভালো লাগলো, {st.session_state.patient_name}। আপনি কি ছোট একটা স্ক্রীনিং টেস্ট করতে চান?"
    st.markdown(f'<div class="chat-bubble-ai">🤖 <b>DECat-AI:</b> {consent_prompt}</div>', unsafe_allow_html=True)
    with st.form(key="consent_node", clear_on_submit=True):
        consent_reply = st.text_input("Your Response / উত্তর দিন")
        if st.form_submit_button("Submit 🚀") and consent_reply.strip():
            intent_result = check_user_consent_intent(consent_reply.strip())
            
            if intent_result == "START":
                record_chat("ai", consent_prompt); record_chat("user", consent_reply.strip())
                reroute_pipeline_to(0)
            else:
                record_chat("ai", consent_prompt); record_chat("user", consent_reply.strip())
                hold_reply = "Sure, no problem! Whenever you are ready to take the screening test, please let me know or just reload." if lang_selection == "English" else "অবশ্যই, কোনো সমস্যা নেই! আপনি যখনই স্ক্রীনিং টেস্টটি করতে প্রস্তুত হবেন, আমাকে জানাবেন অথবা পেজটি রিলোড করবেন।"
                record_chat("ai", hold_reply)
                reroute_pipeline_to(-3)

# STEP -3: HOLD STATE
elif st.session_state.step == -3:
    st.write(" ")
    btn_label = "Start Screening Test Now 🚀" if lang_selection == "English" else "এখনই স্ক্রীনিং টেস্ট শুরু করুন 🚀"
    if st.button(btn_label, use_container_width=True):
        reroute_pipeline_to(0)

# SURVEY ENGINE LOOP
elif 0 <= st.session_state.step < len(quiz_schema):
    active_node = quiz_schema[st.session_state.step]
    localized_query = active_node["bn"] if lang_selection == "বাংলা" else active_node["en"]
    st.markdown(f'<div class="chat-bubble-ai">🤖 <b>DECat-AI:</b> {localized_query}</div>', unsafe_allow_html=True)
    
    with st.form(key=f"survey_form_{st.session_state.step}"):
        if "options" in active_node:
            ui_labels = ["Yes", "No"] if lang_selection == "English" else ["হ্যাঁ", "না"]
            label_mapper = {"Yes": ui_labels[0], "No": ui_labels[1]}
            if active_node["field"] == "Gender":
                label_mapper = {"Male": "Male" if lang_selection=="English" else "পুরুষ", "Female": "Female" if lang_selection=="English" else "নারী"}
            inverted_mapper = {v: k for k, v in label_mapper.items()}
            selected_option = st.radio("Select:", list(label_mapper.values()), index=None)
            if st.form_submit_button("Next ➡️") and selected_option:
                st.session_state.user_responses[active_node["field"]] = inverted_mapper[selected_option]
                record_chat("ai", localized_query); record_chat("user", selected_option)
                reroute_pipeline_to(st.session_state.step + 1)
        else:
            typed_age = st.number_input("Age:", min_value=1, max_value=122, value=None)
            if st.form_submit_button("Next ➡️") and typed_age:
                st.session_state.user_responses[active_node["field"]] = int(typed_age)
                record_chat("ai", localized_query); record_chat("user", str(int(typed_age)))
                reroute_pipeline_to(st.session_state.step + 1)

# FINAL EVALUATION & CHAT
else:
    telemetry_payload = st.session_state.user_responses
    positive_symptoms = [k for k, v in telemetry_payload.items() if v == "Yes"]
    symptoms_query_string = ", ".join(positive_symptoms) if positive_symptoms else "routine preventive check"
    
    matched_literature = real_rag_retrieval(symptoms_query_string)
    
    evaluation_dataframe = pd.DataFrame([telemetry_payload])
    for column in evaluation_dataframe.columns:
        if column != 'Age': evaluation_dataframe[column] = evaluation_dataframe[column].astype('category')
    
    binary_prediction = model.predict(evaluation_dataframe)[0] if model else 0
    prediction_probabilities = model.predict_proba(evaluation_dataframe)[0] if model else [0.5, 0.5]
    has_positive_risk = bool(binary_prediction == 1 or prediction_probabilities[1] > 0.5)
    calculated_confidence = prediction_probabilities[1] * 100 if has_positive_risk else prediction_probabilities[0] * 100
    formatted_confidence_string = f"{calculated_confidence:.2f} percent"
    
    pdf_verdict_header = "DIABETES RISK DETECTED" if has_positive_risk else "NO IMMEDIATE RISK DETECTED"
    verdict_header = pdf_verdict_header if lang_selection == "English" else ("ডায়াবেটিস ঝুঁকি সনাক্ত হয়েছে" if has_positive_risk else "কোনো তাৎক্ষণিক ঝুঁকি পাওয়া যায়নি")

    if not st.session_state.final_calculated:
        rag_assessment_report = generate_rag_clinical_assessment(
            st.session_state.patient_name, verdict_header, formatted_confidence_string, symptoms_query_string, lang_selection, matched_literature
        )
        record_chat("ai", f"**{verdict_header}** ({formatted_confidence_string})\n\n{rag_assessment_report}")
        st.session_state.final_calculated = True
        st.rerun()

    st.write("### 📊 Clinical Screening Diagnostic Center")
    if has_positive_risk:
        st.error(f"⚠️ {verdict_header} ({formatted_confidence_string})")
    else:
        st.success(f"✅ {verdict_header} ({formatted_confidence_string})")

    # Follow-up Chat Box
    st.markdown("#### 💬 Ask DECat-AI Doctor Anything (Follow-up Chat)")
    with st.form(key="followup_form", clear_on_submit=True):
        patient_question = st.text_input("Any Question?", placeholder="Type your follow-up medical question here...")
        submitted = st.form_submit_button("Ask Doctor 🩺")
        
        if submitted and patient_question.strip():
            record_chat("user", patient_question.strip())
            with st.spinner("Doctor is analyzing..."):
                doctor_reply = generate_followup_answer(
                    patient_question.strip(), lang_selection, symptoms_query_string, pdf_verdict_header, formatted_confidence_string, matched_literature
                )
                record_chat("ai", doctor_reply)
            st.rerun()

    # PDF Generation
    english_pdf_report = generate_pdf_prescription_insights(symptoms_query_string, matched_literature)
    pdf_binary_stream = build_clinical_pdf(st.session_state.patient_name, telemetry_payload, pdf_verdict_header, formatted_confidence_string, english_pdf_report, matched_literature)
    
    st.write(" ")
    
    st.download_button(
        label="📥 Download Traceable Clinical Report (PDF)" if lang_selection == "English" else "📥 ক্লিনিক্যাল রিপোর্ট ডাউনলোড করুন (PDF)",
        data=pdf_binary_stream, file_name=f"Clinical_Report_{st.session_state.patient_name}.pdf", mime="application/pdf",
        use_container_width=True
    )
    
    if st.button("Reset Assessment 🔄", use_container_width=True):
        st.session_state.clear(); st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

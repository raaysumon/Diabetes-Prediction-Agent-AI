import os
import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit.components.v1 as components
import json

# --- 1. PAGE CONFIGURATION & UI CLEANUP ---
st.set_page_config(page_title="Early Diabetes Chatbot AI", page_icon="🩸", layout="wide", initial_sidebar_state="collapsed")

# Streamlit-এর নিজস্ব হেডার/ফুটার হাইড করা (পারফরম্যান্স ও ক্লিন লুকের জন্য)
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

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
        }
    ]

# --- 4. CATBOOST ML MODEL LOADER ---
@st.cache_resource
def load_screening_model():
    model = CatBoostClassifier()
    current_dir = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
    path_options = [os.path.join(current_dir, "final_catboost_modol.cbm"), os.path.join(current_dir, "final_catboost_model.cbm")]
    for model_path in path_options:
        if os.path.exists(model_path):
            try: 
                model.load_model(model_path)
                return model
            except Exception: 
                pass
    return None

model = load_screening_model()

def real_rag_retrieval(patient_symptoms_string):
    corpus = load_clinical_knowledge_base()
    documents = [f"{doc['text']} {doc['keywords']}" for doc in corpus]
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
        query_vector = vectorizer.transform([patient_symptoms_string])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        retrieved_chunks = [corpus[idx] for idx, score in enumerate(similarities) if score >= 0.01]
        return retrieved_chunks if retrieved_chunks else corpus
    except:
        return corpus

# --- 5. JAVASCRIPT TO STREAMLIT BRIDGE ENGINE ---
query_params = st.query_params

if "js_payload" in query_params:
    try:
        raw_data = query_params["js_payload"]
        data = json.loads(raw_data)
        
        patient_name = data.get("name", "Patient")
        language = data.get("lang", "English")
        telemetry_payload = data.get("responses", {})
        
        # ML Evaluation via CatBoost
        evaluation_dataframe = pd.DataFrame([telemetry_payload])
        for column in evaluation_dataframe.columns:
            if column != 'Age': 
                evaluation_dataframe[column] = evaluation_dataframe[column].astype('category')
            
        binary_prediction = model.predict(evaluation_dataframe)[0] if model else 0
        prediction_probabilities = model.predict_proba(evaluation_dataframe)[0] if model else [0.5, 0.5]
        has_positive_risk = bool(binary_prediction == 1 or prediction_probabilities[1] > 0.5)
        calculated_confidence = prediction_probabilities[1] * 100 if has_positive_risk else prediction_probabilities[0] * 100
        formatted_confidence_string = f"{calculated_confidence:.2f}%"
        
        verdict_header = "DIABETES RISK DETECTED" if has_positive_risk else "NO IMMEDIATE RISK DETECTED"
        if language == "বাংলা":
            verdict_header = "ডায়াবেটিস ঝুঁকি সনাক্ত হয়েছে" if has_positive_risk else "কোনো তাৎক্ষণিক ঝুঁকি পাওয়া যায়নি"
            
        # Dynamic RAG Search & Prompt Engineering
        positive_symptoms = [k for k, v in telemetry_payload.items() if v == "Yes"]
        symptoms_query_string = ", ".join(positive_symptoms) if positive_symptoms else "routine preventive check"
        matched_literature = real_rag_retrieval(symptoms_query_string)
        context_str = "".join([f"[Source: {chunk['citation']}]: {chunk['text']}\n" for chunk in matched_literature])
        
        client = Groq(api_key=GROQ_API_KEY)
        system_prompt = f"You are DECat-AI, a digital clinician. Response MUST be strictly in {language}. Explain risk: {verdict_header} ({formatted_confidence_string}). Format beautifully with clear headers like 'Diagnostic Guidance', 'Dietary Action Plan', and 'Lifestyle Protocol'. Append inline citations like (Source: WHO 2023)."
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Patient: {patient_name}\nData: {symptoms_query_string}\nReferences:\n{context_str}"}],
            temperature=0.3, max_tokens=600
        )
        report_output = completion.choices[0].message.content
        
        # জাভাস্ক্রিপ্টের Fetch API এর কাছে JSON রেসপন্স রিটার্ন করা
        st.json({"status": "success", "verdict": verdict_header, "confidence": formatted_confidence_string, "has_risk": has_positive_risk, "report": report_output})
        st.stop()
    except Exception as e:
        st.json({"status": "error", "message": str(e)})
        st.stop()

# --- 6. UNIVERSAL HIGH-SPEED HTML/CSS/JS UI ---
html_interface = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background-color: #fcfdfe; color: #212121; padding: 10px; }
        .wrapper { max-width: 600px; margin: 0 auto; background: white; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
        h2 { color: #d32f2f; text-align: center; margin-bottom: 15px; font-size: 1.6rem; }
        .lang-box { text-align: right; margin-bottom: 10px; }
        select { padding: 4px; border-radius: 4px; border: 1px solid #ccc; font-size: 14px; }
        .chat-area { min-height: 180px; max-height: 350px; overflow-y: auto; border: 1px solid #eee; padding: 8px; margin-bottom: 12px; border-radius: 6px; }
        .msg { padding: 10px; margin-bottom: 8px; border-radius: 6px; font-size: 14px; max-width: 85%; line-height: 1.4; }
        .bot { background: #f8f9fa; border-left: 4px solid #d32f2f; float: left; clear: both; }
        .user { background: #d32f2f; color: white; float: right; clear: both; }
        .action-zone { margin-top: 10px; display: flex; gap: 8px; clear: both; }
        input { flex: 1; padding: 8px; border: 1px solid #ccc; border-radius: 4px; font-size: 14px; }
        button { background: #d32f2f; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-size: 14px; }
        button:hover { background: #b71c1c; }
        .btn-group { display: flex; gap: 8px; margin-top: 5px; clear: both; }
        .btn-opt { background: #e0e0e0; color: #212121; }
        .btn-opt:hover { background: #d5d5d5; }
        .result-zone { display: none; margin-top: 15px; padding: 15px; border-radius: 6px; clear: both; }
        .risk { background: #ffebee; border: 1px solid #ffcdd2; color: #c62828; }
        .safe { background: #e8f5e9; border: 1px solid #c8e6c9; color: #2e7d32; }
        .output-text { white-space: pre-wrap; font-size: 14px; margin-top: 12px; line-height: 1.5; color: #333; }
    </style>
</head>
<body>

<div class="wrapper">
    <h2>🩸 DECat-AI Desk</h2>
    <div class="lang-box">
        <select id="sysLang" onchange="toggleLanguage()">
            <option value="English">English</option>
            <option value="বাংলা">বাংলা</option>
        </select>
    </div>
    <div class="chat-area" id="chatDisplay"></div>
    <div id="controlDisplay"></div>

    <div class="result-zone" id="finalResult">
        <h3 id="verdictTitle"></h3>
        <div class="output-text" id="reportDisplay"></div>
        <button onclick="restartApp()" style="margin-top:15px; background:#555;">Restart Assessment 🔄</button>
    </div>
</div>

<script>
    let step = -2;
    let lang = "English";
    let pName = "";
    let userAnswers = {};

    const stepsSchema = [
        { field: "Age", type: "num", en: "Please provide your current age (Years):", bn: "আপনার বর্তমান বয়স কত (বছর)?" },
        { field: "Gender", type: "opt", opts: ["Male", "Female"], en: "Select biological sex parameter:", bn: "আপনার জৈবিক লিঙ্গ নির্বাচন করুন:" },
        { field: "Polyuria", type: "opt", opts: ["Yes", "No"], en: "Do you experience excessive or unusually frequent urination (Polyuria)?", bn: "আপনার কি অতিরিক্ত বা ঘন ঘন প্রস্রাবের সমস্যা (Polyuria) হচ্ছে?" },
        { field: "Polydipsia", type: "opt", opts: ["Yes", "No"], en: "Are you experiencing constant, extreme fluid thirst (Polydipsia)?", bn: "আপনার কি প্রতিনিয়ত অতিরিক্ত বা অস্বাভাবিক তৃষ্ণা (Polydipsia) পাচ্ছে?" },
        { field: "Irritability", type: "opt", opts: ["Yes", "No"], en: "Have you noticed any persistent patterns of sudden irritability or mood spikes?", bn: "আপনি কি ইদানীং অতিরিক্ত খিটখিটে মেজাজ বা মানসিক অস্থিরতা অনুভব করছেন?" },
        { field: "Itching", type: "opt", opts: ["Yes", "No"], en: "Do you experience localized or generalized recurring skin itching?", bn: "আপনার ত্বকে কি ঘন ঘন বা দীর্ঘস্থায়ী চুলকানির সমস্যা হচ্ছে?" },
        { field: "delayed healing", type: "opt", opts: ["Yes", "No"], en: "Do surface cuts, scratches, or flesh wounds take a prolonged time to completely heal?", bn: "আপনার শরীরের কোনো ক্ষত, কাটা বা স্ক্র্যাচ শুকাতে কি স্বাভাবিকের চেয়ে বেশি সময় লাগছে?" },
        { field: "Alopecia", type: "opt", opts: ["Yes", "No"], en: "Are you suffering from active, accelerated hair thinning or loss patches (Alopecia)?", bn: "আপনার কি অতিরিক্ত চুল পড়া বা নির্দিষ্ট স্থান থেকে চুল উঠে যাওয়ার (Alopecia) লক্ষণ দেখা দিচ্ছে?" }
    ];

    function printMsg(txt, type) {
        const box = document.getElementById("chatDisplay");
        const div = document.createElement("div");
        div.className = `msg ${type}`;
        div.innerHTML = `<b>${type==='bot'?'DECat-AI':'You'}:</b> ${txt}`;
        box.appendChild(div);
        box.scrollTop = box.scrollHeight;
    }

    function toggleLanguage() {
        lang = document.getElementById("sysLang").value;
        restartApp();
    }

    function drawUI() {
        const zone = document.getElementById("controlDisplay");
        zone.innerHTML = "";

        if (step === -2) {
            let msg = lang === "English" ? "Hello! Before we talk about your health, could you please tell me your full name?" : "হ্যালো! আপনার স্বাস্থ্য নিয়ে কথা বলার আগে, আমি কি আপনার সম্পূর্ণ নামটা জানতে পারি?";
            printMsg(msg, "bot");
            zone.innerHTML = `<div class="action-zone">
                <input type="text" id="nameInput" placeholder="Name...">
                <button onclick="saveName()">Next ➡️</button>
            </div>`;
        }
        else if (step === -1) {
            let msg = lang === "English" ? `Nice to meet you, ${pName}. Would you like to check your diabetes risks with a quick screening test?` : `আপনার সাথে পরিচিত হয়ে ভালো লাগলো, ${pName}। আপনি কি ছোট একটা স্ক্রীনিং টেস্ট করতে চান?`;
            printMsg(msg, "bot");
            zone.innerHTML = `<div class="btn-group">
                <button class="btn-opt" onclick="confirmConsent(true)">${lang==='English'?'Yes':'হ্যাঁ'}</button>
                <button class="btn-opt" onclick="confirmConsent(false)">${lang==='English'?'No':'না'}</button>
            </div>`;
        }
        else if (step >= 0 && step < stepsSchema.length) {
            let current = stepsSchema[step];
            printMsg(lang === "English" ? current.en : current.bn, "bot");

            if (current.type === "num") {
                zone.innerHTML = `<div class="action-zone">
                    <input type="number" id="ageInput" min="1" max="115">
                    <button onclick="saveAge()">Next ➡️</button>
                </div>`;
            } else {
                let html = `<div class="btn-group">`;
                current.opts.forEach(o => {
                    let lbl = o;
                    if(lang === "বাংলা") {
                        if(o==="Yes") lbl="হ্যাঁ"; if(o==="No") lbl="না";
                        if(o==="Male") lbl="পুরুষ"; if(o==="Female") lbl="নারী";
                    }
                    html += `<button class="btn-opt" onclick="saveOpt('${o}', '${lbl}')">${lbl}</button>`;
                });
                html += `</div>`;
                zone.innerHTML = html;
            }
        } else if (step === stepsSchema.length) {
            zone.innerHTML = `<div style="padding:10px;">Processing data through ML Engine... ⏳</div>`;
            sendToStreamlit();
        }
    }

    function saveName() {
        let input = document.getElementById("nameInput").value.trim();
        if(!input) return;
        pName = input;
        printMsg(pName, "user");
        step = -1;
        drawUI();
    }

    function confirmConsent(agreed) {
        printMsg(agreed ? (lang==='English'?'Yes':'হ্যাঁ') : (lang==='English'?'No':'না'), "user");
        if(agreed) { 
            step = 0; 
            drawUI(); 
        } else { 
            printMsg(lang==='English'?'Screening on hold. Restart or refresh to activate.':'টেস্ট স্থগিত করা হয়েছে। শুরু করতে রিস্টার্ট বা পেজ রিফ্রেশ করুন।', 'bot'); 
            document.getElementById("controlDisplay").innerHTML=""; 
        }
    }

    function saveAge() {
        let input = document.getElementById("ageInput").value;
        if(!input) return;
        userAnswers[stepsSchema[step].field] = parseInt(input);
        printMsg(input, "user");
        step++;
        drawUI();
    }

    function saveOpt(val, lbl) {
        userAnswers[stepsSchema[step].field] = val;
        printMsg(lbl, "user");
        step++;
        drawUI();
    }

    function sendToStreamlit() {
        const payload = JSON.stringify({ name: pName, lang: lang, responses: userAnswers });
        const targetUrl = window.location.origin + window.location.pathname + "?js_payload=" + encodeURIComponent(payload);
        
        fetch(targetUrl)
        .then(res => res.json())
        .then(data => {
            document.getElementById("controlDisplay").innerHTML = "";
            let resBox = document.getElementById("finalResult");
            resBox.style.display = "block";
            resBox.className = "result-zone " + (data.has_risk ? "risk" : "safe");
            document.getElementById("verdictTitle").innerText = data.verdict + " (" + data.confidence + ")";
            document.getElementById("reportDisplay").innerText = data.report;
        })
        .catch(() => {
            document.getElementById("controlDisplay").innerHTML = `<div style="color:#d32f2f; font-size:14px; padding:10px;">Submission complete. Please check the screen or refresh.</div>`;
        });
    }

    function restartApp() {
        step = -2; 
        userAnswers = {};
        document.getElementById("chatDisplay").innerHTML = "";
        document.getElementById("finalResult").style.display = "none";
        drawUI();
    }

    drawUI();
</script>
</body>
</html>
"""

# Streamlit UI স্ক্রিনে পুরো কম্পোনেন্টটিকে এম্বেড করা
components.html(html_interface, height=650, scrolling=True)

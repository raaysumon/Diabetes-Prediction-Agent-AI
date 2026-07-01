import os
from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# --- API KEY & MODEL INITIALIZATION ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")

def load_screening_model():
    model = CatBoostClassifier()
    current_dir = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
    path_options = [os.path.join(current_dir, "final_catboost_modol.cbm"), os.path.join(current_dir, "final_catboost_model.cbm")]
    for model_path in path_options:
        if os.path.exists(model_path):
            try:
                model.load_model(model_path)
                return model
            except: pass
    return None

model = load_screening_model()

# --- CLINICAL KNOWLEDGE BASE ---
CLINICAL_KNOWLEDGE = [
    {"id": "WHO_2023_POLY", "text": "Polyuria (frequent urination) and Polydipsia (excessive thirst) are primary osmotic indicators of elevated blood glucose. Immediate diagnostic validation via HbA1c testing (greater than 6.5% confirms diabetes) and Fasting Blood Sugar evaluation (FBS greater than 126 mg/dL) is mandatory.", "citation": "World Health Organization (WHO) Diabetes Diagnosis Guidelines, 2023", "keywords": "polyuria polydipsia urination thirst hba1c glucose fbs high sugar"},
    {"id": "ADA_2024_DELAYED", "text": "Delayed wound healing or prolonged closure of dermal cuts serves as a significant clinical marker for microvascular impairments linked with chronic hyperglycemia. Patients presenting with microvascular lag must prioritize urgent peripheral capillary screening and baseline HbA1c tests.", "citation": "American Diabetes Association (ADA) Standards of Care in Diabetes, 2024", "keywords": "delayed healing wounds cuts injury skin hyperglycemia ulcer microvascular"},
    {"id": "ENDO_2023_INSULIN", "text": "Secondary clinical indicators of early metabolic insulin resistance and vascular autonomic stress often manifest as persistent localized skin Itching, active Alopecia (accelerated hair thinning), and sudden unexplained emotional Irritability.", "citation": "Endocrine Society Clinical Practice Manual on Insulin Resistance, 2023", "keywords": "itching skin alopecia hair loss irritability mood metabolism stress"}
]

def real_rag_retrieval(symptoms_str):
    documents = [f"{doc['text']} {doc['keywords']}" for doc in CLINICAL_KNOWLEDGE]
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
        query_vector = vectorizer.transform([symptoms_str])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        retrieved = [CLINICAL_KNOWLEDGE[i] for i, score in enumerate(similarities) if score >= 0.01]
        return retrieved if retrieved else CLINICAL_KNOWLEDGE
    except:
        return CLINICAL_KNOWLEDGE

# --- API ENDPOINTS ---

@app.route('/')
def home():
    # সরাসরি ইনডেক্স ফাইলটি সার্ভ করার জন্য (অথবা templates ফোল্ডারে রাখতে পারেন)
    with open("index.html", "r", encoding="utf-8") as f:
        return render_template_string(f.read())

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    name = data.get("name", "Patient")
    lang = data.get("lang", "English")
    responses = data.get("responses", {})
    
    # ML Prediction Prepare
    eval_df = pd.DataFrame([responses])
    for col in eval_df.columns:
        if col != 'Age': eval_df[col] = eval_df[col].astype('category')
        
    binary_pred = model.predict(eval_df)[0] if model else 0
    prob = model.predict_proba(eval_df)[0] if model else [0.5, 0.5]
    
    has_risk = bool(binary_pred == 1 or prob[1] > 0.5)
    conf = prob[1] * 100 if has_risk else prob[0] * 100
    conf_str = f"{conf:.2f}%"
    
    verdict = "DIABETES RISK DETECTED" if has_risk else "NO IMMEDIATE RISK DETECTED"
    if lang == "বাংলা":
        verdict = "ডায়াবেটিস ঝুঁকি সনাক্ত হয়েছে" if has_risk else "কোনো তাৎক্ষণিক ঝুঁকি পাওয়া যায়নি"
        
    # RAG Logic
    pos_symptoms = [k for k, v in responses.items() if v == "Yes"]
    symptoms_str = ", ".join(pos_symptoms) if pos_symptoms else "routine preventive check"
    matched_chunks = real_rag_retrieval(symptoms_str)
    
    context_str = "".join([f"[Source: {c['citation']}]: {c['text']}\n" for c in matched_chunks])
    
    # Groq LLM Call
    try:
        client = Groq(api_key=GROQ_API_KEY)
        system_prompt = f"You are DECat-AI, a digital clinician. Response MUST be in {lang}. Explain risk: {verdict} ({conf_str}). Format clearly with headers: 'Diagnostic Guidance', 'Dietary Action Plan', and 'Lifestyle Protocol'. Append inline citations like (Source: WHO 2023)."
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Patient: {name}\nData: {symptoms_str}\nReferences:\n{context_str}"}],
            temperature=0.3, max_tokens=600
        )
        report = completion.choices[0].message.content
    except Exception as e:
        report = f"Error generating report: {str(e)}"
        
    return jsonify({
        "verdict": verdict,
        "confidence": conf_str,
        "has_risk": has_risk,
        "report": report
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)

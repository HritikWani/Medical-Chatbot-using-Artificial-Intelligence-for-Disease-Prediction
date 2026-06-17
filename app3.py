from flask import Flask, request, render_template, redirect, url_for, session, flash, jsonify
from flask_pymongo import PyMongo
import bcrypt
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
from difflib import get_close_matches
import joblib
import numpy as np
import pandas as pd
import traceback

app = Flask(__name__)
app.secret_key = os.urandom(24)

# MongoDB Configuration
app.config["MONGO_URI"] = "mongodb://localhost:27017/chatbot_members"
mongo = PyMongo(app)

app.config["MODEL_ACCURACIES"] = {
    "Random Forest": 0.7363,
    "Naive Bayes": 0.8497,
    "SVM": 0.8648,
    "XGBoost": 0.8423
}

app.config["BEST_MODEL"] = "SVM"

from groq import Groq
import os, re

client = Groq(api_key="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")

def extract_symptoms(user_text):
    prompt = f"""
    You are a medical symptom extractor.
    From the text below, extract each distinct symptom as an individual item in a Python list.

    IMPORTANT RULES:
    - Return ONLY a valid Python list, nothing else.
    - Do NOT include code blocks, explanations, or variable names.
    - Use lowercase.
    - Use underscores for multi-word symptoms.
    - Split combined phrases like "nauseous, feverish, and headache" into ['nauseous', 'feverish', 'headache'].

    Text: "{user_text}"
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100
        )
    except Groq.BadRequestError as e:
        print(" ERROR:", str(e))
        traceback.print_exc()
        return jsonify({"error": f"Model request failed: {e}"}), 400

    raw = response.choices[0].message.content.strip()
    print(raw)

    import ast

    try:
        symptoms = ast.literal_eval(raw)
        if isinstance(symptoms, list):
            return [s.strip().lower().replace(" ", "_") for s in symptoms]
    except:
        # fallback: split by commas if it's just a string
        if "," in raw:
            return [s.strip().lower().replace(" ", "_") for s in raw.split(",")]

    try:
        # e.g. "['fever', 'headache']"
        symptoms = eval(raw)
        print(symptoms)
        if isinstance(symptoms, list):
            return [s.strip().lower().replace(" ", "_") for s in symptoms]
    except:
        pass
    return []

# -------------------------------
# Load models and preprocessing objects
# -------------------------------
nb_model = joblib.load("naivebayes.pkl")
rf_model = joblib.load("randomforest.pkl")
svm_model = joblib.load("svm.pkl")
xgb_model = joblib.load("xgboost.pkl")

le = joblib.load("label_encoder.pkl")              # disease encoder
symptom_columns = joblib.load("symptom_columns.pkl")  # 378 symptom names

known_symptoms = set(symptom_columns)

from difflib import get_close_matches

def match_symptoms(extracted):
    valid = [s for s in extracted if s in symptom_columns]
    unknown = [s for s in extracted if s not in symptom_columns]

    suggestions = {s: get_close_matches(s, symptom_columns, n=3) for s in unknown}
    return valid, unknown, suggestions

from difflib import get_close_matches

def normalize_symptom(symptom, known_symptoms, cutoff=0.65):
    matches = get_close_matches(symptom, known_symptoms, n=1, cutoff=cutoff)
    return matches[0] if matches else symptom

# -------------------------------
# symptoms -> input vector
# -------------------------------
def symptoms_to_vector(symptoms_list):
    input_vector = np.zeros(len(symptom_columns))
    for symptom in symptoms_list:
        if symptom in symptom_columns:
            idx = list(symptom_columns).index(symptom)
            input_vector[idx] = 1
        else:
            print(f" Warning: '{symptom}' not found in trained symptom list")

    # Return as DataFrame with same feature names
    return pd.DataFrame([input_vector], columns=symptom_columns)

# -------------------------------
# Predict with all models
# -------------------------------
def predict_disease(symptoms_list):
    x_input = symptoms_to_vector(symptoms_list)

    predictions = {}

    # Naive Bayes
    pred_nb = nb_model.predict(x_input)
    predictions["Naive Bayes"] = le.inverse_transform(pred_nb)[0]

    # Random Forest
    pred_rf = rf_model.predict(x_input)
    predictions["Random Forest"] = le.inverse_transform(pred_rf)[0]

    # SVM
    pred_svm = svm_model.predict(x_input)
    predictions["SVM"] = le.inverse_transform(pred_svm)[0]

    # XGBoost
    pred_xgb = xgb_model.predict(x_input)
    predictions["XGBoost"] = le.inverse_transform(pred_xgb)[0]

    return {
        "individual_predictions": predictions,
        "best_model": "XGBoost",
        "best_prediction": predictions["SVM"],
        "ensemble_prediction": predictions["SVM"]
    }

# Routes
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/healthy_habit')
def healthy_habit():
    return render_template('healthy_habit.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password'].encode('utf-8')
        user = mongo.db.users.find_one({'username': username})

        if user and bcrypt.checkpw(password, user['password']):
            session['username'] = username
            flash("Login successful!", "success")
            return redirect(url_for('chatbot'))
        else:
            flash("Invalid credentials. Please try again.", "error")
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = bcrypt.hashpw(request.form['password'].encode('utf-8'), bcrypt.gensalt())

        if mongo.db.users.find_one({'username': username}):
            flash("Username already exists!", "error")
        else:
            mongo.db.users.insert_one({'username': username, 'password': password})
            flash("Account created successfully!", "success")
            return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/chatbot')
def chatbot():
    if 'username' not in session:
        flash("You need to log in first!", "error")
        return redirect(url_for('login'))

    return render_template('chatbot.html')

@app.route('/api/chat', methods=['POST'])
def api_chat():
    if 'username' not in session:
        return jsonify({"error": "Unauthorized", "redirect": url_for('login')}), 401
    
    try:
        data = request.get_json()
        user_text = data.get("text", "")
        history = data.get("history", [])

        if not user_text:
            return jsonify({"error": "No text provided"}), 400
        
        prompt = f"""
        You are a friendly, non-diagnostic medical assistant. 
        Provide helpful, general information, healthy tips, or encouraging conversation.
        related to health, wellness, and common health practices. 
        Do NOT attempt to diagnose a disease or suggest specific treatments.
        Keep your response concise, maximum upto 50 words and polite.
        
        User Query: "{user_text}"
        """
        messages = [{"role": "system", "content": prompt}]

        # add previous conversation
        for msg in history:
            messages.append({
                "role": msg.get("role"),
                "content": msg.get("content")
            })

        # add latest user message
        messages.append({
            "role": "user",
            "content": user_text
        })

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            # messages=[{"role": "user", "content": prompt}],
            messages=messages,
            temperature=0.7,
            max_tokens=60
        )

        bot_reply = response.choices[0].message.content.strip()
        return jsonify({"response": bot_reply})

    except Exception as e:
        print("ERROR in api_chat:", str(e))
        traceback.print_exc()
        return jsonify({"error": "An error occurred during general chat."}), 500
    

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'username' not in session:
        print(" Unauthorized: no username in session")
        return jsonify({"error": "Unauthorized", "redirect": url_for('login')}), 401

    try:
        data = request.get_json()

        if not data or "text" not in data:
            print(" No text in request")
            return jsonify({"error": "No text provided"}), 400

        user_text = data.get("text")
        print("User text:", user_text)

        extracted_symptoms = extract_symptoms(user_text)
        print(" Extracted symptoms:", extracted_symptoms)

        if not extracted_symptoms:
            print(" No symptoms extracted")
            return jsonify({"error":"No symptoms could be extracted"}),400

        # valid, unknown, suggestions =match_symptoms(extracted_symptoms)
        # print(" Matched:", valid, " | Unknown:", unknown, " | Suggestions:", suggestions)

        # normalized_symptoms = [
        #     normalize_symptom(sym, known_symptoms) for sym in extracted_symptoms
        # ]
        # print("normalised: ",normalized_symptoms)

        # if not valid:
        #     print(" No valid symptoms recognised")
        #     return jsonify({
        #         "error":"No valid symptoms recognised",
        #         "unknown":unknown,
        #         "suggestions": suggestions
        #     }),400
        # if not normalized_symptoms:
        #     print(" No normalized symptoms recognised")
        #     return jsonify({
        #         "error":"No normalized symptoms recognised",
        #         "unknown":unknown,
        #         "suggestions": suggestions
        #     }),400

        # 1. Normalize extracted symptoms
        normalized_symptoms = [
            normalize_symptom(sym, known_symptoms)
            for sym in extracted_symptoms
        ]

        # 2. Final symptoms accepted by model
        final_symptoms = list(set(
            s for s in normalized_symptoms if s in known_symptoms
        ))

        # 3. Truly unknown symptoms
        unknown_symptoms = list(set(
            sym for sym in extracted_symptoms
            if normalize_symptom(sym, known_symptoms) not in known_symptoms
        ))

        if not final_symptoms:
            return jsonify({
                "error": "No recognizable symptoms after normalization",
                "unknown_symptoms": unknown_symptoms
            }), 400

        print("normalized:",normalized_symptoms," |final:",final_symptoms," |unknown:",unknown_symptoms)
        #prediction_results = predict_disease(valid)

        prediction_results=predict_disease(final_symptoms)

        # from datetime import datetime

        # prediction_doc = {
        #     "username": session["username"],

        #     "input": {
        #         "raw_text": user_text,
        #         "extracted_symptoms": extracted_symptoms,
        #         "normalized_symptoms": normalized_symptoms,
        #         "final_symptoms": final_symptoms,
        #         "unknown_symptoms": unknown_symptoms
        #     },

        #     "prediction": {
        #         "best_model": prediction_results["best_model"],
        #         "predicted_disease": prediction_results["best_prediction"],
        #         "all_model_predictions": prediction_results["individual_predictions"]
        #     },

        #     "created_at": datetime.utcnow(),

        #     "feedback": {
        #         "doctor_visited": None,
        #         "doctor_diagnosis": None,
        #         "prediction_correct": None,
        #         "cured_symptoms": [],
        #         "current_symptoms": [],
        #         "feedback_given_at": None
        #     }
        # }

        # mongo.db.disease_predictions.insert_one(prediction_doc)


        return jsonify({
            "predictions": prediction_results["individual_predictions"],
            "accuracies": {model: round(acc * 100, 2) 
               for model, acc in app.config["MODEL_ACCURACIES"].items()},
            "best_model": prediction_results["best_model"],
            "best_prediction": prediction_results["best_prediction"],
            "ensemble_prediction": prediction_results["ensemble_prediction"],
            "extracted_symptoms": extracted_symptoms,
            # "valid_symptoms": valid,
            # "unknown_symptoms": unknown,
            # "suggestions": suggestions
            "valid_symptoms": final_symptoms,
            "unknown_symptoms": unknown_symptoms,
            "suggestions": normalized_symptoms

        })

    except Exception as e:
        print("ERROR:", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/doc')
def doctors():
    if 'username' not in session:
        flash("You need to log in first!", "error")
        return redirect(url_for('login'))

    disease = request.args.get('disease')
    if not disease:
        flash("No disease specified", "error")
        return redirect(url_for('chatbot'))

    return render_template('doc.html', disease=disease)

@app.route('/api/doc')
def api_doctors():
    if 'username' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    disease = request.args.get('disease')
    if not disease:
        print("ERROR : No disease specified")
        return jsonify({"error": "No disease specified"}), 400

    try:
        doctors_df = pd.read_csv("Doctor_Versus_Disease.csv")
        matching_doctors = doctors_df[doctors_df['Disease'].str.lower() == disease.lower()]
        
        if matching_doctors.empty:
            return jsonify({"doctors": []})
        
        return jsonify({
            "doctors": matching_doctors.to_dict(orient='records')
        })

    except Exception as e:
        print(" ERROR:", str(e))
        traceback.print_exc()
        return jsonify({"error": f"Error fetching doctors: {str(e)}"}), 500

@app.route('/evaluate')
def evaluate():
    if 'username' not in session:
        flash("You need to log in first!", "error")
        return redirect(url_for('login'))

    accuracies = app.config["MODEL_ACCURACIES"]
    print("accuracies eveluated at call: ",accuracies)

    return render_template('evaluate.html',
                          rf_accuracy=round(accuracies["Random Forest"]*100, 2),
                          nb_accuracy=round(accuracies["Naive Bayes"]*100, 2),
                          svm_accuracy=round(accuracies["SVM"]*100, 2),
                          best_model=app.config["BEST_MODEL"])

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash("You have been logged out.")
    return redirect(url_for('login'))

@app.route('/get_username')
def get_username():
    if 'username' in session:
        return jsonify({"username": session['username']})
    return jsonify({"username": None})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
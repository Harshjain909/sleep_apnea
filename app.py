from flask import Flask, render_template, request, send_file
import joblib, pickle
import numpy as np
import pandas as pd
import io, os

app = Flask(__name__)

# ---------------- LOAD MODEL ----------------
MODEL_DIR = "best_models_nondl"
model = joblib.load(f"{MODEL_DIR}/best_model.joblib")
scaler = pickle.load(open(f"{MODEL_DIR}/scaler.pkl", "rb"))
threshold = float(open(f"{MODEL_DIR}/threshold.txt").read())

# Feature order (must match training)
FEATURES = ["Gender", "BQ", "ESS", "BMI", "Weight", "Height",
            "Head", "Neck", "Waist", "Buttock", "Age"]

@app.route('/')
def home():
    return render_template('index.html')

# ---------------- SINGLE PREDICTION ----------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [
            int(request.form['Gender']),
            int(request.form['BQ']),
            float(request.form['ESS']),
            float(request.form['BMI']),
            float(request.form['Weight']),
            float(request.form['Height']),
            float(request.form['Head']),
            float(request.form['Neck']),
            float(request.form['Waist']),
            float(request.form['Buttock']),
            float(request.form['Age'])
        ]

        scaled = scaler.transform([data])
        prob = model.predict_proba(scaled)[0, 1]
        prediction = 1 if prob >= threshold else 0

        result = {
            "probability": round(prob * 100, 2),
            "risk": "⚠️ High Risk of Sleep Apnea" if prediction else "✅ Low Risk of Sleep Apnea",
            "color": "danger" if prediction else "success"
        }

        return render_template('result.html', result=result)

    except Exception as e:
        return render_template('result.html', result={"error": str(e)})

# ---------------- BATCH UPLOAD ----------------
@app.route('/batch', methods=['POST'])
def batch():
    try:
        file = request.files['file']
        df = pd.read_csv(file)

        if not all(col in df.columns for col in FEATURES):
            missing = [c for c in FEATURES if c not in df.columns]
            return render_template('result.html', result={"error": f"Missing columns: {missing}"})

        scaled = scaler.transform(df[FEATURES])
        probs = model.predict_proba(scaled)[:, 1]
        preds = (probs >= threshold).astype(int)

        df['Apnea_Probability'] = np.round(probs * 100, 2)
        df['Risk'] = np.where(preds == 1, 'High Risk ⚠️', 'Low Risk ✅')

        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name='batch_predictions.csv'
        )
    except Exception as e:
        return render_template('result.html', result={"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)

import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from itertools import zip_longest 

class HeartDiseasePredictor:
    def __init__(self, dataset_path='../datasets/combined_heart_data.csv'):
        # Load and prepare dataset
        self.dataset = pd.read_csv(dataset_path)
        self.prepare_data()
        self.train_models()

    def prepare_data(self):
        # Separate features and target
        self.y = self.dataset['num']
        self.X = self.dataset.drop(['num'], axis=1)

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.33, random_state=42
        )

        # Scale the features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def train_models(self):
        # Train multiple models
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=250, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42),
            'Naive Bayes': GaussianNB(),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'SVM': SVC(random_state=42)
        }

        # Fit models and store performance
        self.model_performance = {}
        for name, model in self.models.items():
            model.fit(self.X_train_scaled, self.y_train)
            y_pred = model.predict(self.X_test_scaled)
            accuracy = accuracy_score(self.y_test, y_pred)
            self.model_performance[name] = round(accuracy * 100, 2)

    def predict(self, input_data, model_name='Random Forest'):
        # Prepare input data
        input_scaled = self.scaler.transform(np.array(input_data).reshape(1, -1))
        
        # Predict using specified model
        model = self.models[model_name]
        prediction = model.predict(input_scaled)
        return int(prediction[0])

    def get_model_details(self):
        return self.model_performance

# Create Flask Application
app = Flask(__name__)

# Initialize Predictor
predictor = HeartDiseasePredictor()

# Feature descriptions for user guidance
FEATURE_DESCRIPTIONS = {
    'age': 'Age of the patient (in years)',
    'sex': '1 = Male, 0 = Female',
    'cp': 'Chest Pain Type (0-3): 0 = Typical Angina, 1 = Atypical Angina, 2 = Non-Anginal Pain, 3 = Asymptomatic',
    'trestbps': 'Resting Blood Pressure (in mm Hg)',
    'chol': 'Serum Cholesterol in mg/dl',
    'fbs': 'Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)',
    'restecg': 'Resting Electrocardiographic Results (0-2)',
    'thalach': 'Maximum Heart Rate Achieved',
    'exang': 'Exercise Induced Angina (1 = yes, 0 = no)',
    'oldpeak': 'ST Depression Induced by Exercise Relative to Rest',
    'slope': 'The Slope of the Peak Exercise ST Segment (0-2)',
    
}

@app.route('/')
def home():
    model_performance = predictor.get_model_details()
    return render_template('index.html', 
                           feature_descriptions=FEATURE_DESCRIPTIONS, 
                           model_performance=model_performance)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input features
        input_features = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            float(request.form['fbs']),
            float(request.form['restecg']),
            float(request.form['thalach']),
            float(request.form['exang']),
            float(request.form['oldpeak']),
            float(request.form['slope'])
          
        ]

        # Selected model
        selected_model = request.form.get('model', 'Random Forest')

        # Make prediction
        prediction = predictor.predict(input_features, selected_model)
        
        # Interpret result
        result = "High Risk of Heart Disease" if prediction == 1 else "Low Risk of Heart Disease"
        
        return render_template('result.html', 
                               prediction=result, 
                               model_used=selected_model,
                               input_features=input_features,
                               feature_descriptions=FEATURE_DESCRIPTIONS)

    except Exception as e:
        return render_template('error.html', error=str(e))

# Create templates directory and HTML files
def create_templates():
    os.makedirs('templates', exist_ok=True)
    
    # index.html
    with open('templates/index.html', 'w') as f:
        f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Heart Health Guardian</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-color: #3498db;
            --secondary-color: #2ecc71;
            --background-color: #f4f6f7;
            --text-color: #2c3e50;
            --card-shadow: 0 10px 25px rgba(0,0,0,0.1);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Poppins', sans-serif;
            background-color: var(--background-color);
            color: var(--text-color);
            line-height: 1.6;
        }

        .container {
            max-width: 1000px;
            margin: 0 auto;
            padding: 2rem;
            background: white;
            box-shadow: var(--card-shadow);
            border-radius: 15px;
        }

        .header {
            text-align: center;
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 2rem;
            border-radius: 15px 15px 0 0;
            margin-bottom: 2rem;
        }

        .header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        .model-performance {
            display: flex;
            justify-content: space-around;
            margin-bottom: 2rem;
            background-color: #f1f4f8;
            padding: 1rem;
            border-radius: 10px;
        }

        .model-performance div {
            text-align: center;
            background-color: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        }

        .feature-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
        }

        .feature-input {
            background-color: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        }

        input, select {
            width: 100%;
            padding: 0.75rem;
            margin-top: 0.5rem;
            border: 1px solid #ddd;
            border-radius: 7px;
            transition: all 0.3s ease;
        }

        input:focus, select:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.2);
        }

        .submit-btn {
            width: 100%;
            padding: 1rem;
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.3s ease;
        }

        .submit-btn:hover {
            transform: scale(1.05);
        }

        .feature-label {
            font-weight: 600;
            color: var(--text-color);
        }

        .feature-description {
            font-size: 0.8rem;
            color: #7f8c8d;
            margin-top: 0.25rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Heart Health Guardian</h1>
            <p>Predict your heart disease risk with advanced machine learning</p>
        </div>

        <div class="model-performance">
            {% for model, score in model_performance.items() %}
            <div>
                <h3>{{ model }}</h3>
                <p>{{ score }}% Accuracy</p>
            </div>
            {% endfor %}
        </div>

        <form action="/predict" method="post">
            <div class="feature-grid">
                {% for feature, description in feature_descriptions.items() %}
                <div class="feature-input">
                    <label class="feature-label" for="{{ feature }}">
                        {{ feature.upper() }}
                        <span class="feature-description">{{ description }}</span>
                    </label>
                    <input type="number" step="0.01" name="{{ feature }}" required>
                </div>
                {% endfor %}
            </div>
            
            <div class="feature-input" style="margin-top: 1rem;">
                <label class="feature-label">Select Prediction Model</label>
                <select name="model">
                    {% for model in model_performance.keys() %}
                        <option value="{{ model }}">{{ model }}</option>
                    {% endfor %}
                </select>
            </div>
            
            <button type="submit" class="submit-btn">Predict Heart Disease Risk</button>
        </form>
    </div>
</body>
</html>
        """)
    
    # result.html
    with open('templates/result.html', 'w') as f:
        f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Heart Health Prediction Result</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-color: #3498db;
            --secondary-color: #2ecc71;
            --background-color: #f4f6f7;
            --text-color: #2c3e50;
            --high-risk-color: #e74c3c;
            --low-risk-color: #2ecc71;
            --card-shadow: 0 10px 25px rgba(0,0,0,0.1);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Poppins', sans-serif;
            background-color: var(--background-color);
            color: var(--text-color);
            line-height: 1.6;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }

        .result-container {
            background: white;
            border-radius: 15px;
            box-shadow: var(--card-shadow);
            padding: 2rem;
            max-width: 700px;
            width: 100%;
        }

        .result-header {
            text-align: center;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
            color: white;
            font-size: 1.5rem;
            font-weight: 600;
            background: {{ 'linear-gradient(135deg, #e74c3c, #c0392b)' if prediction == 'High Risk of Heart Disease' else 'linear-gradient(135deg, #2ecc71, #27ae60)' }};
        }

        .feature-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0 10px;
        }

        .feature-table th, .feature-table td {
            padding: 0.75rem;
            text-align: left;
            background-color: #f1f4f8;
            border-radius: 7px;
        }

        .feature-table th {
            background-color: var(--primary-color);
            color: white;
        }

        .back-btn {
            display: block;
            width: 100%;
            padding: 1rem;
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            text-align: center;
            text-decoration: none;
            border-radius: 10px;
            margin-top: 1.5rem;
            font-weight: 600;
            transition: transform 0.3s ease;
        }

        .back-btn:hover {
            transform: scale(1.05);
        }
    </style>
</head>
<body>
    <div class="result-container">
        <div class="result-header">
            {{ prediction }} (Predicted by {{ model_used }})
        </div>

        <table class="feature-table">
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>Value</th>
                    <th>Description</th>
                </tr>
            </thead>
            <tbody>
                {% for feature, value in (feature_descriptions.keys(), input_features) %}
        <tr>
            <td>{{ feature.upper() }}</td>
            <td>{{ value }}</td>
            <td>{{ feature_descriptions[feature] }}</td>
        </tr>
        {% endfor %}
    </table>

    <br>
    <a href="/">Back to Prediction</a>
</body>
</html>
        """)
    
    # error.html
    with open('templates/error.html', 'w') as f:
        f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Error</title>
</head>
<body>
    <h1>An Error Occurred</h1>
    <p>{{ error }}</p>
    <a href="/">Go Back</a>
</body>
</html>
        """)

# Main execution
if __name__ == '__main__':
    app.run(debug=True)

print("Flask application and templates created successfully!")
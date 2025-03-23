import os
import numpy as np
import pandas as pd
import joblib
import requests
import json
from flask import request, jsonify
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

# Add this to your Flask application

# Add these imports to your Flask applicatio

# Add this to your Flask application routes
# @app.route('/api/chat', methods=['POST'])
# def chat():
#     data = request.json
#     risk_level = data.get('risk_level')
#     user_features = data.get('user_features')
#     conversation_history = data.get('conversation_history', [])
#     user_message = data.get('user_message')
    
#     # Check if message is about irrelevant topics
#     irrelevant_topics = ['weather', 'sports', 'movie', 'politics', 'travel', 'vacation', 
#                         'game', 'music', 'crypto', 'dating', 'social media', 'tv show']
    
#     is_irrelevant = any(topic in user_message.lower() for topic in irrelevant_topics)
    
#     if is_irrelevant:
#         return jsonify({
#             'response': "I'm specifically designed to discuss heart health concerns. Could we focus on your heart health results and recommendations?"
#         })
    
#     # Prepare the system prompt for Ollama
#     system_prompt = f"""
#     You are a Heart Health Assistant chatbot embedded in a medical application.
#     The user has been assessed with a {risk_level} risk of heart disease.
    
#     Their input features are:
#     {json.dumps(user_features, indent=2)}
    
#     IMPORTANT GUIDELINES:
#     1. ONLY provide information related to heart health, cardiovascular issues, and lifestyle recommendations related to heart health.
#     2. If the user asks about topics unrelated to heart health, politely redirect them to heart health topics.
#     3. Do not answer questions about non-heart-related medical conditions, diagnosis, or treatment.
#     4. Keep responses concise (under 150 words) and focused on actionable advice.
#     5. Be empathetic but direct and factual.
    
#     If the user has HIGH RISK:
#     - Emphasize the importance of seeing a cardiologist promptly
#     - Suggest booking a doctor's appointment
#     - Provide urgent but calm guidance
#     - Mention monitoring vital signs
#     - Discuss immediate lifestyle modifications
    
#     If the user has LOW RISK:
#     - Focus on preventive measures and maintaining heart health
#     - Recommend regular check-ups, not urgent medical attention
#     - Suggest proper diet, exercise, and stress management techniques
#     - Emphasize the importance of maintaining current healthy practices
    
#     Respond to the user's most recent message while considering the conversation history.
#     """
    
#     # Format conversation history for Ollama
#     formatted_messages = [{"role": "system", "content": system_prompt}]
    
#     # Add conversation history
#     for message in conversation_history:
#         formatted_messages.append({
#             "role": message["role"],
#             "content": message["content"]
#         })
    
#     # Add current user message
#     formatted_messages.append({"role": "user", "content": user_message})
    
#     try:
#         # Call Ollama API running locally
#         ollama_url = "http://localhost:11434/api/chat"  # Default Ollama API endpoint
        
#         payload = {
#             "model": "llama2",  # Specify your model here (llama2, llama2-uncensored, mistral, etc.)
#             "messages": formatted_messages,
#             "options": {
#                 "temperature": 0.7,
#                 "top_p": 0.9,
#                 "max_tokens": 500,
#             }
#         }
        
#         response = requests.post(ollama_url, json=payload)
#         response.raise_for_status()  # Raise exception for HTTP errors
        
#         result = response.json()
#         assistant_response = result.get('message', {}).get('content', '')
        
#         # Fallback if response is empty
#         if not assistant_response.strip():
#             raise Exception("Empty response from Ollama")
            
#         return jsonify({
#             'response': assistant_response
#         })
        
#     except Exception as e:
#         app.logger.error(f"Ollama API error: {str(e)}")
        
#         # Fallback responses based on risk level
#         if risk_level == "high":
#             fallback = "With your high risk assessment, I recommend consulting a cardiologist soon. In the meantime, monitor your blood pressure, maintain a heart-healthy diet low in sodium and saturated fat, and avoid strenuous activities until cleared by a doctor."
#         else:
#             fallback = "To maintain your good heart health, focus on regular exercise, a balanced diet rich in fruits and vegetables, stress management, and regular check-ups. These habits will help keep your heart disease risk low."
            
#         return jsonify({
#             'response': fallback
#         })
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    risk_level = data.get('risk_level')
    # user_features = data.get('user_features')
    conversation_history = data.get('conversation_history', [])
    user_message = data.get('user_message')

    # List of irrelevant topics
    irrelevant_topics = ['weather', 'sports', 'movie', 'politics', 'travel', 'vacation', 
                         'game', 'music', 'crypto', 'dating', 'social media', 'tv show']

    # If user message contains irrelevant topics, redirect the conversation
    if any(topic in user_message for topic in irrelevant_topics):
        return jsonify({
            'response': "I'm specifically designed to discuss heart health concerns. Could we focus on your heart health results and recommendations?"
        })

    # System prompt to enforce conversation guidelines
    system_prompt = f"""
        You are a Heart Health Assistant chatbot embedded in a medical application.
        The user has been assessed with a {risk_level} risk of heart disease.

        IMPORTANT GUIDELINES:
        1. ONLY discuss heart health, cardiovascular issues, and lifestyle recommendations.
        2. STRICTLY AVOID unrelated topics like celebrities, sports, politics, movies, etc.
        3. If the user asks anything unrelated, reply:  
        "I'm here to assist you with heart health. Let's focus on that."
        4. Responses must be **under 150 words** and provide actionable advice.
        5. If the user has HIGH RISK:
        - Emphasize seeing a cardiologist soon.
        - Suggest lifestyle changes immediately.
        - Be empathetic but direct.

        If the user has LOW RISK:
        - Focus on prevention and maintaining heart health.
        - Recommend diet, exercise, and stress management.
        """ 
            


    # Format conversation history
    formatted_history = [{"role": msg["role"], "content": msg["content"]} for msg in conversation_history]

    # Add system message & user message to history
    messages = [
        {"role": "system", "content": system_prompt},  # 🛑 Enforces topic restrictions
        *formatted_history,  
        {"role": "user", "content": user_message}
    ]

    # Call Ollama API
    try:
        ollama_url = "http://localhost:11434/api/chat"
        payload = {
            "model": "llama2:latest",
            "messages": messages,
            "temperature": 0.7,
            "stream": False
        }
        # print(payload)

        response = requests.post(ollama_url, json=payload)
        response.raise_for_status()
        # print("res:",response)

        result = response.json()
        # print("result:",result)
        raw_response = result.get('message', {}).get('content', '')
        # print("raw response:",raw_response)
        # Ensure process_ollama_response function exists
        processed_response = process_ollama_response(raw_response, risk_level)

        if len(processed_response.strip()) < 10:
            raise ValueError("Response too short or empty")

        return jsonify({'response': processed_response})

    except requests.exceptions.RequestException as e:
        app.logger.error(f"Ollama API error: {str(e)} - Response: {response.text if response else 'No response'}")

        fallback = ("With your high risk assessment, I recommend consulting a cardiologist soon." 
                    " In the meantime, monitor your blood pressure, maintain a heart-healthy diet, "
                    "and avoid strenuous activities.") if risk_level == "high" else \
                   ("To maintain your good heart health, focus on regular exercise, a balanced diet, "
                    "stress management, and regular check-ups.")

        return jsonify({'response': fallback})
def process_ollama_response(response_text, risk_level):
    """
    Process and improve Ollama responses to ensure quality and relevance
    
    Args:
        response_text (str): Raw response from Ollama
        risk_level (str): User's risk level ('high' or 'low')
        
    Returns:
        str: Processed and improved response
    """
    # Remove any standard prefixes that Llama models sometimes add
    prefixes_to_remove = [
        "As an AI assistant,", 
        "As a language model,",
        "I'm an AI language model,",
        "I cannot provide medical advice,",
        "I'm not a medical professional,"
    ]
    
    for prefix in prefixes_to_remove:
        if response_text.startswith(prefix):
            response_text = response_text[len(prefix):].strip()
    print("rw:",response_text)
    # Ensure the response stays on topic about heart health
    if not any(topic in response_text.lower() for topic in [
        "heart", "cardiac", "cardiovascular", "blood pressure", "cholesterol", 
        "exercise", "diet", "stress", "lifestyle", "doctor", "health"
    ]):
        # Add relevant content if the model went off-topic
        if risk_level == "high":
            response_text += " Remember, with your high risk of heart disease, it's important to consult with a doctor soon and focus on heart-healthy habits."
        else:
            response_text += " To maintain your good heart health, continue with heart-healthy lifestyle habits including a balanced diet and regular exercise."
    
    # Remove any disclaimers at the end
    disclaimers = [
        "Please consult with a healthcare professional",
        "This is not medical advice",
        "I'm not a doctor",
        "Always seek professional medical advice"
    ]
    
    for disclaimer in disclaimers:
        if disclaimer in response_text:
            parts = response_text.split(disclaimer)
            response_text = parts[0].strip()
    
    # Ensure the response ends properly without being cut off
    if not response_text.endswith((".", "!", "?")):
        response_text += "."
    print(response_text)
    return response_text

# Add this function to your Flask route
def create_chatbot_prompt(risk_level, conversation_history, current_query):
    """
    Creates an optimized prompt for Ollama based on conversation context
    """
    # Convert features to a readable string
    # feature_str = "\n".join([f"- {k}: {v}" for k, v in user_features.items()])
    
    # Determine prompt focus based on query topic
    query_lower = current_query.lower()
    
    if any(word in query_lower for word in ["doctor", "appointment", "specialist", "cardiologist"]):
        focus = "medical_consultation"
    elif any(word in query_lower for word in ["diet", "food", "eat", "nutrition"]):
        focus = "diet"
    elif any(word in query_lower for word in ["exercise", "activity", "fitness", "workout"]):
        focus = "exercise"
    elif any(word in query_lower for word in ["emergency", "chest pain", "symptoms", "warning"]):
        focus = "emergency"
    else:
        focus = "general"
    
    # Create base prompt with risk-specific guidance
    base_prompt = f"""
    You are a Heart Health Assistant chatbot in a medical application.
    
    USER STATUS:
    - Risk level: {risk_level.upper()} risk of heart disease
    
    
    RESPONSE REQUIREMENTS:
    - ONLY discuss heart health and related lifestyle factors
    - Keep responses under 150 words
    - Be conversational but direct
    - Focus on providing practical advice
    """
    
    # Add focus-specific instructions
    if focus == "medical_consultation":
        if risk_level == "high":
            base_prompt += """
            For this question about medical consultation:
            - Emphasize the URGENCY of seeing a cardiologist (within 1-2 weeks)
            - Mention specific tests they might expect (ECG, stress test, etc.)
            - Suggest preparing questions for their doctor
            """
        else:
            base_prompt += """
            For this question about medical consultation:
            - Recommend regular primary care check-ups (annually)
            - Mention standard screenings (blood pressure, cholesterol)
            - Focus on preventive care
            """
    
    elif focus == "diet":
        if risk_level == "high":
            base_prompt += """
            For this diet question:
            - Emphasize immediate dietary changes needed
            - Focus on low sodium, low saturated fat options
            - Mention DASH or Mediterranean diet specifically
            - Suggest specific foods to avoid and include
            """
        else:
            base_prompt += """
            For this diet question:
            - Recommend heart-healthy eating patterns
            - Suggest balanced nutrition with examples
            - Mention benefits of whole foods
            """
    
    # Truncate conversation history to last 4 exchanges to fit context window
    recent_history = conversation_history[-8:] if len(conversation_history) > 8 else conversation_history
    history_str = "\n".join([f"{'User' if msg['role']=='user' else 'Assistant'}: {msg['content']}" for msg in recent_history])
    
    # Complete the prompt
    full_prompt = f"""
    {base_prompt}
    
    CONVERSATION HISTORY:
    {history_str}
    
    Current user question: {current_query}
    
    Your response (stay focused on heart health, be conversational, and under 150 words):
    """
    
    return full_prompt

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
        
        return render_template('chatbot1.html', 
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

# Main executional
if __name__ == '__main__':
    app.run(debug=True)

print("Flask application and templates created successfully!")
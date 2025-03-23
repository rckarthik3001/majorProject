# Add this to your Flask application

from flask import jsonify


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    risk_level = data.get('risk_level')
    user_features = data.get('user_features')
    conversation_history = data.get('conversation_history')
    user_message = data.get('user_message')
    
    # Check if message is about irrelevant topics
    irrelevant_topics = ['weather', 'sports', 'movie', 'politics', 'travel', 'vacation', 
                        'game', 'music', 'crypto', 'dating', 'social media', 'tv show']
    
    is_irrelevant = any(topic in user_message.lower() for topic in irrelevant_topics)
    
    if is_irrelevant:
        return jsonify({
            'response': "I'm specifically designed to discuss heart health concerns. Could we focus on your heart health results and recommendations?"
        })
    
    # Prepare the prompt for the LLM
    system_prompt = f"""
    You are a Heart Health Assistant chatbot embedded in a medical application.
    The user has been assessed with a {risk_level} risk of heart disease.
    
    Their input features are:
    {json.dumps(user_features, indent=2)}
    
    IMPORTANT GUIDELINES:
    1. ONLY provide information related to heart health, cardiovascular issues, and lifestyle recommendations related to heart health.
    2. If the user asks about topics unrelated to heart health, politely redirect them to heart health topics.
    3. Do not answer questions about non-heart-related medical conditions, diagnosis, or treatment.
    4. Keep responses concise (under 150 words) and focused on actionable advice.
    5. Be empathetic but direct and factual.
    
    If the user has HIGH RISK:
    - Emphasize the importance of seeing a cardiologist promptly
    - Suggest booking a doctor's appointment
    - Provide urgent but calm guidance
    - Mention monitoring vital signs
    - Discuss immediate lifestyle modifications
    
    If the user has LOW RISK:
    - Focus on preventive measures and maintaining heart health
    - Recommend regular check-ups, not urgent medical attention
    - Suggest proper diet, exercise, and stress management techniques
    - Emphasize the importance of maintaining current healthy practices
    
    Respond to the user's most recent message while considering the conversation history.
    """
    
    # Convert conversation history to format expected by LLM API
    formatted_history = []
    for message in conversation_history:
        formatted_history.append({
            "role": message["role"],
            "content": message["content"]
        })
    
    try:
        # Call your LLM API here (example with Anthropic's Claude)
        # This is pseudocode - replace with your actual LLM integration
        import anthropic  # or equivalent library for your chosen LLM
        
        client = anthropic.Client(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        
        response = client.messages.create(
            model="claude-3-sonnet-20240229",  # or your preferred model
            system=system_prompt,
            messages=formatted_history + [{"role": "user", "content": user_message}],
            max_tokens=350
        )
        
        assistant_response = response.content[0].text
        
        return jsonify({
            'response': assistant_response
        })
        
    except Exception as e:
        app.logger.error(f"LLM API error: {str(e)}")
        
        # Fallback responses based on risk level
        if risk_level == "high":
            fallback = "With your high risk assessment, I recommend consulting a cardiologist soon. In the meantime, monitor your blood pressure, maintain a heart-healthy diet low in sodium and saturated fat, and avoid strenuous activities until cleared by a doctor."
        else:
            fallback = "To maintain your good heart health, focus on regular exercise, a balanced diet rich in fruits and vegetables, stress management, and regular check-ups. These habits will help keep your heart disease risk low."
            
        return jsonify({
            'response': fallback
        })
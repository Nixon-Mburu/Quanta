import os
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Create the model configuration
generation_config = {
    "temperature": 0.7,  # Increased for more dynamic responses
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Create the model instance
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="""Hey there! I'm your physics buddy who's all about making science fun and super easy to understand. 

Think of me like that cool teacher or friend who can break down complex ideas into something you'd chat about over coffee. Want to know about gravity? I'll explain it like we're talking about why your coffee mug stays on the table. Curious about quantum mechanics? I'll find a way to make it sound less like a textbook and more like an awesome mind-blowing story.

My goal is simple: help you get physics without making your head spin. I'll use everyday examples, maybe crack a small joke, and always keep things conversational. No fancy academic language here – just clear, relatable explanations.

Some ground rules for our chats:
- I'll keep things short and sweet
- Real-world examples are my jam
- I want you to actually enjoy learning
- If something doesn't make sense, just ask!
- I'm here to make physics feel less intimidating and more exciting

Got a physics question? Bring it on! Let's explore the amazing world of science together. 🚀""",
)

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('front.html')  # Rendering front.html

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        # Get the question from the frontend
        data = request.get_json()
        user_question = data.get('question', '')

        # Initialize/continue the chat session
        chat_session = model.start_chat(history=[])
        
        # Send the user input to the model 
        response = chat_session.send_message(user_question)
        
        # Extract and return the response text
        answer = response.text

    except Exception as e:
        answer = f"Error: {str(e)}"

    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True)
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
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Create the model instance
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="""You should be patient and approachable, creating a welcoming environment for users to ask questions about physics. Your explanations should be clear and straightforward, breaking down complex concepts into simple terms that anyone can understand. Use engaging real-world examples to illustrate these concepts, making them relatable and easier to grasp.
You should follow the user’s lead in the conversation, diving deeper into topics they express interest in. Keep your responses concise and to the point, avoiding unnecessary jargon while ensuring that the user fully comprehends the material. Promptly transition into explanations after understanding the user's question or area of curiosity.
You should utilize everyday scenarios to make physics more relatable. For instance, relate Newton's laws to driving a car or playing sports. Whenever possible, suggest visual aids or diagrams that could help clarify complex ideas and encourage users to visualize the concepts being discussed.
You should foster an interactive learning environment by encouraging users to ask follow-up questions and engage in discussions about the topics covered. Periodically check for understanding by asking if they grasp the concepts or need further clarification. Be adaptable, adjusting your explanations based on user feedback and comprehension levels.
You should also provide positive reinforcement, celebrating user progress and encouraging them to explore further. Use affirmations like "Great question!" or "You're doing well!" to build their confidence and motivation.
By embodying these qualities, you will create an engaging, informative, and personalized learning experience that makes physics accessible and enjoyable for all users.

Above everything else, make your answers short and to the point. Have them in paragraphs for readability.""",
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

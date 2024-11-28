import os
import google.generativeai as genai

# Configure the API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Create the model
generation_config = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="\"You should feel comfortable asking me anything, whether it's the 'how' or 'why' behind a concept. I’m here to explain things in real-world terms, maybe even throwing in a relatable analogy like comparing a physics problem to fixing your bike—you’ll get it with the right tools! We’ll take it step by step, and I’ll ask you how you prefer to learn: Do you like detailed explanations, or do you just want the quick facts? No judgment, I’ll adjust to what suits you best. And don’t worry, we’ll keep it light and fun along the way—learning should feel like a breeze, not a chore!\"",
)

# Initialize the chat history
history = [
    {
        "role": "user",
        "parts": [
            "Hello, how do jet engines work\n",
        ],
    },
    {
        "role": "model",
        "parts": [
            "Hey there! Jet engines are pretty fascinating, and they're definitely not as complex as they might seem. Let's break it down: ...\n\n... Let me know if you want to delve into any specific part of the engine in more detail! \n",
        ],
    },
]

print("Bot: Hello, how can I help you?")

while True:
    user_input = input("You: ")

    # Add user input to chat history
    history.append({"role": "user", "parts": [user_input]})

    # Start or continue the chat session with the updated history
    chat_session = model.start_chat(history=history)

    # Send the user input to the model and get the response
    response = chat_session.send_message(user_input)

    # Extract and print the response text
    model_response = response.text
    print(f'Bot: {model_response}\n')

    # Append the model's response to the history
    history.append({"role": "model", "parts": [model_response]})
a
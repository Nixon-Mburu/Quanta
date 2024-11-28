import os
import google.generativeai as genai
from dotenv import load_dotenv
import time
from sklearn.metrics import precision_score, recall_score, f1_score
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Create the model configuration
generation_config = {
    "temperature": 0.7,  
    "top_p": 0.9,
    "max_output_tokens": 150,
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="""You should be patient and approachable, creating a welcoming environment for users to ask questions about physics. Your explanations should be clear and straightforward, breaking down complex concepts into simple terms that anyone can understand. Use engaging real-world examples to illustrate these concepts, making them relatable and easier to grasp."""
)

# Test questions for evaluation
test_questions = [
    "What is Newton's second law?",
    "Explain the difference between kinetic and potential energy.",
    "How do black holes form?",
    "What is the first law of thermodynamics?",
    "Can you explain Einstein's theory of relativity?",
    "What is the law of conservation of energy?"
]

# Store evaluation metrics
metrics = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1_score": [],
    "perplexity": [],
    "bleu_score": [],
    "rouge_score": [],
    "latency": [],
    "user_satisfaction": [],
    "error_handling": [],
}

def evaluate_model():
    for question in test_questions:
        print(f"Evaluating question: {question}")
        
        # Track latency (response time)
        start_time = time.time()
        
        try:
            # Initialize the chat session
            chat_session = model.start_chat(history=[])
            response = chat_session.send_message(question)
            answer = response.text
            
            # Measure response time (latency)
            latency = time.time() - start_time
            
            print(f"Answer: {answer}")
            
            # Metrics (You can evaluate these manually or use custom scoring)
            accuracy_score = evaluate_accuracy(question, answer)
            precision, recall, f1 = evaluate_precision_recall_f1(question, answer)
            perplexity_score = evaluate_perplexity(answer)
            bleu_score_value = evaluate_bleu_score(answer)
            rouge_score_value = evaluate_rouge_score(answer)
            user_satisfaction_score = get_user_satisfaction(answer)
            error_handling_score = evaluate_error_handling(answer)

            # Store results
            metrics["accuracy"].append(accuracy_score)
            metrics["precision"].append(precision)
            metrics["recall"].append(recall)
            metrics["f1_score"].append(f1)
            metrics["perplexity"].append(perplexity_score)
            metrics["bleu_score"].append(bleu_score_value)
            metrics["rouge_score"].append(rouge_score_value)
            metrics["latency"].append(latency)
            metrics["user_satisfaction"].append(user_satisfaction_score)
            metrics["error_handling"].append(error_handling_score)
        
        except Exception as e:
            print(f"Error with question: {question}. Exception: {str(e)}")
            metrics["error_handling"].append(1)  # If there's an error, consider it a failure.

# Evaluation Functions

# Accuracy: Check if the answer matches predefined correct answers
def evaluate_accuracy(question, answer):
    correct_answers = {
        "What is Newton's second law?": "F = ma",
        "Explain the difference between kinetic and potential energy.": "Kinetic energy is energy of motion, potential energy is energy of position.",
        # Add other correct answers here
    }
    return 1 if answer.strip().lower() in correct_answers.get(question, "").lower() else 0

# Precision, Recall, and F1-score: Can be calculated based on correct/incorrect classifications of responses
def evaluate_precision_recall_f1(question, answer):
    # We will classify answers as "correct" (1) or "incorrect" (0)
    correct_answers = {
        "What is Newton's second law?": "F = ma",
        "Explain the difference between kinetic and potential energy.": "Kinetic energy is energy of motion, potential energy is energy of position.",
        # Other correct answers here
    }
    # If the model's answer is close to the correct answer, we treat it as correct
    y_true = [1 if question in correct_answers else 0]
    y_pred = [1 if question.lower() in answer.lower() else 0]
    
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return precision, recall, f1

# Perplexity: Calculate perplexity based on the model's output (we use a rough estimate for now)
def evaluate_perplexity(answer):
    return len(answer.split())  # A simple proxy for perplexity (longer = less perplexity)

# BLEU Score: Measures the overlap with a reference answer (good for natural language tasks)
def evaluate_bleu_score(answer):
    reference = "F = ma"  # The correct answer to the question about Newton's second law
    candidate = answer.split()
    reference = reference.split()
    
    return sentence_bleu([reference], candidate)

# ROUGE Score: Measures the overlap of n-grams between generated and reference text
def evaluate_rouge_score(answer):
    reference = "F = ma"  # For example, compare answer to a reference answer
    candidate = answer
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    
    return scores['rouge1'].fmeasure

def get_user_satisfaction(answer):
    # User satisfaction could be based on manual feedback, for example
    return 1 if "good" in answer.lower() else 0

def evaluate_error_handling(answer):
    # Check if the answer includes an error message or an indication of failure
    return 1 if "error" not in answer.lower() else 0

def save_results():
    # Save the metrics to a file for future analysis
    import json
    with open('evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("Evaluation metrics saved.")

def main():
    # Evaluate the model
    evaluate_model()

    # Save the results
    save_results()

if __name__ == "__main__":
    main()

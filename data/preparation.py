import pandas as pd
import re
import spacy
from textblob import TextBlob
import contractions

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Load the dataset
df = pd.read_csv('new_tutor_questions.csv')

# Drop missing values and duplicates
df = df.dropna(subset=['Concept', 'Questions'])
df = df.drop_duplicates(subset=['Concept', 'Questions'])

# Clean text function 
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z0-9\s&%\-.]', '', text)
    
    # Fix contractions 
    text = contractions.fix(text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Spacy Text Processing(lemmatization/ stopword removal)
    doc = nlp(text)
    
    # Lemmatize and remove stopwords
    text = ' '.join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

    # Correct spelling 
    text = str(TextBlob(text).correct())
    
    return text

# Display cleaned data sample
print(df.head())

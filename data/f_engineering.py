import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from sklearn.decomposition import LatentDirichletAllocation, NMF
from collections import Counter
import re
import syllapy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load data
df = pd.read_csv('new_tutor_questions.csv')

# Clean text function with NaN check
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply text cleaning
df['question'] = df['Questions'].apply(clean_text)
df['concept'] = df['Concept'].apply(clean_text)

# TF-IDF Vectorization (combined)
vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
X_combined = vectorizer.fit_transform(df['question'] + ' ' + df['concept'])

# Structural Features
df['question_word_count'] = df['question'].apply(lambda x: len(x.split()))
df['concept_word_count'] = df['concept'].apply(lambda x: len(x.split()))
df['question_char_count'] = df['question'].apply(len)
df['concept_char_count'] = df['concept'].apply(len)

df['question_avg_word_length'] = df.apply(lambda row: row['question_char_count'] / row['question_word_count'] if row['question_word_count'] > 0 else 0, axis=1)
df['concept_avg_word_length'] = df.apply(lambda row: row['concept_char_count'] / row['concept_word_count'] if row['concept_word_count'] > 0 else 0, axis=1)

df['question_sentence_count'] = df['question'].apply(lambda x: x.count('.') + 1)
df['concept_sentence_count'] = df['concept'].apply(lambda x: x.count('.') + 1)
df['question_unique_word_count'] = df['question'].apply(lambda x: len(set(x.split())))
df['concept_unique_word_count'] = df['concept'].apply(lambda x: len(set(x.split())))

# Stopword Ratios
stop_words = set(nlp.Defaults.stop_words)
df['question_stopword_ratio'] = df.apply(lambda row: len([word for word in row['question'].split() if word in stop_words]) / row['question_word_count'] if row['question_word_count'] > 0 else 0, axis=1)
df['concept_stopword_ratio'] = df.apply(lambda row: len([word for word in row['concept'].split() if word in stop_words]) / row['concept_word_count'] if row['concept_word_count'] > 0 else 0, axis=1)

# Interaction Feature
df['word_count_ratio'] = df.apply(lambda row: row['question_word_count'] / row['concept_word_count'] if row['concept_word_count'] > 0 else 0, axis=1)

# Sentiment Analysis Function
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

df[['question_polarity', 'question_subjectivity']] = df['question'].apply(lambda x: pd.Series(get_sentiment(x)))
df[['concept_polarity', 'concept_subjectivity']] = df['concept'].apply(lambda x: pd.Series(get_sentiment(x)))

# Named Entity Recognition (NER)
def extract_named_entities(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents], [ent.label_ for ent in doc.ents]

df[['question_entities', 'question_entity_types']] = df['question'].apply(extract_named_entities).apply(pd.Series)
df[['concept_entities', 'concept_entity_types']] = df['concept'].apply(extract_named_entities).apply(pd.Series)

# Count Named Entities by Type
df["num_question_entities"] = df["question_entities"].apply(len)
df["num_concept_entities"] = df["concept_entities"].apply(len)
df["question_person_count"] = df["question_entity_types"].apply(lambda x: x.count('PERSON'))
df["concept_org_count"] = df["concept_entity_types"].apply(lambda x: x.count('ORG'))

# Part-of-Speech (POS) Tagging Function
def pos_tag_features(text):
    doc = nlp(text)
    pos_tags = [token.pos_ for token in doc]
    return Counter(pos_tags)

df['question_pos_tags'] = df['question'].apply(pos_tag_features)
df['concept_pos_tags'] = df['concept'].apply(pos_tag_features)

# Topic Modeling with LDA and NMF
lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
lda_topics = lda_model.fit_transform(X_combined)

for i in range(5):
    df[f'lda_topic_{i+1}_prob'] = lda_topics[:, i]

nmf_model = NMF(n_components=5, random_state=42)
nmf_topics = nmf_model.fit_transform(X_combined)

for i in range(5):
    df[f'nmf_topic_{i+1}_prob'] = nmf_topics[:, i]

# Readability Functions
def syllable_count(word):
    return syllapy.count(word)

def flesch_kincaid(text):
    words = text.split()
    sentences = text.count('.') + 1
    syllables = sum(syllable_count(word) for word in words)
    return (206.835 - (1.015 * (len(words) / sentences)) - (84.6 * (syllables / len(words)))) if len(words) > 0 else 0

def gunning_fog_index(text):
    words = text.split()
    sentences = text.count('.') + 1
    complex_words = sum(1 for word in words if syllable_count(word) >= 3)
    return (0.4 * ((len(words) / sentences) + (100 * complex_words / len(words)))) if len(words) > 0 else 0

def smog_index(text):
    sentences = text.count('.') + 1
    polysyllabic_words = sum(1 for word in text.split() if syllable_count(word) >= 3)
    return (1.0430 * np.sqrt(polysyllabic_words * 30 / sentences) + 3.1291) if sentences > 0 else 0

# Apply readability features
df['question_flesch_kincaid'] = df['question'].apply(flesch_kincaid)
df['concept_flesch_kincaid'] = df['concept'].apply(flesch_kincaid)
df['question_gunning_fog'] = df['question'].apply(gunning_fog_index)
df['concept_gunning_fog'] = df['concept'].apply(gunning_fog_index)
df['question_smog'] = df['question'].apply(smog_index)
df['concept_smog'] = df['concept'].apply(smog_index)

# Lexical Diversity Function
def lexical_diversity(text):
    words = text.split()
    return len(set(words)) / len(words) if len(words) > 0 else 0

# Apply lexical diversity calculation
df['question_lexical_diversity'] = df['question'].apply(lexical_diversity)
df['concept_lexical_diversity'] = df['concept'].apply(lexical_diversity)

# Print selected features and their outputs
features_to_print = [
    'Questions', 'Concept', 
    'num_question_entities', 
    'num_concept_entities', 
    'lda_topic_1_prob', 'nmf_topic_1_prob',
    'question_polarity', 'concept_polarity',
    'question_flesch_kincaid', 'concept_flesch_kincaid'
]

print(df[features_to_print].head())

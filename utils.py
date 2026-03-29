"""
Utility functions for Fake News Detection application.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class InvalidInputError(Exception):
    pass


class ModelNotFoundError(Exception):
    pass


class VectorizerNotFoundError(Exception):
    pass


class TrainingFailedError(Exception):
    pass


class DatabaseError(Exception):
    pass


stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[' + string.punctuation + ']', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)


def validate_input(text):
    if not text:
        return False, "Input cannot be empty."

    text = text.strip()

    if len(text) < 10:
        return False, "Input must be at least 10 characters long."

    if not any(c.isalnum() for c in text):
        return False, "Input must contain letters or numbers."

    return True, ""


def create_vectorizer(max_features=5000, ngram_range=(1, 2)):
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words='english',
        min_df=1,
        max_df=0.95,
        sublinear_tf=True
    )


def format_confidence(score):
    return f"{score * 100:.2f}%"


def get_label_text(label):
    if label == 1 or str(label).lower() == "fake":
        return "FAKE NEWS", "✗", "danger"

    return "REAL NEWS", "✓", "success"


def truncate_text(text, max_length=100):
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."
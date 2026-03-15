# chatbot/preprocessing.py

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(tokens):

    tokens = [t.lower() for t in tokens]

    tokens = [
        t for t in tokens
        if t.isalpha()
    ]

    tokens = [
        t for t in tokens
        if t not in stop_words
    ]

    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
    ]

    return tokens
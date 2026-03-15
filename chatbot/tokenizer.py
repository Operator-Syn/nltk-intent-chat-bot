# chatbot/tokenizer.py

import nltk

def tokenize(text: str):
    return nltk.word_tokenize(text)
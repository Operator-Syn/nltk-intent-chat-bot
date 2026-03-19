# app/__init__.py
import nltk

def download_nltk_resources():
    resources = ['tokenizers/punkt', 'corpora/wordnet', 'corpora/stopwords']
    for resource in resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split('/')[-1])

# Run it once upon import
download_nltk_resources()
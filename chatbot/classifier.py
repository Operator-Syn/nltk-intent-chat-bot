# chatbot/classifier.py

intents = {
    "projects": ["project", "build", "portfolio"],
    "skills": ["skills", "technologies", "stack"],
    "education": ["school", "degree", "study"]
}

def classify(tokens):

    for intent, keywords in intents.items():

        for token in tokens:
            if token in keywords:
                return intent

    return "unknown"
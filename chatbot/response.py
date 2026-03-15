# chatbot/response

import random

responses = {
    "projects": [
        "I have several projects in my portfolio including NLP and software engineering work."
    ],
    "skills": [
        "My skills include Python, NLP, and backend development."
    ],
    "education": [
        "I am currently studying computer science."
    ],
    "unknown": [
        "I’m not sure how to answer that yet."
    ]
}

def generate(intent):

    return random.choice(responses[intent])
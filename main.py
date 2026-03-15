# main.py
from chatbot.tokenizer import tokenize
from chatbot.preprocessing import preprocess
from chatbot.classifier import classify
from chatbot.response import generate

def main():

    print("Chatbot ready. Type 'quit' to exit.")

    try:
        while True:
            text = input("> ")

            if text.lower() == "quit":
                break

            tokens = tokenize(text)
            clean = preprocess(tokens)

            intent = classify(clean)

            reply = generate(intent)

            print(reply)

    except KeyboardInterrupt:
        print("\nExiting chatbot. Goodbye!")


if __name__ == "__main__":
    main()
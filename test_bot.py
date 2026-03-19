# test_bot.py
import json5
import os
from app.core.engine import ChatEngine
from app.core.response import ResponseEngine
from common.api_client import get_full_data_parallel

def main():
    # Load Intents
    intents_path = os.path.join(os.path.dirname(__file__), 'brain/data/intents.jsonc')
    with open(intents_path, 'r') as f:
        intents = json5.load(f).get('intents', [])

    # Load Slang/Normalization map
    slang_path = os.path.join(os.path.dirname(__file__), 'brain/data/slang.jsonc')
    slang_data = {}
    if os.path.exists(slang_path):
        with open(slang_path, 'r') as f:
            slang_data = json5.load(f).get('slang', {})

    engine = ChatEngine(intents, slang_data)
    api_data = get_full_data_parallel().get("profile", [])
    responder = ResponseEngine(api_data)

    print("--- Portfolio Chatbot Ready! (Type 'quit' or Ctrl+C to exit) ---")

    try:
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["quit", "exit"]: break
            
            # Engine returns: processed, intent, proba, runner_up
            # [Debug] language and processed are printed inside predict_intent
            processed, intent, proba, runner_up = engine.predict_intent(user_input)
            
            # Logic: Calculate margin to detect ambiguity
            margin = proba - runner_up[1]
            
            # Debugging
            print(f"[Debug] Best: {intent} ({proba:.2f}) | Margin: {margin:.2f}")

            # Fallback logic: Unknown tag OR low confidence OR too close to runner-up
            if intent == "unknown" or proba < 0.35 or margin < 0.03:
                response = "I'm not quite sure about that. Could you ask me something else?"
            else:
                response = responder.get_response(intent)

            # Translate the English response back to the user's detected language
            final_response = engine.translate_response(response)

            print(f"Bot: {final_response} (Intent: {intent}, Conf: {proba:.2f})")
            
    except KeyboardInterrupt:
        print("\nBot: Goodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
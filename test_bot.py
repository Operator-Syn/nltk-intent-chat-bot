# test_bot.py
import json5
import os
import random
from app.core.engine import ChatEngine
from app.core.response import ResponseEngine
from common.api_client import get_full_data_parallel

def main():
    # Load Intents
    base_dir = os.path.dirname(__file__)
    intents_path = os.path.join(base_dir, 'brain/data/intents.jsonc')
    slang_path = os.path.join(base_dir, 'brain/data/slang.jsonc')
    tl_overrides_path = os.path.join(base_dir, 'brain/data/tl_overrides.jsonc')
    en_overrides_path = os.path.join(base_dir, 'brain/data/en_overrides.jsonc')

    with open(intents_path, 'r') as f:
        intents = json5.load(f).get('intents', [])

    # Load Slang/Normalization map
    slang_data = {}
    if os.path.exists(slang_path):
        with open(slang_path, 'r') as f:
            slang_data = json5.load(f).get('slang', {})
            
    # Load Overrides for the new Stateless Engine constructor
    tl_overrides_data = json5.load(open(tl_overrides_path, 'r')) if os.path.exists(tl_overrides_path) else {}
    en_overrides_data = json5.load(open(en_overrides_path, 'r')) if os.path.exists(en_overrides_path) else {}

    # Initialize Engine with all data parameters
    engine = ChatEngine(intents, slang_data, tl_overrides_data, en_overrides_data)
    api_data = get_full_data_parallel().get("profile", [])
    responder = ResponseEngine(api_data)

    print("--- Portfolio Chatbot Ready! (Type 'quit' or Ctrl+C to exit) ---")

    try:
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["quit", "exit"]: break
            
            # --- STATELESS UNPACKING ---
            # Added 'detected_lang' as the 5th return value
            processed, intent, proba, runner_up, detected_lang = engine.predict_intent(user_input)
            
            # Logic: Calculate margin to detect ambiguity
            margin = proba - runner_up[1]
            
            # Debugging
            print(f"[Debug] L: {detected_lang} | Best: {intent} ({proba:.2f}) | Margin: {margin:.2f}")

            # Fallback logic: Unknown tag OR low confidence OR too close to runner-up
            if intent == "unknown" or proba < 0.35 or margin < 0.03:
                response = "I'm not quite sure about that. Could you ask me something else?"
            else:
                response = responder.get_response(intent)

            # --- STATELESS TRANSLATION ---
            # Explicitly pass the detected language baton
            final_response = engine.translate_response(response, detected_lang)

            print(f"Bot: {final_response} (Intent: {intent}, Conf: {proba:.2f})")
            
    except KeyboardInterrupt:
        print("\nBot: Goodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
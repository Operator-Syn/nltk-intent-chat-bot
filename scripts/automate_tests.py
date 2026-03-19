import json5
import os
import sys
import random
from io import StringIO
import contextlib

# Ensure the root directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.engine import ChatEngine

# ANSI Color Codes for the terminal
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"
BOLD = "\033[1m"

def run_automated_tests():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    intents_path = os.path.join(base_dir, 'brain/data/intents.jsonc')
    slang_path = os.path.join(base_dir, 'brain/data/slang.jsonc')
    markers_path = os.path.join(base_dir, 'brain/data/markers.jsonc')
    guards_path = os.path.join(base_dir, 'brain/data/guards.jsonc')
    tests_path = os.path.join(base_dir, 'scripts/test_cases.jsonc')

    with open(intents_path, 'r') as f:
        intents = json5.load(f).get('intents', [])
    
    slang_data = {}
    if os.path.exists(slang_path):
        with open(slang_path, 'r') as f:
            slang_data = json5.load(f).get('slang', {})

    markers_data = {}
    if os.path.exists(markers_path):
        with open(markers_path, 'r') as f:
            markers_data = json5.load(f)

    guards_data = {}
    if os.path.exists(guards_path):
        with open(guards_path, 'r') as f:
            guards_data = json5.load(f)

    with open(tests_path, 'r') as f:
        test_cases = json5.load(f).get('tests', [])

    engine = ChatEngine(intents, slang_data, markers_data, guards_data)
    
    # Header - Reorganized for better debugging
    print(f"\n{BOLD}{'='*155}{RESET}")
    print(f"{BOLD}{'INPUT':<22} | {'PROCESSED (TRANS)':<22} | {'L':<2} | {'INTENT':<15} | {'CONF':<4} | {'RESPONSE':<30} | {'STATUS'}{RESET}")
    print(f"{'-'*155}")

    passed = 0
    for case in test_cases:
        user_input = case['input']
        expected_intent = case['expected_intent']
        expected_lang = case['expected_lang']
        
        f = StringIO()
        with contextlib.redirect_stdout(f):
            # 'processed' here is the result of translation + cleaning
            processed, intent, proba, runner_up = engine.predict_intent(user_input)
        
        margin = proba - runner_up[1]
        final_intent = intent
        
        # --- TUNED FALLBACK LOGIC ---
        # If confidence is absolute (1.0), we ignore the margin requirement.
        # This prevents identity questions from failing due to slight similarities.
        margin_threshold = 0.03 if proba < 1.0 else 0.0
        
        if intent == "unknown" or proba < 0.35 or margin < margin_threshold:
            final_intent = "unknown"

        # --- GET RESPONSE ---
        raw_response = "I don't understand."
        if final_intent != "unknown":
            matched_intent = next((i for i in intents if i['tag'] == final_intent), None)
            if matched_intent:
                raw_response = random.choice(matched_intent['responses'])
        
        final_response = engine.translate_response(raw_response)

        intent_match = final_intent == expected_intent
        lang_match = engine.user_lang == expected_lang
        is_success = intent_match and lang_match
        
        # Formatting
        lang_display = f"{GREEN}{engine.user_lang:<2}{RESET}" if lang_match else f"{RED}{engine.user_lang:<2}{RESET}"
        intent_display = f"{GREEN}{final_intent:<15}{RESET}" if intent_match else f"{RED}{final_intent:<15}{RESET}"
        
        if is_success:
            passed += 1
            status = f"{GREEN}✅ PASS{RESET}"
        else:
            err = []
            if not intent_match: 
                err.append(f"Exp:{expected_intent}")
                # Debugging: Show who was almost picked
                if runner_up[0]:
                    err.append(f"Runner:{runner_up[0]}({runner_up[1]:.2f})")
            if not lang_match: err.append(f"ExpL:{expected_lang}")
            status = f"{RED}❌ {', '.join(err)}{RESET}"

        # Truncate strings for table view
        proc_view = (processed[:19] + '..') if len(processed) > 22 else processed
        resp_view = (final_response[:27] + '..') if len(final_response) > 30 else final_response

        print(f"{user_input[:22]:<22} | {CYAN}{proc_view:<22}{RESET} | {lang_display} | {intent_display} | {proba:.2f} | {MAGENTA}{resp_view:<30}{RESET} | {status}")

    print(f"{'-'*155}")
    accuracy = (passed/len(test_cases))*100 if len(test_cases) > 0 else 0
    color = GREEN if accuracy > 80 else YELLOW if accuracy > 50 else RED
    print(f"{BOLD}Total: {len(test_cases)} | Passed: {passed} | Accuracy: {color}{accuracy:.1f}%{RESET}")
    print(f"{BOLD}{'='*155}{RESET}\n")

if __name__ == "__main__":
    run_automated_tests()
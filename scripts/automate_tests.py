import json5
import os
import sys
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure the root directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.engine import ChatEngine

# ANSI Color Codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"
BOLD = "\033[1m"

def run_test_case(engine, case, intents):
    user_input = case['input']
    expected_intent = case['expected_intent']
    expected_lang = case['expected_lang']
    
    # Destructure all 6 return values from the updated engine
    processed, intent, proba, runner_up, detected_lang, lang_conf = engine.predict_intent(user_input)
    
    margin = proba - runner_up[1]
    final_intent = intent
    
    margin_threshold = 0.03 if proba < 1.0 else 0.0
    is_same_intent = (intent == runner_up[0])
    
    if intent == "unknown" or proba < 0.40:
        final_intent = "unknown"
    elif not is_same_intent and margin < margin_threshold:
        final_intent = "unknown"

    raw_response = "I don't understand."
    if final_intent != "unknown":
        matched_intent = next((i for i in intents if i['tag'] == final_intent), None)
        if matched_intent:
            raw_response = random.choice(matched_intent['responses'])
    
    final_response = engine.translate_response(raw_response, detected_lang)

    intent_match = final_intent == expected_intent
    lang_match = detected_lang == expected_lang
    
    return {
        "input": user_input,
        "processed": processed,
        "lang": detected_lang,
        "lang_conf": lang_conf,  # Store the language confidence for the table
        "intent": final_intent,
        "proba": proba,
        "response": final_response,
        "is_success": intent_match and lang_match,
        "expected_intent": expected_intent,
        "expected_lang": expected_lang,
        "runner_up": runner_up,
        "lang_match": lang_match,
        "intent_match": intent_match
    }

def run_automated_tests():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    intents_path = os.path.join(base_dir, 'brain/data/intents.jsonc')
    slang_path = os.path.join(base_dir, 'brain/data/slang.jsonc')
    tl_overrides_path = os.path.join(base_dir, 'brain/data/tl_overrides.jsonc')
    en_overrides_path = os.path.join(base_dir, 'brain/data/en_overrides.jsonc')
    tests_cases_path = os.path.join(base_dir, 'scripts/test_cases.jsonc')

    with open(intents_path, 'r') as f:
        intents = json5.load(f).get('intents', [])
    slang_data = json5.load(open(slang_path, 'r')).get('slang', {}) if os.path.exists(slang_path) else {}
    tl_overrides_data = json5.load(open(tl_overrides_path, 'r')) if os.path.exists(tl_overrides_path) else {}
    en_overrides_data = json5.load(open(en_overrides_path, 'r')) if os.path.exists(en_overrides_path) else {}
    
    with open(tests_cases_path, 'r') as f:
        test_cases = json5.load(f).get('tests', [])

    engine = ChatEngine(intents, slang_data, tl_overrides_data, en_overrides_data)
    
    print(f"\n{BOLD}Executing {len(test_cases)} tests...{RESET}")

    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_case = {executor.submit(run_test_case, engine, case, intents): case for case in test_cases}
        for future in as_completed(future_to_case):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"{RED}Thread error: {e}{RESET}")

    # Adjusted table headers to include LC (Language Confidence)
    print(f"\n{BOLD}{'='*165}{RESET}")
    print(f"{BOLD}{'INPUT':<22} | {'PROCESSED (TRANS)':<22} | {'L':<2} | {'LC':<5} | {'INTENT':<15} | {'CONF':<4} | {'RESPONSE':<30} | {'STATUS'}{RESET}")
    print(f"{'-'*165}")

    passed = 0
    for r in sorted(results, key=lambda x: x['input']):
        lang_display = f"{GREEN}{r['lang']:<2}{RESET}" if r['lang_match'] else f"{RED}{r['lang']:<2}{RESET}"
        intent_display = f"{GREEN}{r['intent']:<15}{RESET}" if r['intent_match'] else f"{RED}{r['intent']:<15}{RESET}"
        
        if r['is_success']:
            passed += 1
            status = f"{GREEN}✅ PASS{RESET}"
        else:
            err = [f"Exp:{r['expected_intent']}"] if not r['intent_match'] else []
            if not r['lang_match']: err.append(f"ExpL:{r['expected_lang']}")
            status = f"{RED}❌ {', '.join(err)}{RESET}"

        proc_view = (r['processed'][:19] + '..') if len(r['processed']) > 22 else r['processed']
        resp_view = (r['response'][:27] + '..') if len(r['response']) > 30 else r['response']

        # Format: Added LC column with 2 decimal places
        print(f"{r['input'][:22]:<22} | {CYAN}{proc_view:<22}{RESET} | {lang_display} | {r['lang_conf']:<5.2f} | {intent_display} | {r['proba']:.2f} | {MAGENTA}{resp_view:<30}{RESET} | {status}")

    print(f"{'-'*155}")
    accuracy = (passed/len(test_cases))*100 if test_cases else 0
    color = GREEN if accuracy > 80 else YELLOW
    print(f"{BOLD}Total: {len(test_cases)} | Passed: {passed} | Accuracy: {color}{accuracy:.1f}%{RESET}")
    print(f"{BOLD}{'='*165}{RESET}\n")

if __name__ == "__main__":
    run_automated_tests()
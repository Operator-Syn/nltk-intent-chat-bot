# core/engine.py
import re
import logging
import os
import numpy as np
import langid
from huggingface_hub import login
from transformers import MarianMTModel, MarianTokenizer
from spellchecker import SpellChecker
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.getLogger("transformers").setLevel(logging.ERROR)

class ChatEngine:
    def __init__(self, intents_data, slang_data=None, markers_data=None, guards_data=None):
        # Authenticate with Hugging Face if token is available
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            login(token=hf_token)

        self.lemmatizer = WordNetLemmatizer()
        self.spell = SpellChecker()
        self.slang_map = slang_data or {}
        # Load protected Tagalog markers from markers.jsonc
        self.tl_markers = set(markers_data.get("tl_markers", [])) if markers_data else set()
        
        # Load English overrides from guards.jsonc
        self.en_overrides = set(guards_data.get("en_overrides", [])) if guards_data else set()

        self.user_lang = "en"
        self.last_conf = 0.0

        # Force detection to only consider English and Tagalog
        langid.set_languages(['en', 'tl'])

        # Pre-load MarianMT translation models at startup
        self._translators = {}
        for pair in [("tl", "en"), ("en", "tl")]:
            model_name = f"Helsinki-NLP/opus-mt-{pair[0]}-{pair[1]}"
            self._translators[pair] = {
                "tokenizer": MarianTokenizer.from_pretrained(model_name),
                "model": MarianMTModel.from_pretrained(model_name),
            }

        # Dual-layer vectorizers
        self.char_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5))
        self.word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))

        self.intents = intents_data
        self.all_patterns = []
        self.pattern_tags = []

        for intent in intents_data:
            for pattern in intent["patterns"]:
                # We do NOT autocorrect the KB patterns to keep them as canonical ground truth
                processed = self._preprocess(pattern, autocorrect=False, translate=False)
                self.all_patterns.append(processed)
                self.pattern_tags.append(intent["tag"])

        self.char_matrix = self.char_vectorizer.fit_transform(self.all_patterns)
        self.word_matrix = self.word_vectorizer.fit_transform(self.all_patterns)

    def _get_translation(self, text, from_code, to_code):
        """Translates text using MarianMT — fully offline, no stanza dependency."""
        pair = (from_code, to_code)
        if pair not in self._translators:
            return None
        tokenizer = self._translators[pair]["tokenizer"]
        model = self._translators[pair]["model"]
        tokens = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**tokens)
        return tokenizer.decode(translated[0], skip_special_tokens=True)

    def _preprocess(self, text, autocorrect=True, translate=True):
        # --- LAYER 0: NOISE CLEANING ---
        text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z~]', '', text)
        text = re.sub(r'\^\[+\[[A-Z0-9~]*', '', text)
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s]", "", text)

        # --- LAYER 0.5: ENGLISH GUARD OVERRIDE ---
        # If the phrase is in our guards list, we force English and skip translation logic
        if translate and text in self.en_overrides:
            self.user_lang = "en"
            translate = False

        # --- LAYER 1: OFFLINE DETECTION & TRANSLATION ---
        is_native_english = True
        if translate:
            lang, conf = langid.classify(text)
            
            # Check if any word in the input is a protected Tagalog marker
            has_tl_markers = any(w in self.tl_markers for w in text.split())

            # Trigger translation if detected as TL, has markers, or is low-confidence English
            if lang == 'tl' or has_tl_markers or (lang == 'en' and conf < 2.0):
                self.user_lang = "tl"
                is_native_english = False
                try:
                    translated = self._get_translation(text, "tl", "en")
                    if translated and translated.strip().lower() != text.lower():
                        text = translated
                except Exception as e:
                    print(f"[Translation Error] {e}")
            else:
                self.user_lang = "en"
            
            self.last_conf = conf

        # --- LAYER 2: NORMALIZATION ---
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        words = text.split()

        # --- LAYER 3: SLANG MAPPING (Priority) ---
        current_words = [self.slang_map.get(w, w) for w in words]

        # --- LAYER 4: AUTO-CORRECT (Conditional Fallback) ---
        # Only spellcheck if native English and the word isn't a protected marker
        if autocorrect and is_native_english:
            final_words = []
            for w in current_words:
                if len(w) <= 3 or w in self.tl_markers:
                    final_words.append(w)
                else:
                    final_words.append(self.spell.correction(w) or w)
            current_words = final_words

        # --- LAYER 5: LEMMATIZATION ---
        return " ".join([self.lemmatizer.lemmatize(w) for w in current_words]).strip()

    def translate_response(self, response_text):
        """Translates the English response back to the user's detected language."""
        if self.user_lang == "en":
            return response_text
        try:
            return self._get_translation(response_text, "en", "tl") or response_text
        except Exception as e:
            print(f"[Translation Error] {e}")
            return response_text

    def predict_intent(self, user_text):
        processed = self._preprocess(user_text)

        print(f"[Debug] Detected Language: {self.user_lang} ({self.last_conf:.2f})")
        print(f"[Debug] Processed: {processed}")

        # --- VECTOR ANALYSIS ---
        char_input = self.char_vectorizer.transform([processed])
        word_input = self.word_vectorizer.transform([processed])

        char_sims = cosine_similarity(char_input, self.char_matrix).flatten()
        word_sims = cosine_similarity(word_input, self.word_matrix).flatten()

        # ENSEMBLE: 70% weight on exact word logic, 30% on character similarity
        combined_sims = (word_sims * 0.7) + (char_sims * 0.3)

        if combined_sims.size == 0 or np.max(combined_sims) < 0.01:
            return processed, "unknown", 0.0, ("none", 0.0)

        top_indices = combined_sims.argsort()[::-1]
        idx1 = top_indices[0]
        score1 = combined_sims[idx1]
        intent1 = self.pattern_tags[idx1]

        # Track runner-up to calculate margin (ambiguity check)
        if len(top_indices) > 1:
            idx2 = top_indices[1]
            score2 = combined_sims[idx2]
            intent2 = self.pattern_tags[idx2]
        else:
            score2, intent2 = 0.0, "none"

        return processed, intent1, score1, (intent2, score2)
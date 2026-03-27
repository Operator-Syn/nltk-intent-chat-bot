# core/engine.py
import re
import logging
import os
import numpy as np
import langid
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import login
from transformers import MarianMTModel, MarianTokenizer
from sentence_transformers import SentenceTransformer, util
from spellchecker import SpellChecker
from nltk.stem import WordNetLemmatizer

logging.getLogger("transformers").setLevel(logging.ERROR)


class ChatEngine:
    def __init__(
        self,
        intents_data,
        slang_data=None,
        tl_overrides_data=None,
        en_overrides_data=None,
    ):
        # Authenticate with Hugging Face if token is available
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            try:
                login(token=hf_token)
            except Exception as e:
                logging.warning(f"Hugging Face login failed: {e}")

        self.lemmatizer = WordNetLemmatizer()
        self.spell = SpellChecker()
        self.slang_map = slang_data or {}

        # Load Tagalog overrides
        self.tl_overrides = (
            set(tl_overrides_data.get("tl_overrides", []))
            if tl_overrides_data
            else set()
        )

        # Load English overrides
        self.en_overrides = (
            set(en_overrides_data.get("en_overrides", []))
            if en_overrides_data
            else set()
        )

        # --- PRE-CALCULATE SKELETONS (Fuzzy Matching) ---
        # We pre-calculate these once to avoid doing it on every message
        self.en_skeletons = {re.sub(r"(.)\1+", r"\1", o): o for o in self.en_overrides}
        self.tl_skeletons = {re.sub(r"(.)\1+", r"\1", o): o for o in self.tl_overrides}

        # Force detection to only consider English and Tagalog
        langid.set_languages(["en", "tl"])

        # --- REFINED PARALLEL STARTUP ---
        self._translators = {}
        with ThreadPoolExecutor(max_workers=1) as executor:
            future_semantic = executor.submit(SentenceTransformer, "all-MiniLM-L6-v2")

            for pair in [("tl", "en"), ("en", "tl")]:
                model_name = f"Helsinki-NLP/opus-mt-{pair[0]}-{pair[1]}"
                self._translators[pair] = {
                    "tokenizer": MarianTokenizer.from_pretrained(model_name),
                    "model": MarianMTModel.from_pretrained(model_name),
                }

            self.semantic_model = future_semantic.result()

        self.intents = intents_data
        self.all_patterns = []
        self.pattern_tags = []

        for intent in intents_data:
            for pattern in intent["patterns"]:
                processed, _, _ = self._preprocess(
                    pattern, autocorrect=False, translate=False
                )
                self.all_patterns.append(processed)
                self.pattern_tags.append(intent["tag"])

        self.pattern_embeddings = self.semantic_model.encode(
            self.all_patterns, convert_to_tensor=True
        )

    def _get_translation(self, text, from_code, to_code):
        """Translates text using MarianMT — fully offline."""
        pair = (from_code, to_code)
        if pair not in self._translators:
            return None
        tokenizer = self._translators[pair]["tokenizer"]
        model = self._translators[pair]["model"]
        tokens = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**tokens)
        return tokenizer.decode(translated[0], skip_special_tokens=True)

    def _preprocess(self, text, autocorrect=True, translate=True):
        """Returns tuple of (processed_text, detected_lang, confidence)."""
        current_lang = "en"
        lang_conf = 0.0

        # --- LAYER 0: NOISE CLEANING ---
        text = re.sub(r"\x1b\[[0-9;]*[a-zA-Z~]", "", text)
        text = re.sub(r"\^\[+\[[A-Z0-9~]*", "", text)
        text = text.lower().strip()

        raw_clean = re.sub(r"[^a-z0-9\s]", "", text).strip()
        text = raw_clean

        # --- LAYER 0.2: NORMALIZATION ---
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)

        # --- LAYER 0.5: FUZZY HARD OVERRIDES (Canonical Replacement) ---
        if translate:
            skeleton = re.sub(r"(.)\1+", r"\1", text)

            if skeleton in self.en_skeletons:
                text = self.en_skeletons[skeleton]
                current_lang = "en"
                translate = False
            elif text in self.en_overrides:
                current_lang = "en"
                translate = False
            elif skeleton in self.tl_skeletons:
                text = self.tl_skeletons[skeleton]
                current_lang = "tl"
            elif text in self.tl_overrides:
                current_lang = "tl"

        # --- LAYER 1: OFFLINE DETECTION & TRANSLATION ---
        is_native_english = current_lang == "en"

        if translate:
            lang, conf = langid.classify(text)
            lang_conf = conf
            has_tl_override = any(w in self.tl_overrides for w in text.split())

            if lang == "tl" or has_tl_override or (lang == "en" and abs(conf) < 2.0):
                current_lang = "tl"
                is_native_english = False
                try:
                    translated = self._get_translation(text, "tl", "en")
                    if translated and translated.strip().lower() != text.lower():
                        text = translated
                except Exception as e:
                    print(f"[Translation Error] {e}")
            else:
                current_lang = "en"
                is_native_english = True

        # --- LAYER 2: TOKENIZATION ---
        words = text.split()

        # --- LAYER 3: SLANG MAPPING ---
        current_words = [self.slang_map.get(w, w) for w in words]

        # --- LAYER 4: AUTO-CORRECT (Conditional) ---
        if autocorrect and is_native_english:
            final_words = []
            for w in current_words:
                # Skip short words or Tagalog overrides to prevent false corrections
                if len(w) <= 3 or w in self.tl_overrides:
                    final_words.append(w)
                else:
                    final_words.append(self.spell.correction(w) or w)
            current_words = final_words

        # --- LAYER 5: LEMMATIZATION ---
        processed_text = " ".join(
            [self.lemmatizer.lemmatize(w) for w in current_words]
        ).strip()

        return processed_text, current_lang, lang_conf

    def translate_response(self, response_text, user_lang):
        """Translates the English response back based on the provided user_lang."""
        if user_lang == "en":
            return response_text
        try:
            return self._get_translation(response_text, "en", "tl") or response_text
        except Exception as e:
            print(f"[Translation Error] {e}")
            return response_text

    def predict_intent(self, user_text):
        """Returns (processed, intent, confidence, runner_up, detected_lang, lang_conf)."""
        processed, detected_lang, lang_conf = self._preprocess(user_text)

        # --- SEMANTIC VECTOR ANALYSIS ---
        user_embedding = self.semantic_model.encode(processed, convert_to_tensor=True)
        cosine_scores = util.cos_sim(user_embedding, self.pattern_embeddings)[0]
        combined_sims = cosine_scores.cpu().numpy()

        if combined_sims.size == 0 or np.max(combined_sims) < 0.2:
            return processed, "unknown", 0.0, ("none", 0.0), detected_lang, lang_conf

        top_indices = combined_sims.argsort()[::-1]
        idx1 = top_indices[0]
        score1 = combined_sims[idx1]
        intent1 = self.pattern_tags[idx1]

        if len(top_indices) > 1:
            idx2 = top_indices[1]
            score2 = combined_sims[idx2]
            intent2 = self.pattern_tags[idx2]
        else:
            score2, intent2 = 0.0, "none"

        return (
            processed,
            intent1,
            float(score1),
            (intent2, float(score2)),
            detected_lang,
            lang_conf,
        )

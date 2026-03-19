# app/core/response.py

import random
import json5
import os
import re

class ResponseEngine:
    def __init__(self, api_data):
        self.api_data = api_data
        data_path = os.path.join(os.path.dirname(__file__), '../../brain/data/intents.jsonc')
        with open(data_path, 'r') as f:
            self.intents = json5.load(f)['intents']

    def _get_value_by_label(self, label):
        # Normalize the label for search (e.g., "Year Level" -> "Year")
        for item in self.api_data:
            if item.get('label') == label:
                return item.get('value')
        return None

    def get_response(self, intent_tag):
        intent_obj = next((i for i in self.intents if i['tag'] == intent_tag), None)
        
        if not intent_obj or intent_tag == "unknown":
            return "I'm not sure about that. Could you ask me something else about the portfolio?"

        template = random.choice(intent_obj['responses'])

        # 1. Use Regex to find ALL placeholders like {Something}
        placeholders = re.findall(r'\{(.*?)\}', template)
        
        # 2. Automatically replace each placeholder found
        for p in placeholders:
            # We assume your API label is "Program" and your template uses {Program}
            val = self._get_value_by_label(p)
            if val:
                template = template.replace(f"{{{p}}}", val)
            else:
                # Fallback if no matching label is found in API
                template = template.replace(f"{{{p}}}", "unknown information")
        
        return template
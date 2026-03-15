# chatbot/api.py

import os
import requests
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv()
BASE_API = os.getenv("API_URL")

# Persistent session for connection pooling
session = requests.Session()

def _get_json(url):
    """Internal helper to fetch JSON with a timeout and shared session."""
    if not url or "None" in url:
        print("Error: API_URL is not set in environment.")
        return []
        
    try:
        resp = session.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"API call failed for {url}: {e}")
    return []

def get_profile():
    """
    Fetch profile information.
    Returns a list of dicts (id, label, value, display_order).
    """
    return _get_json(f"{BASE_API}/profile")

def get_full_data_parallel():
    """
    Example of how to maintain parallelism if you add more calls later.
    Currently only fetches profile but is structured for multi-tasking.
    """
    tasks = {
        "profile": f"{BASE_API}/profile",
        # Add more endpoints here later: "projects": f"{BASE_API}/projects"
    }

    results = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_key = {executor.submit(_get_json, url): key for key, url in tasks.items()}
        
        for future in future_to_key:
            key = future_to_key[future]
            results[key] = future.result()

    return results
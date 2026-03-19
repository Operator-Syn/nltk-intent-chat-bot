import os
import requests
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv()
BASE_API = os.getenv("API_URL")

# Persistent session for connection pooling
session = requests.Session()
_cache = {}

def _get_json(url):
    """Internal helper to fetch JSON with a timeout, shared session, and caching."""
    
    if not url or "None" in url:
        print("Error: API_URL is not set in environment.")
        return []

    # --- Cache check ---
    if url in _cache:
        return _cache[url]

    try:
        resp = session.get(url, timeout=10)

        if resp.status_code == 200:
            data = resp.json()

            # Store in cache
            _cache[url] = data

            return data

    except Exception as e:
        print(f"API call failed for {url}: {e}")

    return []

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

        future_to_key = {
            executor.submit(_get_json, url): key
            for key, url in tasks.items()
        }

        for future in future_to_key:
            key = future_to_key[future]
            results[key] = future.result()

    return results
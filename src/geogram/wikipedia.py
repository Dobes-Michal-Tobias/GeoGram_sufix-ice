"""Wikipedia helpers for Czech municipality description analysis."""

import requests


def fetch_wikipedia_intro(title: str, lang: str = "cs") -> str:
    """Fetch the first sentence of a Wikipedia page via the API."""
    endpoint = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "titles": title,
        "format": "json",
    }
    response = requests.get(endpoint, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        return page.get("extract", "")
    return ""

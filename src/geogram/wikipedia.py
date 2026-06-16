"""Wikipedia helpers for Czech municipality grammar analysis."""

import re

import requests


SINGULAR = "singular"
PLURAL = "plural"
BOTH = "both"
UNKNOWN = "unknown"
NOT_FOUND = "not_found"

_API = "https://cs.wikipedia.org/w/api.php"
_UA = "GeoGram/1.0 (research; +https://github.com/yourname/GeoGram_sufix-ice)"


# ---------------------------------------------------------------------------
# Low-level Wikipedia API helpers
# ---------------------------------------------------------------------------

def _query(titles: str) -> dict:
    """Return the pages dict from a Wikipedia API titles query."""
    params = {
        "action": "query",
        "prop": "extracts|pageprops",
        "exintro": True,
        "explaintext": True,
        "redirects": True,
        "titles": titles,
        "format": "json",
    }
    r = requests.get(_API, params=params, headers={"User-Agent": _UA}, timeout=(5, 15))
    r.raise_for_status()
    return r.json().get("query", {}).get("pages", {})


def _search(query: str, limit: int = 3) -> list[dict]:
    """Return top search results (list of {title, snippet})."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "format": "json",
    }
    r = requests.get(_API, params=params, headers={"User-Agent": _UA}, timeout=(5, 15))
    r.raise_for_status()
    return r.json().get("query", {}).get("search", [])


def _page_status(page: dict) -> str:
    """Classify a page dict as 'found', 'disambiguation', or 'missing'."""
    if page.get("pageid", -1) < 0 or "missing" in page:
        return "missing"
    if "disambiguation" in page.get("pageprops", {}):
        return "disambiguation"
    return "found"


def _extract(page: dict) -> str:
    return page.get("extract", "") or ""


# ---------------------------------------------------------------------------
# Municipality resolution with disambiguation handling
# ---------------------------------------------------------------------------

def resolve_municipality(
    name: str,
    district_name: str = "",
    region_name: str = "",
) -> tuple[str, str]:
    """Find the Wikipedia intro for a Czech municipality.

    Tries three strategies in order:
    1. Exact title match (e.g. "Mohelnice")
    2. Title with district qualifier (e.g. "Bystřice (okres Benešov)")
    3. Full-text search fallback (e.g. "Bystřice obec Benešov")

    Returns (intro_text, resolution_status) where resolution_status is one of:
      'found'                – exact title matched directly
      'resolved_with_okres'  – disambiguation resolved via "(okres X)" qualifier
      'resolved_with_search' – found via search API
      'missing'              – not found by any strategy
    """
    # 1. Exact title
    pages = _query(name)
    page = next(iter(pages.values()))
    if _page_status(page) == "found":
        return _extract(page), "found"

    # 2. Disambiguated title using district name
    if district_name:
        title_okres = f"{name} (okres {district_name})"
        pages2 = _query(title_okres)
        page2 = next(iter(pages2.values()))
        if _page_status(page2) == "found":
            return _extract(page2), "resolved_with_okres"

    # 3. Search API fallback
    query = f"{name} obec {district_name}".strip()
    results = _search(query)
    for hit in results:
        pages3 = _query(hit["title"])
        page3 = next(iter(pages3.values()))
        if _page_status(page3) == "found":
            intro = _extract(page3)
            # Sanity-check: the intro should mention the municipality name
            if name.lower() in intro.lower():
                return intro, "resolved_with_search"

    return "", "missing"


# ---------------------------------------------------------------------------
# Grammar number extraction
# ---------------------------------------------------------------------------

def extract_grammar_number(intro: str, name: str) -> str:
    """Determine grammatical number from a Czech Wikipedia intro.

    Strategy (in order of confidence):
    1. Explicit keyword "pomnožné" in intro → plural
    2. Verb agreement directly after the name in the first ~300 chars:
       "X jsou ..." → plural, "X je ..." → singular
    3. Any "jsou"/"je" verb in the first sentence as fallback
    """
    if not intro:
        return UNKNOWN

    # 1. Explicit plurale tantum marker
    if "pomnožné" in intro.lower():
        return PLURAL

    window = intro[:400]
    name_esc = re.escape(name)

    # 2. Verb immediately following the name (with optional short clause in between)
    if re.search(rf'\b{name_esc}\b[^.{{}}]{{0,80}}\bjsou\b', window, re.IGNORECASE):
        return PLURAL
    if re.search(rf'\b{name_esc}\b[^.{{}}]{{0,80}}\bje\b', window, re.IGNORECASE):
        return SINGULAR

    # 3. First-sentence fallback (catches "X, město v …, jsou/je …" constructions)
    first_sent = re.split(r'[.\n]', intro)[0]
    if re.search(r'\bjsou\b', first_sent):
        return PLURAL
    if re.search(r'\bje\b', first_sent):
        return SINGULAR

    return UNKNOWN


# ---------------------------------------------------------------------------
# Public top-level function
# ---------------------------------------------------------------------------

def fetch_municipality_number(
    name: str,
    district_name: str = "",
    region_name: str = "",
) -> tuple[str, str]:
    """Full pipeline: resolve Wikipedia article and extract grammar number.

    Returns (grammar_number, resolution_status).
    grammar_number: singular | plural | both | unknown | not_found
    resolution_status: found | resolved_with_okres | resolved_with_search | missing
    """
    intro, status = resolve_municipality(name, district_name, region_name)
    if status == "missing":
        return NOT_FOUND, status
    number = extract_grammar_number(intro, name)
    return number, status


# ---------------------------------------------------------------------------
# Legacy helper kept for backwards compatibility
# ---------------------------------------------------------------------------

def fetch_wikipedia_intro(title: str, lang: str = "cs") -> str:
    """Fetch the intro text of a Wikipedia page by exact title."""
    pages = _query(title)
    page = next(iter(pages.values()))
    return _extract(page)

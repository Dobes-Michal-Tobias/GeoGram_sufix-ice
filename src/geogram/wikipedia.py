"""Wikipedia helpers for Czech municipality grammar analysis."""

import re
import time

import requests


SINGULAR = "singular"
PLURAL = "plural"
BOTH = "both"
UNKNOWN = "unknown"
NOT_FOUND = "not_found"

_API = "https://cs.wikipedia.org/w/api.php"
_UA = "GeoGram/1.0 (research; +https://github.com/yourname/GeoGram_sufix-ice)"
_RETRIES = 3
_BACKOFF = 1.5


# ---------------------------------------------------------------------------
# Low-level Wikipedia API helpers
# ---------------------------------------------------------------------------

def _get(params: dict) -> dict:
    """GET against the Wikipedia API with retries on transient network errors."""
    last_error = None
    for attempt in range(_RETRIES):
        try:
            r = requests.get(_API, params=params, headers={"User-Agent": _UA}, timeout=(5, 15))
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < _RETRIES - 1:
                time.sleep(_BACKOFF * (attempt + 1))
    raise last_error


def _query(titles: str, exintro: bool = True) -> dict:
    """Return the pages dict from a Wikipedia API titles query."""
    params = {
        "action": "query",
        "prop": "extracts|pageprops",
        "exintro": exintro,
        "explaintext": True,
        "redirects": True,
        "titles": titles,
        "format": "json",
    }
    return _get(params).get("query", {}).get("pages", {})


def _search(query: str, limit: int = 3) -> list[dict]:
    """Return top search results (list of {title, snippet})."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "format": "json",
    }
    return _get(params).get("query", {}).get("search", [])


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
    full_text: bool = False,
) -> tuple[str, str]:
    """Find the Wikipedia article text for a Czech municipality.

    Tries three strategies in order:
    1. Exact title match (e.g. "Mohelnice")
    2. Title with district qualifier (e.g. "Bystřice (okres Benešov)")
    3. Full-text search fallback (e.g. "Bystřice obec Benešov")

    By default returns only the intro paragraph. Pass full_text=True to fetch
    the entire article body instead (needed for the locative-case fallback,
    since the boilerplate intro sentence "Obec X se nachází..." never uses it).

    Returns (text, resolution_status) where resolution_status is one of:
      'found'                – exact title matched directly
      'resolved_with_okres'  – disambiguation resolved via "(okres X)" qualifier
      'resolved_with_search' – found via search API
      'missing'              – not found by any strategy
    """
    exintro = not full_text

    # 1. Exact title
    pages = _query(name, exintro=exintro)
    page = next(iter(pages.values()))
    if _page_status(page) == "found":
        return _extract(page), "found"

    # 2. Disambiguated title using district name
    if district_name:
        title_okres = f"{name} (okres {district_name})"
        pages2 = _query(title_okres, exintro=exintro)
        page2 = next(iter(pages2.values()))
        if _page_status(page2) == "found":
            return _extract(page2), "resolved_with_okres"

    # 3. Search API fallback
    query = f"{name} obec {district_name}".strip()
    results = _search(query)
    for hit in results:
        pages3 = _query(hit["title"], exintro=exintro)
        page3 = next(iter(pages3.values()))
        if _page_status(page3) == "found":
            text = _extract(page3)
            # Sanity-check: the text should mention the municipality name
            if name.lower() in text.lower():
                return text, "resolved_with_search"

    return "", "missing"


# ---------------------------------------------------------------------------
# Grammar number extraction
# ---------------------------------------------------------------------------

# Hard adjectives whose -é form is unambiguously inanimate-plural when preceding
# an -ice name (feminine plural).  Soft adjectives like horní/dolní are invariant
# across sg/pl and therefore excluded.
_ADJ_PLURAL_ICE = re.compile(
    r"^(velk[eé]|nov[eé]|star[eé]|česk[eé]|mal[eé]|b[ií]l[eé]|čern[eé]|"
    r"červen[eé]|zelen[eé]|svat[eé]|zlaté|modr[eé])\s",
    re.IGNORECASE,
)


def name_implies_plural(name: str) -> bool:
    """Return True if the municipality name itself implies plural grammatical number.

    Specifically: a hard adjective in its inanimate-plural -é form preceding
    an -ice name unambiguously signals plural (e.g. "Velké Popovice").
    Soft adjectives (horní, dolní, …) are invariant across sg/pl and excluded.
    """
    return bool(_ADJ_PLURAL_ICE.match(name)) and name.rstrip().endswith("ice")


def extract_grammar_number(intro: str, name: str) -> str:
    """Determine grammatical number from a Czech Wikipedia intro.

    Strategy (in order of confidence):
    0. Name morphology: hard adjective -é + -ice ending → plural
    1. Explicit keyword "pomnožn*" in intro → plural
    2. jsou/je + municipality type word anywhere in first 600 chars
    3. "se nacházejí" anywhere in first 600 chars → plural
       ("se nachází" is excluded: "Obec X se nachází" has obec as subject,
        not the municipality name, so it is ambiguous)
    4. Name followed by jsou/je within 150 chars
    5. First two sentences fallback
    """
    # 0. Name morphology (independent of intro text)
    if name_implies_plural(name):
        return PLURAL

    if not intro:
        return UNKNOWN

    # 1. Explicit plurale tantum marker ("pomnožné jméno", "název pomnožný", …)
    if "pomnožn" in intro.lower():
        return PLURAL

    window = intro[:600]
    name_esc = re.escape(name)
    _MUNI = r"(?:obec|město|vesnice|osada|městys|sídlo|statutární\s+město|část\s+obce)"

    # 2. jsou/je immediately before a municipality type word (high confidence)
    if re.search(rf'\bjsou\b\s+{_MUNI}', window, re.IGNORECASE):
        return PLURAL
    if re.search(rf'\bje\b\s+{_MUNI}', window, re.IGNORECASE):
        return SINGULAR

    # 3. "se nacházejí" / "nacházejí se" → definitely plural subject
    if re.search(r'\bnach[aá]zej[íi]\b', window, re.IGNORECASE):
        return PLURAL

    # 4. Name followed by jsou/je (expanded window to 150 chars)
    if re.search(rf'\b{name_esc}\b[^.{{}}]{{0,150}}\bjsou\b', window, re.IGNORECASE):
        return PLURAL
    if re.search(rf'\b{name_esc}\b[^.{{}}]{{0,150}}\bje\b', window, re.IGNORECASE):
        return SINGULAR

    # 5. First two sentences fallback
    sentences = re.split(r'[.\n]', intro)[:2]
    for sent in sentences:
        if re.search(r'\bjsou\b', sent):
            return PLURAL
        if re.search(r'\bje\b', sent):
            return SINGULAR

    return UNKNOWN


def extract_grammar_number_locative(text: str, name: str) -> str:
    """Fallback: infer grammatical number from Czech locative case forms.

    Feminine -ice names decline differently depending on their number:
    plural forms take locative "-icích" (e.g. "v Kunčicích"), singular forms
    take locative "-ici" (e.g. "v Bystřici"). This only shows up in body text
    (phrases like "narozen v Chroustovicích", "škola v Pticích"), never in the
    boilerplate intro sentence ("Obec X se nachází...", where "obec" is always
    singular regardless of X) — so callers should pass the full article text,
    not just the intro, and use this only when the intro-based extractor
    returns UNKNOWN.

    Returns UNKNOWN if neither form is attested, or if both are (ambiguous —
    e.g. a village with a differently-numbered part, like "Ptice"/"Horní Ptici").
    """
    if not text or not name.endswith("ice"):
        return UNKNOWN

    stem = re.escape(name[:-3])
    has_plural = bool(re.search(rf"{stem}ic[ií]ch\b", text, re.IGNORECASE))
    has_singular = bool(re.search(rf"\b{stem}ici\b", text, re.IGNORECASE))

    if has_plural and not has_singular:
        return PLURAL
    if has_singular and not has_plural:
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

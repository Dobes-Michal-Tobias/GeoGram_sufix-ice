"""Utilities for querying IJP ÚJČ and parsing Czech morphological number data."""

import re
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup


SINGULAR = "singular"
PLURAL = "plural"
BOTH = "both"
UNKNOWN = "unknown"


def build_ujc_url(term: str) -> str:
    """Build the IJP ÚJČ search URL for a given term."""
    encoded_term = quote_plus(term, encoding="utf-8")
    return f"https://prirucka.ujc.cas.cz/?slovo={encoded_term}"


def fetch_ujc_entry(term: str) -> str:
    """Fetch the IJP ÚJČ page HTML for a given term."""
    url = build_ujc_url(term)
    headers = {
        "User-Agent": "GeoGram/1.0 (+https://github.com/yourname/GeoGram_sufix-ice)"
    }
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    return response.text


def _normalize_cell_text(text: str) -> str:
    if text is None:
        return ""
    normalized = re.sub(r"\d+", "", text)
    return normalized.strip()


def parse_number_from_ujc(html: str) -> str:
    """Parse singular/plural information from the IJP ÚJČ entry HTML."""
    soup = BeautifulSoup(html, "lxml-xml")
    table = soup.find("table", {"class": "para"})
    if table is None:
        return UNKNOWN

    rows = table.find_all("tr")
    if len(rows) < 2:
        return UNKNOWN

    header_cells = [cell.get_text(strip=True).lower() for cell in rows[0].find_all(["th", "td"])]
    try:
        singular_index = next(i for i, cell in enumerate(header_cells) if "jednotn" in cell)
        plural_index = next(i for i, cell in enumerate(header_cells) if "množn" in cell or "mno�n" in cell)
    except StopIteration:
        singular_index = 1
        plural_index = 2

    singular_values = []
    plural_values = []
    for row in rows[1:]:
        cells = row.find_all(["th", "td"])
        if len(cells) <= max(singular_index, plural_index):
            continue
        singular_values.append(_normalize_cell_text(cells[singular_index].get_text()))
        plural_values.append(_normalize_cell_text(cells[plural_index].get_text()))

    has_singular = any(bool(value) for value in singular_values)
    has_plural = any(bool(value) for value in plural_values)

    if has_singular and not has_plural:
        return SINGULAR
    if has_plural and not has_singular:
        return PLURAL
    if has_singular and has_plural:
        return BOTH
    return UNKNOWN


def fetch_ujc_number(term: str) -> str:
    """Fetch the IJP ÚJČ entry for a term and return singular/plural classification."""
    html = fetch_ujc_entry(term)
    return parse_number_from_ujc(html)

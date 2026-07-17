"""Morfologická klasifikace sufixu obcí na -ice.

Datově řízená kategorizace (viz notebook 11): tři skupiny podle koncovky
jména, motivované produktivními vzory české onomastiky, ne libovolný
string-matching.
"""

from __future__ import annotations

import pandas as pd

SUFFIX_ORDER = ["ovice", "nice", "bare_ice"]
SUFFIX_TITLES = {
    "ovice": "-ovice",
    "nice": "-nice",
    "bare_ice": "holé -ice",
}


def classify_suffix(name: str) -> str:
    """Zařadí jméno obce do jedné ze 3 sufixálních skupin dle koncovky.

    `-ovice` (patronymický formant -ov- + -ice, např. Kunovice — "ves lidí
    Kunových") je nejčastější a samostatně dobře doložený produktivní vzor
    české onomastiky. `-nice` pokrývá zbylé varianty na -nice (-inice,
    -enice, -anice, ...), jednotlivě příliš vzácné na samostatnou skupinu
    (viz notebook 11 sekce 2 pro rozpad četností). Zbytek je "holé -ice" —
    souhláska přímo před -ice, heterogenní zbytková kategorie.
    """
    if name.endswith("ovice"):
        return "ovice"
    if name.endswith("nice"):
        return "nice"
    return "bare_ice"


def add_suffix_column(df: pd.DataFrame, name_col: str = "name") -> pd.DataFrame:
    """Přidá sloupec `suffix_group` dle `classify_suffix()`."""
    out = df.copy()
    out["suffix_group"] = out[name_col].map(classify_suffix)
    return out


def suffix_ending(name: str, n: int = 5) -> str:
    """Posledních n písmen jména — pro jemnější popisnou analýzu (ne formální test,
    skupiny jsou příliš malé a početné na spolehlivý chi²)."""
    return name[-n:] if len(name) >= n else name

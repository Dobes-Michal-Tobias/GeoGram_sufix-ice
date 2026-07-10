"""Konfigurace cest a vizualizačního stylu pro GeoGram."""

from __future__ import annotations

from pathlib import Path

# --- Cesty -------------------------------------------------------------------

BASE_DIR      = Path(__file__).resolve().parent.parent.parent
DATA_DIR      = BASE_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Statické grafy (PNG) pro publikaci na michaltobiasdobes.cz
ASSETS_IMG_DIR = BASE_DIR / "assets" / "img" / "geogram"


def ensure_dirs() -> None:
    """Vytvoří datové a výstupní adresáře, pokud neexistují."""
    for d in [PROCESSED_DIR, ASSETS_IMG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# --- Vizualizační styl ---------------------------------------------------------

STYLE       = "whitegrid"
FONT_SCALE  = 1.05
PALETTE     = "muted"

# Datová paleta projektu — 3 role, ne libovolné barvy (sdíleno napříč projekty,
# stejné hodnoty jako v PeriodSim/src/config.py):
#   PRIMARY  = hlavní/většinová datová řada (90 % grafů)
#   ACCENT   = výhradně problém/upozornění (neshoda zdrojů, nerozpoznané záznamy,
#              práh) — stejná terakota jako UI webu, návštěvník si ji spojí
#              s jedním významem napříč celým webem
#   NEUTRAL  = vedlejší/srovnávací série (šedá, ne další "barevný" tón)
PRIMARY_COLOR = "#4682b4"   # steelblue — plurál (majoritní kategorie, hlavní zjištění)
ACCENT_COLOR  = "#c1440e"   # terakota — neznámé/nenalezené/neshoda zdrojů
NEUTRAL_COLOR = "#9a9a92"   # neutrální šedá — singulár (srovnávací kategorie)

# Kategorická paleta pro grafy se 3+ srovnávanými skupinami (např. kraje, IJP
# kategorie shody) — NE automatická sns.color_palette(), žádné nečekané barvy.
CATEGORICAL_PALETTE = [PRIMARY_COLOR, NEUTRAL_COLOR, "#3d3a33", "#c7c4ba"]

BASELINE_LW = 2.0
BASELINE_LS = "--"

FIGSIZE_WIDE    = (11, 5)
FIGSIZE_DEFAULT = (8, 5)
FIGSIZE_SQUARE   = (7, 5)

DPI_SAVE = 150

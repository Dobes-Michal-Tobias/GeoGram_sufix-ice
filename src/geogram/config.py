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

# Fonty (stejné .ttf jako na webu přes Google Fonts <link>, staženy lokálně
# pro matplotlib — systém nemusí mít IBM Plex nainstalovaný jako OS font).
FONTS_DIR = BASE_DIR / "assets" / "fonts"


def ensure_dirs() -> None:
    """Vytvoří datové a výstupní adresáře, pokud neexistují."""
    for d in [PROCESSED_DIR, ASSETS_IMG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# --- Vizualizační styl ---------------------------------------------------------

STYLE       = "whitegrid"
FONT_SCALE  = 1.05
PALETTE     = "muted"

# Typografie sladěná se styles.css webu (--font-sans / --font-mono).
# FONT_SERIF je jen pro compose_poster() (samostatné postery/LinkedIn export)
# — nikdy pro graf, co jde přímo do článku, viz STYLE_GUIDE.md v research-template.
FONT_SANS = "IBM Plex Sans"
FONT_MONO = "IBM Plex Mono"
FONT_SERIF = "IBM Plex Serif"

# Mřížka/linky odpovídají --rule (#e3e1da) z webu — ne seaborn default šedá.
# Dvě volitelná pozadí (přepínáno přes set_style(bg=...)): "white" (výchozí,
# sedí na bílý web beze švu) nebo "cream" (teplé "papírové" pozadí, editorial
# styl a la European Correspondent — vhodné pro postery/PDF export, ne pro
# grafy vložené do bílé stránky).
BG_WHITE = "#ffffff"
BG_CREAM = "#f8f6f0"
GRID_ON_WHITE = "#e3e1da"
GRID_ON_CREAM = "#e2ddd5"
GRID_COLOR = GRID_ON_WHITE  # zpětná kompatibilita, kde se GRID_COLOR čte přímo
INK_COLOR  = "#14171a"
MUTED_COLOR = "#696e76"

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

# "Tenké" sloupce/čáry místo plných seaborn-default barů (~0.8 šířky) — méně
# barevné plochy na obrazovce, blíž datové žurnalistice než výchozímu Excelu.
BAR_WIDTH   = 0.5    # šířka ax.bar/ax.barh (místo default ~0.8)
LOLLIPOP_LW = 2.5     # tloušťka čáry u lollipop grafů (barh náhrada pro 1 řadu)
LOLLIPOP_MS = 7       # velikost tečky na konci lollipop čáry

# Adaptivní výška horizontálních grafů podle počtu kategorií — 3 řádky
# nepotřebují stejnou plochu jako 15 (ROW_HEIGHT na kategorii + okraj).
ROW_HEIGHT   = 0.42
FIGSIZE_MIN_H = 2.0

# Grafy se na webu zobrazují na ~700px sloupci, ale často v ~1600-1900px
# fyzické šířce (figsize x DPI) — 150 DPI dávalo jen ~1.3-1.7x přesamplování,
# na HiDPI/retina obrazovkách (2-3x) tedy měkký/rozostřený text oproti
# ostrému HTML. 220 DPI dá bezpečnou rezervu i pro 2x displeje.
DPI_SAVE = 220

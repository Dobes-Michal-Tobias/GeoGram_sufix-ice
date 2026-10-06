"""Data loading and processing utilities for GeoGram."""

import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .ingest import integrate_municipalities


FALLBACK_NAME_COLUMNS = ["name", "název", "nazev", "obec", "OBEC"]


def infer_name_column(columns: Iterable[str]) -> str:
    """Infer the municipality name column from common Czech column names."""
    for candidate in FALLBACK_NAME_COLUMNS:
        if candidate in columns:
            return candidate
    return "name"


def load_municipalities(path: Path, encoding: str = "utf-8", **kwargs) -> pd.DataFrame:
    """Load municipality metadata from a CSV file and normalize the name column."""
    df = pd.read_csv(path, encoding=encoding, **kwargs)
    if "name" not in df.columns:
        name_column = infer_name_column(df.columns)
        if name_column in df.columns:
            df = df.rename(columns={name_column: "name"})
    return df


def load_all_municipalities(
    data_dir: Path = Path("data/raw"),
    population_file: Optional[str] = "pocty_obyvatel/1300722503.xlsx",
    coordinates_file: Optional[str] = "33bcdd-souradnice-mest-3d4b1fd/souradnice.csv",
) -> pd.DataFrame:
    """Load and integrate all municipality data from RÚIAN, ČSÚ and coordinate sources."""
    return integrate_municipalities(
        ui_obec_path=data_dir / "UI_OBEC.csv",
        ui_okres_path=data_dir / "UI_OKRES.csv",
        ui_vusc_path=data_dir / "UI_VUSC.csv",
        population_path=data_dir / population_file if population_file else None,
        coordinates_path=data_dir / coordinates_file if coordinates_file else None,
    )


# Prepositions introducing a locational complement ("Lipnice nad Sázavou").
# The complement is in an oblique case, so only the part before it decides
# whether the name is an -ice name.
_COMPLEMENT_PREPOSITIONS = {"nad", "pod", "u", "v", "ve", "na", "při", "za", "před", "mezi", "k", "ke"}
_ROMAN_NUMERAL = re.compile(r"^[IVX]+$")

NAME_SIMPLE = "simple"            # Pardubice, Velké Popovice
NAME_COMPLEMENT = "complement"    # Lipnice nad Sázavou, Babice u Rosic
NAME_COMPOUND = "compound"        # Bílovice-Lutotín (merged municipalities)
NAME_NUMBERED = "numbered"        # Opatovice I, Petrovice II


def _split_name(name: str) -> tuple[str, str]:
    """Split a name into (core, name_type) — core is the part that bears number."""
    tokens = name.split()
    name_type = NAME_SIMPLE
    if len(tokens) > 1 and _ROMAN_NUMERAL.match(tokens[-1]):
        tokens = tokens[:-1]
        name_type = NAME_NUMBERED
    for i, tok in enumerate(tokens[1:], start=1):
        if tok.lower() in _COMPLEMENT_PREPOSITIONS:
            tokens = tokens[:i]
            name_type = NAME_COMPLEMENT
            break
    core = " ".join(tokens)
    if "-" in core:
        parts = [p.strip() for p in core.split("-")]
        ice_parts = [p for p in parts if p.endswith("ice")]
        if ice_parts:
            return ice_parts[0], NAME_COMPOUND
    return core, name_type


def ice_core(name: str) -> Optional[str]:
    """Return the -ice part of a municipality name, or None if it is not an -ice name.

    >>> ice_core("Lipnice nad Sázavou")
    'Lipnice'
    >>> ice_core("Velké Popovice")
    'Velké Popovice'
    >>> ice_core("Bílovice-Lutotín")
    'Bílovice'
    >>> ice_core("Ústí nad Orlicí") is None
    True
    """
    if not isinstance(name, str):
        return None
    core, _ = _split_name(name)
    return core if core.endswith("ice") else None


def ice_head(name: str) -> Optional[str]:
    """Return the single -ice word ("Velké Popovice" → "Popovice"), or None."""
    core = ice_core(name)
    return core.split()[-1] if core else None


def ice_name_type(name: str) -> Optional[str]:
    """Return simple / complement / compound / numbered for -ice names, else None."""
    if ice_core(name) is None:
        return None
    return _split_name(name)[1]


def filter_suffix_ice(df: pd.DataFrame, name_column: str = "name") -> pd.DataFrame:
    """Return -ice municipalities, incl. multi-word names ("Lipnice nad Sázavou").

    Adds ``ice_head`` (the -ice word) and ``name_type`` columns.
    """
    out = df[df[name_column].map(ice_core).notna()].copy()
    out["ice_head"] = out[name_column].map(ice_head)
    out["name_type"] = out[name_column].map(ice_name_type)
    return out


def save_filtered_ice(df: pd.DataFrame, destination: Path, name_column: str = "name", **kwargs) -> pd.DataFrame:
    """Filter municipalities ending in '-ice' and save them to a CSV file."""
    filtered = filter_suffix_ice(df, name_column=name_column)
    destination.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(destination, index=False, **kwargs)
    return filtered


# ---------------------------------------------------------------------------
# Referenční data mimo -ice obce — celostátní populace, rozlohy krajů, hranice
# ---------------------------------------------------------------------------

# ČSÚ, "Počet obyvatel v regionech soudržnosti, krajích a okresech ČR" k 1. 1. 2025
# (data/raw/pocty_obyvatel/1300722501.xlsx, řádek "CZ0 Česko").
CZ_POPULATION_TOTAL_2025 = 10_909_500


def load_kraj_areas(path: Path | None = None) -> pd.DataFrame:
    """Načte rozlohy krajů v km² (Wikidata, P2046 — cituje ČSÚ ročenku ČR).

    Vrací DataFrame se sloupci `kraj` (shoduje se s `region_name` v ostatních
    datasetech) a `area_km2`. Praha je v datech zahrnutá pro úplnost, ale
    žádná obec na -ice v ní neleží (Praha je jedno město, ne soubor obcí).

    `path=None` (výchozí) čte z `config.RAW_DIR` — absolutní cesta, funguje
    stejně z notebooku (cwd = notebooks/) i odjinud.
    """
    from . import config

    path = path or config.RAW_DIR / "kraj_area_km2.csv"
    return pd.read_csv(path)


def load_cz_boundary(path: Path | None = None):
    """Načte zjednodušenou hranici ČR (GeoJSON, WGS84 / EPSG:4326).

    Zdroj: johan/world.geo.json (veřejný, zjednodušený polygon státní
    hranice — dostatečně přesný pro orientační obrys na mapě, ne pro
    přesné geodetické účely). Vrací `geopandas.GeoDataFrame`.

    `path=None` (výchozí) čte z `config.RAW_DIR` — viz `load_kraj_areas()`.
    """
    from . import config
    import geopandas as gpd

    path = path or config.RAW_DIR / "boundary" / "cz_boundary.geojson"

    return gpd.read_file(path)


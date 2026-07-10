"""Data loading and processing utilities for GeoGram."""

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


def filter_suffix_ice(df: pd.DataFrame, name_column: str = "name") -> pd.DataFrame:
    """Return rows where the municipality name ends with '-ice'."""
    return df[df[name_column].str.endswith("ice", na=False)].copy()


def save_filtered_ice(df: pd.DataFrame, destination: Path, name_column: str = "name", **kwargs) -> pd.DataFrame:
    """Filter municipalities ending in '-ice' and save them to a CSV file."""
    filtered = filter_suffix_ice(df, name_column=name_column)
    destination.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(destination, index=False, **kwargs)
    return filtered


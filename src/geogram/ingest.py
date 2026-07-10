"""Data ingestion and integration pipeline for GeoGram."""

from pathlib import Path
from typing import Optional

import pandas as pd


def load_ui_obec(path: Path) -> pd.DataFrame:
    """Load UI_OBEC.csv and normalize structure."""
    df = pd.read_csv(path, sep=";", encoding="cp1250")
    return df.rename(columns={
        "KOD": "code",
        "NAZEV": "name",
        "OKRES_KOD": "district_code",
        "STATUS_KOD": "status_code",
    })


def load_ui_okres(path: Path) -> pd.DataFrame:
    """Load UI_OKRES.csv to map district codes to names."""
    df = pd.read_csv(path, sep=";", encoding="cp1250")
    return df.rename(columns={
        "KOD": "district_code",
        "NAZEV": "district_name",
        "VUSC_KOD": "region_code",
    })[["district_code", "district_name", "region_code"]]


def load_ui_vusc(path: Path) -> pd.DataFrame:
    """Load UI_VUSC.csv to map region codes to names."""
    df = pd.read_csv(path, sep=";", encoding="cp1250")
    return df.rename(columns={
        "KOD": "region_code",
        "NAZEV": "region_name",
    })[["region_code", "region_name"]]


def load_population(path: Path) -> pd.DataFrame:
    """Load population data from ČSÚ Excel file."""
    df = pd.read_excel(path, sheet_name="List1", skiprows=4)
    df.columns = [
        "district_code",
        "municipality_code",
        "municipality_name",
        "population_total",
        "population_men",
        "population_women",
        "avg_age_total",
        "avg_age_men",
        "avg_age_women",
    ]
    df = df.dropna(subset=["municipality_code"])
    df["municipality_code"] = df["municipality_code"].astype("Int64")
    return df


def load_coordinates(path: Path) -> pd.DataFrame:
    """Load municipality coordinates from the souradnice-mest CSV export."""
    df = pd.read_csv(path, encoding="utf-8")
    df = df.rename(columns={
        "Kód obce": "municipality_code",
        "Latitude": "latitude",
        "Longitude": "longitude",
    })[["municipality_code", "latitude", "longitude"]]
    # Some municipalities (e.g. shared name+code across sources) can repeat;
    # keep the first occurrence per code.
    df = df.drop_duplicates(subset="municipality_code")
    df["municipality_code"] = df["municipality_code"].astype("Int64")
    return df


def integrate_municipalities(
    ui_obec_path: Path,
    ui_okres_path: Path,
    ui_vusc_path: Path,
    population_path: Optional[Path] = None,
    coordinates_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Integrate all data sources into a single denormalized table."""
    # Load base tables
    obec = load_ui_obec(ui_obec_path)
    okres = load_ui_okres(ui_okres_path)
    vusc = load_ui_vusc(ui_vusc_path)

    # Merge with administrative hierarchy
    df = obec.merge(okres, on="district_code", how="left")
    df = df.merge(vusc, on="region_code", how="left")

    # Add population if provided
    if population_path and population_path.exists():
        population = load_population(population_path)
        # Keep only the population columns we need
        pop_subset = population[
            ["municipality_code", "population_total", "population_men", "population_women", "avg_age_total"]
        ]
        df = df.merge(pop_subset, left_on="code", right_on="municipality_code", how="left")
        df = df.drop(columns=["municipality_code"], errors="ignore")

    # Add coordinates if provided
    if coordinates_path and coordinates_path.exists():
        coordinates = load_coordinates(coordinates_path)
        df = df.merge(coordinates, left_on="code", right_on="municipality_code", how="left")
        df = df.drop(columns=["municipality_code"], errors="ignore")

    # Clean up columns
    columns = [
        "code",
        "name",
        "district_code",
        "district_name",
        "region_code",
        "region_name",
        "status_code",
        "population_total",
        "population_men",
        "population_women",
        "avg_age_total",
    ]
    if coordinates_path and coordinates_path.exists():
        columns += ["latitude", "longitude"]
    df = df[columns]

    # Sort by region, then district, then name
    df = df.sort_values(by=["region_code", "district_code", "name"]).reset_index(drop=True)

    return df

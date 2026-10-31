"""Export dat pro článek na webu (michaltobiasdobes.cz, projekt Obce na -ice).

Vyrobí z ``ice_grammar_wiki_v2.csv`` dva soubory, které web načítá:

* ``obce-ice-table.json`` — řádky interaktivní tabulky (``obce-ice-table.js``)
* ``obce-ice-stats.json`` — výsledky statistických testů + blok ``text``
  se všemi čísly, která se citují v textu článku (počty, mediány, LISA…)

Počítá se stejnými funkcemi z ``viz.py`` a stejnými parametry (k = 6,
seed = 42, počty permutací) jako notebooky 07–13, takže čísla sedí na grafy.

Použití::

    python -m geogram.web_export                       # do ../personalWeb/assets/geogram
    python -m geogram.web_export --out-dir some/dir
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, data, viz
from .morphology import add_suffix_column, suffix_ending

WIKI_CSV = config.PROCESSED_DIR / "ice_grammar_wiki_v2.csv"
INTEGRATED_CSV = config.PROCESSED_DIR / "municipalities_ice_integrated.csv"
DEFAULT_OUT_DIR = config.BASE_DIR.parent / "personalWeb" / "assets" / "geogram"

_NUMBER_CS = {"plural": "Plurál", "singular": "Singulár", "unknown": "Neznámé"}
_SUFFIX_CS = {"ovice": "-ovice", "nice": "-nice", "bare_ice": "holé -ice"}


def load_wiki(wiki_csv: Path | None = None, integrated_csv: Path | None = None) -> pd.DataFrame:
    """Wiki labely + souřadnice (a typ názvu) z integrovaného CSV, jako v nb08."""
    wiki = pd.read_csv(wiki_csv or WIKI_CSV)
    extra = pd.read_csv(integrated_csv or INTEGRATED_CSV)
    cols = ["code"] + [c for c in ("latitude", "longitude", "name_type") if c in extra and c not in wiki]
    return add_suffix_column(wiki.merge(extra[cols], on="code", how="left"))


def _table_zeme(df: pd.DataFrame) -> pd.Series:
    """Čechy / Morava / Morava+Slezsko — jako ve stávající tabulce na webu.

    Kraje dle ``add_land_column``, jen Vysočina rozpuštěná podle okresu
    (Havlíčkobrodsko + Pelhřimovsko → Čechy, zbytek → Morava).
    """
    land = viz.add_land_column(df)["land"]
    binary = viz.add_land_column_binary(df)["land_binary"]
    is_vys = df["region_name"] == "Kraj Vysočina"
    return land.where(~is_vys, binary)


def build_table(df: pd.DataFrame) -> list[dict]:
    out = pd.DataFrame({
        "name": df["name"],
        "region": df["region_name"].str.replace(r"^Kraj | kraj$", "", regex=True),
        "zeme": _table_zeme(df),
        "population": df["population_total"].fillna(0).astype(int),
        "number": df["wiki_number"].map(_NUMBER_CS).fillna("Neznámé"),
        "suffix": df["suffix_group"].map(_SUFFIX_CS),
    })
    return out.sort_values("population", ascending=False).to_dict("records")


def _clean(obj):
    """Drop non-JSON values (DataFrames, statsmodels objects), cast numpy scalars."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()
                if not isinstance(v, (pd.DataFrame, pd.Series)) and _jsonable(v)}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def _jsonable(v) -> bool:
    return isinstance(v, (dict, list, tuple, np.ndarray, str, int, float, bool, type(None),
                          np.integer, np.floating))


def build_stats(df: pd.DataFrame) -> dict:
    cls = df[df["wiki_number"].isin(["singular", "plural"])]

    stats = {
        "chi2_suffix": viz.chi2_suffix_test(df),
        "chi2_land": viz.chi2_land_test(df),
        "chi2_region": viz.chi2_region_test(df),
        "join_count": viz.join_count_test(df, k=6, permutations=9999, seed=42),
        "moran": viz.moran_binary_test(df, k=6, permutations=9999, seed=42),
        "unknown_population": viz.test_unknown_population(df),
        "population_by_label": viz.test_population_by_label(df),
    }
    models = viz.fit_nested_logit_models(df)
    names = list(models)
    stats["pseudo_r2"] = {n: models[n].prsquared for n in names}
    stats["lr_tests"] = {
        "populace_to_sufix_plus_populace": viz.lr_test(models["populace"], models["sufix + populace"]),
        "sufix_to_full": viz.lr_test(models["sufix"], models["sufix + populace + region"]),
    }
    stats["chi2_unknown_land"] = viz.chi2_unknown_land_test(df)
    stats["chi2_unknown_suffix"] = viz.chi2_unknown_suffix_test(df)

    stats["text"] = build_text_numbers(df, cls)
    return _clean(stats)


def build_text_numbers(df: pd.DataFrame, cls: pd.DataFrame) -> dict:
    """Čísla citovaná v próze článku (mimo testy výše)."""
    pop = df["population_total"]
    counts = df["wiki_number"].value_counts()

    by_suffix = (cls.groupby("suffix_group")["wiki_number"]
                 .agg(n="count", n_plural=lambda s: (s == "plural").sum()))
    by_suffix["pct_plural"] = by_suffix["n_plural"] / by_suffix["n"]
    suffix_sizes = df["suffix_group"].value_counts()

    land = viz.add_land_column_binary(cls)
    pct_land = land.groupby("land_binary")["wiki_number"].apply(lambda s: (s == "plural").mean())

    gdf, _ = viz.moran_local_test(df, k=6, permutations=9999, seed=42)
    sig = gdf[gdf["lisa_p"] < 0.05]
    q = sig["lisa_q"].map({1: "HH", 2: "LH", 3: "LL", 4: "HL"}).value_counts()
    ll_land = viz.add_land_column(sig[sig["lisa_q"] == 3])["land"].value_counts()

    areas = data.load_kraj_areas()
    region_counts = df["region_name"].value_counts()
    dens = areas.set_index("kraj").join(region_counts.rename("n"), how="inner")
    dens["density"] = dens["n"] / dens["area_km2"] * 1000
    rank_count = region_counts.rank(ascending=False, method="min").astype(int)
    rank_density = dens["density"].rank(ascending=False, method="min").astype(int)
    rank_area = dens["area_km2"].rank(ascending=False, method="min").astype(int)

    end5 = cls.assign(end5=cls["name"].map(suffix_ending))
    g5 = end5.groupby("end5")["wiki_number"].agg(n="count", pct_plural=lambda s: (s == "plural").mean())
    g5 = g5[g5["n"] >= 15].sort_values("pct_plural")

    ovice_sg = cls[(cls["suffix_group"] == "ovice") & (cls["wiki_number"] == "singular")]
    oles = df[df["name"].str.match(r"^(Olešnice|Líšnice)\b")]

    return {
        "n_total": len(df),
        "n_plural": int(counts.get("plural", 0)),
        "n_singular": int(counts.get("singular", 0)),
        "n_unknown": int(counts.get("unknown", 0)),
        "n_classified": len(cls),
        "name_type": df["name_type"].value_counts().to_dict() if "name_type" in df else {},
        "population_sum": int(pop.sum()),
        "population_mean": float(pop.mean()),
        "population_median": float(pop.median()),
        "population_share_cz": float(pop.sum() / data.CZ_POPULATION_TOTAL_2025),
        "region_counts": region_counts.to_dict(),
        "region_rank_count": rank_count.to_dict(),
        "region_rank_density": rank_density.to_dict(),
        "region_rank_area": rank_area.to_dict(),
        "region_density": dens["density"].round(2).to_dict(),
        "suffix_sizes": suffix_sizes.to_dict(),
        "suffix_share_of_all": (suffix_sizes / len(df)).to_dict(),
        "suffix_plural": by_suffix.to_dict("index"),
        "land_binary_pct_plural": pct_land.to_dict(),
        "lisa_counts": q.to_dict(),
        "lisa_ll_by_land": ll_land.to_dict(),
        "end5_pct_plural": g5.to_dict("index"),
        "ovice_singular": ovice_sg["name"].tolist(),
        "olesnice_lisnice": dict(zip(oles["name"], oles["wiki_number"])),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    df = load_wiki()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    table = build_table(df)
    (args.out_dir / "obce-ice-table.json").write_text(
        json.dumps(table, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    stats = build_stats(df)
    (args.out_dir / "obce-ice-stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{len(table)} řádků tabulky + statistiky → {args.out_dir}")


if __name__ == "__main__":
    main()

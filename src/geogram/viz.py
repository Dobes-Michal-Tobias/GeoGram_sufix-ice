"""Visualization functions for GeoGram -ice municipality analysis.

All public functions build and return their own `matplotlib.Figure` (unless an
`ax` is passed in for composition into a larger grid). Notebooks should only
call these functions and save the result via `save_fig()` — no plotting logic
or styling decisions live in notebooks.

Barevná paleta a rozměry grafů se řídí `src/geogram/config.py` (3 sémantické
role: PRIMARY / ACCENT / NEUTRAL), stejná konvence jako v PeriodSim.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

from . import config

# ---------------------------------------------------------------------------
# Global theme and palette
# ---------------------------------------------------------------------------


def set_style() -> None:
    """Nastaví seaborn theme dle config.py. Volej jednou na začátku notebooku."""
    sns.set_theme(style=config.STYLE, palette=config.PALETTE, font_scale=config.FONT_SCALE)


def save_fig(fig: plt.Figure, filename: str, subdir: Path | None = None) -> Path:
    """Uloží figuru jako PNG do assets/img/geogram/ (pro publikaci na webu)."""
    target_dir = subdir or config.ASSETS_IMG_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{filename}.png"
    fig.savefig(path, dpi=config.DPI_SAVE, bbox_inches="tight")
    return path


# Datová paleta — role, ne libovolné barvy:
#   plural/match        = PRIMARY  (hlavní/očekávaná kategorie)
#   singular/wiki_only   = NEUTRAL  (vedlejší/srovnávací kategorie)
#   mismatch             = ACCENT   (jediné legitimní použití accent v projektu:
#                                    neshoda zdrojů Wikipedia vs. IJP — skutečný
#                                    "problém", ne jen odlišovací barva)
#   unknown/both/…       = odstíny šedé z CATEGORICAL_PALETTE (bez dat, ne risk)
COLORS = {
    "plural":        config.PRIMARY_COLOR,
    "singular":      config.NEUTRAL_COLOR,
    "both":          config.CATEGORICAL_PALETTE[2],
    "unknown":       config.CATEGORICAL_PALETTE[3],
    "not_found":     config.CATEGORICAL_PALETTE[3],
    "error":         config.CATEGORICAL_PALETTE[3],
    # IJP agreement categories
    "match":         config.PRIMARY_COLOR,
    "mismatch":      config.ACCENT_COLOR,
    "wiki_only":     config.NEUTRAL_COLOR,
    "ujc_only":      config.CATEGORICAL_PALETTE[2],
    "both_unknown":  config.CATEGORICAL_PALETTE[3],
}

_SOURCE_LABELS = {"wiki_number": "Wikipedia", "ujc_number": "IJP ÚJČ"}

# ---------------------------------------------------------------------------
# Region helpers
# ---------------------------------------------------------------------------

_LAND_MAP = {
    "Středočeský kraj":     "Čechy",
    "Jihočeský kraj":       "Čechy",
    "Plzeňský kraj":        "Čechy",
    "Karlovarský kraj":     "Čechy",
    "Ústecký kraj":         "Čechy",
    "Liberecký kraj":       "Čechy",
    "Královéhradecký kraj": "Čechy",
    "Pardubický kraj":      "Čechy",
    "Jihomoravský kraj":    "Morava",
    "Olomoucký kraj":       "Morava",
    "Zlínský kraj":         "Morava+Slezsko",
    "Moravskoslezský kraj": "Morava+Slezsko",
    "Kraj Vysočina":        "Vysočina",
}

_REGION_SHORT = {
    "Středočeský kraj":     "Středočeský",
    "Jihočeský kraj":       "Jihočeský",
    "Plzeňský kraj":        "Plzeňský",
    "Karlovarský kraj":     "Karlovarský",
    "Ústecký kraj":         "Ústecký",
    "Liberecký kraj":       "Liberecký",
    "Královéhradecký kraj": "Královéhradecký",
    "Pardubický kraj":      "Pardubický",
    "Jihomoravský kraj":    "Jihomoravský",
    "Olomoucký kraj":       "Olomoucký",
    "Zlínský kraj":         "Zlínský",
    "Moravskoslezský kraj": "Moravskoslezský",
    "Kraj Vysočina":        "Vysočina",
}


def add_land_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'land' column (Čechy / Morava / Morava+Slezsko / Vysočina)."""
    out = df.copy()
    out["land"] = out["region_name"].map(_LAND_MAP).fillna("Neznámý")
    return out


# ---------------------------------------------------------------------------
# 1. Overview: distribuce sg/pl/unknown
# ---------------------------------------------------------------------------

def plot_overview(df: pd.DataFrame, column: str = "wiki_number", ax: plt.Axes | None = None) -> plt.Figure:
    """Horizontal bar chart with absolute counts and percentages.

    column: "wiki_number" (default) nebo "ujc_number" — obě mají stejnou
    sadu kategorií (singular/plural/both/unknown/not_found/error).
    """
    set_style()
    counts = df[column].value_counts()
    order = [c for c in ["plural", "singular", "both", "unknown", "not_found", "error"] if c in counts.index]
    labels = {"plural": "Plurál", "singular": "Singulár", "both": "Obojí",
              "unknown": "Neznámé", "not_found": "Nenalezeno", "error": "Chyba"}

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
    else:
        fig = ax.figure

    total = len(df)
    palette = [COLORS.get(c, config.NEUTRAL_COLOR) for c in order]
    sns.barplot(
        x=[counts[c] for c in order],
        y=[labels.get(c, c) for c in order],
        palette=palette, orient="h", ax=ax,
    )
    for i, c in enumerate(order):
        n = counts[c]
        ax.text(n + 15, i, f"{n} ({n/total:.1%})", va="center", fontsize=9)

    ax.set_xlabel("Počet obcí")
    ax.set_title(f"Distribuce gramatického čísla obcí -ice\n(zdroj: {_SOURCE_LABELS.get(column, column)})")
    ax.set_xlim(0, max(counts) * 1.20)
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Stacked bar po krajích
# ---------------------------------------------------------------------------

def plot_by_region(df: pd.DataFrame, column: str = "wiki_number", ax: plt.Axes | None = None) -> plt.Figure:
    """Horizontal stacked bar: počet sg/pl per kraj, seřazeno dle % plurálu."""
    set_style()
    d = df[df[column].isin(["singular", "plural"])].copy()
    pivot = (
        d.groupby(["region_name", column])
        .size()
        .unstack(fill_value=0)
    )
    pivot["total"] = pivot.sum(axis=1)
    pivot["pct_pl"] = pivot.get("plural", 0) / pivot["total"]
    pivot = pivot.sort_values("pct_pl", ascending=True)
    short = [_REGION_SHORT.get(r, r) for r in pivot.index]

    if ax is None:
        fig, ax = plt.subplots(figsize=config.FIGSIZE_WIDE)
    else:
        fig = ax.figure

    bottom = np.zeros(len(pivot))
    for cat, label in [("plural", "Plurál"), ("singular", "Singulár")]:
        vals = pivot.get(cat, pd.Series(0, index=pivot.index)).values
        ax.barh(short, vals, left=bottom, label=label, color=COLORS[cat])
        bottom += vals

    # % label on each bar
    for i, (_, row) in enumerate(pivot.iterrows()):
        ax.text(row["total"] + 2, i, f"{row['pct_pl']:.0%}",
                va="center", fontsize=8, color="#444")

    ax.set_xlabel("Počet obcí")
    ax.set_title(f"Singulár vs. plurál dle kraje ({_SOURCE_LABELS.get(column, column)})\n"
                 "(seřazeno vzestupně dle podílu plurálu)")
    ax.legend(loc="lower right", frameon=False)
    ax.invert_yaxis()
    sns.despine(ax=ax, left=True, bottom=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Stacked bar Čechy / Morava / Vysočina
# ---------------------------------------------------------------------------

def plot_by_land(df: pd.DataFrame, column: str = "wiki_number", ax: plt.Axes | None = None) -> plt.Figure:
    """Stacked bar: sg vs. pl pro Čechy / Morava+Slezsko / Vysočina."""
    set_style()
    d = add_land_column(df)
    d = d[d[column].isin(["singular", "plural"])]
    pivot = (
        d.groupby(["land", column])
        .size()
        .unstack(fill_value=0)
    )
    pivot["total"] = pivot.sum(axis=1)
    pivot["pct_pl"] = pivot.get("plural", 0) / pivot["total"]
    pivot = pivot.sort_values("pct_pl")

    if ax is None:
        fig, ax = plt.subplots(figsize=config.FIGSIZE_SQUARE)
    else:
        fig = ax.figure

    bottom = np.zeros(len(pivot))
    for cat, label in [("plural", "Plurál"), ("singular", "Singulár")]:
        vals = pivot.get(cat, pd.Series(0, index=pivot.index)).values
        ax.bar(pivot.index, vals, bottom=bottom, label=label, color=COLORS[cat])
        bottom += vals

    for i, (_, row) in enumerate(pivot.iterrows()):
        ax.text(i, row["total"] + 5, f"{row['pct_pl']:.0%}",
                ha="center", fontsize=10, fontweight="bold", color="#222")

    ax.set_ylabel("Počet obcí")
    ax.set_title(f"Singulár vs. plurál ({_SOURCE_LABELS.get(column, column)})\ndle historické země")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Logistická regrese: populace → sg/pl
# ---------------------------------------------------------------------------

def plot_population_logistic(
    df: pd.DataFrame,
    column: str = "wiki_number",
    ax: plt.Axes | None = None,
    log_scale: bool = True,
) -> tuple[plt.Figure, object]:
    """Scatter y=0/1 (sg/pl) × populace + logistická regresní křivka.

    Returns (fig, statsmodels LogitResults).
    """
    set_style()
    import statsmodels.api as sm

    d = df[df[column].isin(["singular", "plural"])].dropna(subset=["population_total"])
    d = d[d["population_total"] > 0].copy()
    d["y"] = (d[column] == "plural").astype(int)

    x_raw = np.log10(d["population_total"]) if log_scale else d["population_total"]
    X = sm.add_constant(x_raw)
    result = sm.Logit(d["y"], X).fit(disp=False)

    if ax is None:
        fig, ax = plt.subplots(figsize=config.FIGSIZE_DEFAULT)
    else:
        fig = ax.figure

    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.03, 0.03, size=len(d))
    for cat in ["singular", "plural"]:
        mask = d[column] == cat
        ax.scatter(
            x_raw[mask], d["y"][mask] + jitter[mask],
            color=COLORS[cat], alpha=0.20, s=9, linewidths=0,
            label=cat.capitalize(),
        )

    x_line = np.linspace(x_raw.min(), x_raw.max(), 300)
    X_line = sm.add_constant(x_line)
    ax.plot(x_line, result.predict(X_line), color="#222", lw=2.2, label="Logistická regrese")

    ax.set_xlabel("log₁₀(počet obyvatel)" if log_scale else "Počet obyvatel")
    ax.set_ylabel("Gramatické číslo")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Singulár", "Plurál"])

    coef = result.params.iloc[1]
    pval = result.pvalues.iloc[1]
    ax.set_title(f"Logistická regrese ({_SOURCE_LABELS.get(column, column)}): populace → sg/pl\n"
                 f"β = {coef:.3f}, p = {pval:.2e}")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig, result


# ---------------------------------------------------------------------------
# 5. Violin plot: distribuce populace dle sg/pl
# ---------------------------------------------------------------------------

def plot_population_violin(
    df: pd.DataFrame,
    column: str = "wiki_number",
    ax: plt.Axes | None = None,
    log_scale: bool = True,
) -> plt.Figure:
    """Violin plot: populace_total rozdělená dle column (wiki_number/ujc_number)."""
    set_style()
    d = df[df[column].isin(["singular", "plural"])].dropna(subset=["population_total"])
    d = d[d["population_total"] > 0].copy()
    d["pop_plot"] = np.log10(d["population_total"]) if log_scale else d["population_total"]
    ylabel = "log₁₀(počet obyvatel)" if log_scale else "Počet obyvatel"

    if ax is None:
        fig, ax = plt.subplots(figsize=config.FIGSIZE_SQUARE)
    else:
        fig = ax.figure

    order = ["singular", "plural"]
    sns.violinplot(
        data=d, x=column, y="pop_plot", order=order,
        palette={k: COLORS[k] for k in order},
        inner="box", density_norm="width", ax=ax,
    )
    ax.set_xticklabels(["Singulár", "Plurál"])
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Distribuce populace ({_SOURCE_LABELS.get(column, column)}): singulár vs. plurál")

    for i, cat in enumerate(order):
        med = d.loc[d[column] == cat, "pop_plot"].median()
        n = (d[column] == cat).sum()
        label = f"n={n}\nmed={10**med:,.0f}" if log_scale else f"n={n}"
        ax.text(i, ax.get_ylim()[1] * 0.99, label, ha="center", va="top", fontsize=8)

    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Unknown analýza: velikost nerozpoznaných obcí
# ---------------------------------------------------------------------------

def plot_missing_analysis(df: pd.DataFrame, column: str = "wiki_number", ax: plt.Axes | None = None) -> plt.Figure:
    """Histogramy: populace klasifikovaných vs. neznámých obcí."""
    set_style()
    classified = df[df[column].isin(["singular", "plural"]) & (df["population_total"] > 0)]
    unknown = df[(df[column] == "unknown") & (df["population_total"] > 0)]

    if ax is None:
        fig, ax = plt.subplots(figsize=config.FIGSIZE_WIDE)
    else:
        fig = ax.figure

    bins = np.logspace(
        np.log10(min(classified["population_total"].min(), unknown["population_total"].min())),
        np.log10(max(classified["population_total"].max(), unknown["population_total"].max())),
        30,
    )
    sns.histplot(classified["population_total"], bins=bins, ax=ax, color=COLORS["plural"],
                 alpha=0.55, label=f"Klasifikované (n={len(classified)})", stat="density", log_scale=True)
    sns.histplot(unknown["population_total"], bins=bins, ax=ax, color=COLORS["unknown"],
                 alpha=0.65, label=f"Neznámé (n={len(unknown)})", stat="density", log_scale=True)

    ax.set_xlabel("Počet obyvatel (log škála)")
    ax.set_ylabel("Hustota")
    ax.set_title(f"Velikost obcí ({_SOURCE_LABELS.get(column, column)}): klasifikované vs. nerozpoznané")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. Statistické testy
# ---------------------------------------------------------------------------

def chi2_land_test(df: pd.DataFrame, column: str = "wiki_number") -> dict:
    """Chi² test: je distribuce sg/pl nezávislá na historické zemi?"""
    d = add_land_column(df)[lambda x: x[column].isin(["singular", "plural"])]
    ct = pd.crosstab(d["land"], d[column])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    return {"chi2": chi2, "p": p, "dof": dof, "contingency": ct}


def chi2_region_test(df: pd.DataFrame, column: str = "wiki_number") -> dict:
    """Chi² test: distribuce sg/pl vs. kraj."""
    d = df[df[column].isin(["singular", "plural"])]
    ct = pd.crosstab(d["region_name"], d[column])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    return {"chi2": chi2, "p": p, "dof": dof, "contingency": ct}


# ---------------------------------------------------------------------------
# 8. IJP vs. Wikipedia – srovnání zdrojů
#    (funkce připraveny dopředu, data přijdou po mailu na ÚJČ)
# ---------------------------------------------------------------------------

_AGREE_ORDER = ["match", "mismatch", "wiki_only", "ujc_only", "both_unknown"]
_AGREE_LABELS = {
    "match":        "Oba zdroje shodně",
    "mismatch":     "Neshoda zdrojů",
    "wiki_only":    "Jen Wikipedia",
    "ujc_only":     "Jen IJP ÚJČ",
    "both_unknown": "Neznámé v obou",
}


def classify_agreement(df: pd.DataFrame,
                        wiki_col: str = "wiki_number",
                        ujc_col: str = "ujc_number") -> pd.DataFrame:
    """Přidá sloupec 'agreement' na základě wiki_number a ujc_number.

    Kategorie:
      match        – oba zdroje mají stejnou hodnotu (singular/plural)
      mismatch     – oba klasifikují, ale různě
      wiki_only    – Wikipedia klasifikuje, IJP ne (not_found/unknown)
      ujc_only     – IJP klasifikuje, Wikipedia ne
      both_unknown – oba zdroje neznají
    """
    KNOWN = {"singular", "plural", "both"}
    out = df.copy()

    wiki = out[wiki_col].fillna("unknown")
    ujc = out[ujc_col].fillna("unknown")
    wiki_known = wiki.isin(KNOWN)
    ujc_known = ujc.isin(KNOWN)

    conditions = [
        wiki_known & ujc_known & (wiki == ujc),
        wiki_known & ujc_known & (wiki != ujc),
        wiki_known & ~ujc_known,
        ~wiki_known & ujc_known,
    ]
    choices = ["match", "mismatch", "wiki_only", "ujc_only"]
    out["agreement"] = np.select(conditions, choices, default="both_unknown")
    return out


def plot_agreement_summary(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Figure:
    """Sloupcový graf kategorií shody Wikipedia vs. IJP."""
    set_style()
    if "agreement" not in df.columns:
        df = classify_agreement(df)

    counts = df["agreement"].value_counts()
    order = [c for c in _AGREE_ORDER if c in counts.index]
    total = len(df)

    if ax is None:
        fig, ax = plt.subplots(figsize=config.FIGSIZE_DEFAULT)
    else:
        fig = ax.figure

    palette = [COLORS[c] for c in order]
    sns.barplot(
        x=[counts[c] for c in order],
        y=[_AGREE_LABELS[c] for c in order],
        palette=palette, orient="h", ax=ax,
    )
    for i, c in enumerate(order):
        n = counts[c]
        ax.text(n + 5, i, f"{n} ({n/total:.1%})", va="center", fontsize=9)

    ax.set_xlabel("Počet obcí")
    ax.set_title("Shoda Wikipedia vs. IJP ÚJČ")
    ax.set_xlim(0, max(counts) * 1.22)
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    return fig


def plot_agreement_heatmap(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Figure:
    """Matice záměn (Wikipedia × IJP) jako seaborn heatmap."""
    set_style()
    if "agreement" not in df.columns:
        df = classify_agreement(df)

    KNOWN = ["singular", "plural", "both"]
    d = df[df["wiki_number"].isin(KNOWN) & df["ujc_number"].isin(KNOWN)]
    ct = pd.crosstab(
        d["wiki_number"].rename("Wikipedia"),
        d["ujc_number"].rename("IJP ÚJČ"),
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=config.FIGSIZE_SQUARE)
    else:
        fig = ax.figure

    sns.heatmap(
        ct, annot=True, fmt="d", cmap="Blues",
        linewidths=0.5, ax=ax, cbar_kws={"label": "Počet obcí"},
    )
    ax.set_title("Matice záměn: Wikipedia vs. IJP ÚJČ\n(jen klasifikované v obou)")
    fig.tight_layout()
    return fig


def plot_agreement_by_region(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Figure:
    """Stacked bar: kategorie shody po krajích."""
    set_style()
    if "agreement" not in df.columns:
        df = classify_agreement(df)

    pivot = (
        df.groupby(["region_name", "agreement"])
        .size()
        .unstack(fill_value=0)
    )
    order_cols = [c for c in _AGREE_ORDER if c in pivot.columns]
    pivot = pivot[order_cols]
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=True)
    short = [_REGION_SHORT.get(r, r) for r in pivot.index]

    if ax is None:
        fig, ax = plt.subplots(figsize=(config.FIGSIZE_WIDE[0], 5.5))
    else:
        fig = ax.figure

    bottom = np.zeros(len(pivot))
    patches = []
    for cat in order_cols:
        vals = pivot[cat].values
        ax.barh(short, vals, left=bottom, color=COLORS[cat], label=_AGREE_LABELS[cat])
        patches.append(mpatches.Patch(color=COLORS[cat], label=_AGREE_LABELS[cat]))
        bottom += vals

    ax.set_xlabel("Počet obcí")
    ax.set_title("Shoda Wikipedia vs. IJP ÚJČ dle kraje")
    ax.legend(handles=patches, loc="lower right", fontsize=8)
    ax.invert_yaxis()
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    return fig


def plot_mismatch_details(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Figure:
    """Bar chart neshodujících se obcí: co říká Wikipedia vs. IJP."""
    set_style()
    if "agreement" not in df.columns:
        df = classify_agreement(df)

    mis = df[df["agreement"] == "mismatch"].copy()
    if ax is None:
        fig, ax = plt.subplots(figsize=config.FIGSIZE_DEFAULT if not mis.empty else (6, 3))
    else:
        fig = ax.figure

    if mis.empty:
        ax.text(0.5, 0.5, "Žádné neshody", ha="center", va="center", transform=ax.transAxes)
        return fig

    label = mis["wiki_number"] + " → " + mis["ujc_number"]
    counts = label.value_counts()

    sns.barplot(x=counts.values, y=counts.index, color=COLORS["mismatch"], ax=ax)
    ax.set_xlabel("Počet obcí")
    ax.set_title(f"Neshody Wikipedia → IJP (n={len(mis)})")
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 9. Interaktivní mapa (jediný interaktivní prvek v projektu — folium)
# ---------------------------------------------------------------------------

# Na mapě celé ČR dvě podobně syté šedé (singulár/neznámé) vizuálně splývají,
# ale terakota (ACCENT) se sem nesmí přesunout jako "druhá barva" — ta nese na
# celém webu jeden konkrétní význam (problém/upozornění), ne kategorii dat.
# Řešení je vizuální kódování místo další barvy: singulár dostává tmavou,
# maximálně kontrastní "neutrální" barvu; neznámé/nenalezené je prázdný kroužek
# (bez výplně) — čte se jako "chybí data", běžná mapová konvence.
_MAP_STYLE = {
    "plural":    {"color": config.PRIMARY_COLOR,          "radius": 4, "fill": True,  "fill_opacity": 0.80, "weight": 1},
    "singular":  {"color": config.CATEGORICAL_PALETTE[2], "radius": 4, "fill": True,  "fill_opacity": 0.85, "weight": 1},
    "both":      {"color": config.CATEGORICAL_PALETTE[2], "radius": 4, "fill": True,  "fill_opacity": 0.85, "weight": 1},
}
_MAP_STYLE_DEFAULT = {
    "color": config.NEUTRAL_COLOR, "radius": 3, "fill": False, "fill_opacity": 0.0, "weight": 1.5,
}


def plot_map_folium(df: pd.DataFrame, lat_col: str = "latitude", lon_col: str = "longitude"):
    """Folium mapa s body obcí obarvených dle wiki_number.

    Plurál/singulár jsou plné tečky (steelblue/tmavá), neznámé a nenalezené
    jsou prázdné kroužky — na mapě celé ČR se tak neznámé vizuálně "ztrácí"
    do pozadí, místo aby soutěžily o pozornost s klasifikovanými kategoriemi.

    Vyžaduje sloupce latitude/longitude (z municipalities_ice_integrated.csv).
    Vrací folium.Map — v notebooku display(m), nebo m.save('mapa.html').
    """
    import folium

    d = df.dropna(subset=[lat_col, lon_col])
    m = folium.Map(location=[49.8, 15.5], zoom_start=7, tiles="CartoDB positron")

    for _, row in d.iterrows():
        wiki_number = row.get("wiki_number", "unknown")
        style = _MAP_STYLE.get(wiki_number, _MAP_STYLE_DEFAULT)
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=style["radius"], color=style["color"], weight=style["weight"],
            fill=style["fill"], fill_color=style["color"], fill_opacity=style["fill_opacity"],
            popup=(f"<b>{row.get('name','')}</b><br>"
                   f"{wiki_number} | {int(row.get('population_total',0))} obyv."),
        ).add_to(m)

    legend = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:999;background:white;
                padding:10px 14px;border:1px solid #ccc;border-radius:6px;font-size:13px;
                box-shadow:2px 2px 6px rgba(0,0,0,0.15)">
      <b>Gramatické číslo</b><br>
      <span style="color:{_MAP_STYLE['plural']['color']};">&#9679;</span> Plurál<br>
      <span style="color:{_MAP_STYLE['singular']['color']};">&#9679;</span> Singulár<br>
      <span style="color:{_MAP_STYLE_DEFAULT['color']};">&#9675;</span> Neznámé
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    return m


# ---------------------------------------------------------------------------
# 9. Prostorová segregace sg/pl (join-count, Moran's I) — notebook 08
#
# Otázka "leží stejné labely (plurál/singulár) blíž sobě navzájem, než by
# odpovídalo náhodnému rozmístění týchž labelů na týchž souřadnicích?" se
# testuje join-count statistikou (Cliff & Ord) a binárním Moran's I, oba s
# permutační (Monte Carlo) inferencí — klasické p-value zde nejsou platná,
# protože prostorová data nejsou i.i.d. (Toblerův zákon).
# ---------------------------------------------------------------------------

def build_knn_weights(df: pd.DataFrame, k: int = 6,
                       lat_col: str = "latitude", lon_col: str = "longitude"):
    """Projikuje lat/lon do S-JTSK (EPSG:5514, metrické) a postaví k-NN spatial weights.

    Vrací (gdf, w): gdf s metrickými souřadnicemi seřazený stejně jako
    w.id_order (0..n-1), a libpysal.weights.KNN objekt.
    """
    import geopandas as gpd
    from libpysal.weights import KNN

    d = df.dropna(subset=[lat_col, lon_col]).reset_index(drop=True)
    gdf = gpd.GeoDataFrame(
        d, geometry=gpd.points_from_xy(d[lon_col], d[lat_col]), crs="EPSG:4326"
    ).to_crs("EPSG:5514")
    w = KNN.from_dataframe(gdf, k=k)
    return gdf, w


def join_count_test(df: pd.DataFrame, column: str = "wiki_number",
                     k: int = 6, permutations: int = 9999, seed: int = 42) -> dict:
    """Join-count test prostorové segregace sg/pl na k-NN grafu (esda.Join_Counts).

    plurál kóduje se jako "B" (1), singulár jako "W" (0). bb/ww = počet
    sousedících dvojic se stejným labelem, bw = smíšené dvojice. p_sim_*
    je podíl z `permutations` náhodných přeskupení labelů na týchž
    souřadnicích, které dosáhly stejně extrémní hodnoty jako pozorovaná
    (random-labeling null model) — vhodná inference pro prostorová data.

    Vrací dict s pozorovanými/očekávanými počty joinů a permutačními p-values.
    """
    from esda.join_counts import Join_Counts

    np.random.seed(seed)
    d = df[df[column].isin(["singular", "plural"])].copy()
    gdf, w = build_knn_weights(d, k=k)
    y = (gdf[column] == "plural").astype(int).values

    jc = Join_Counts(y, w, permutations=permutations)

    return {
        "bb_obs": jc.bb, "bb_exp": jc.mean_bb, "p_sim_bb": jc.p_sim_bb,
        "ww_obs": jc.ww, "bw_obs": jc.bw, "bw_exp": jc.mean_bw, "p_sim_bw": jc.p_sim_bw,
        "chi2": jc.chi2, "chi2_p": jc.chi2_p,
        "k": k, "n": len(gdf), "permutations": permutations,
        "n_plural": int(y.sum()), "n_singular": int((1 - y).sum()),
        "result": jc,
    }


def moran_binary_test(df: pd.DataFrame, column: str = "wiki_number",
                       k: int = 6, permutations: int = 9999, seed: int = 42) -> dict:
    """Globální Moran's I pro binární proměnnou (1=plurál, 0=singulár) na k-NN grafu.

    Doplňkový, obecně známý ukazatel prostorové autokorelace k
    `join_count_test()`. Row-standardizované váhy (w.transform='r'),
    permutační p-value (esda.Moran).
    """
    from esda.moran import Moran

    np.random.seed(seed)
    d = df[df[column].isin(["singular", "plural"])].copy()
    gdf, w = build_knn_weights(d, k=k)
    w.transform = "r"
    y = (gdf[column] == "plural").astype(float).values

    mi = Moran(y, w, permutations=permutations)

    return {
        "I": mi.I, "EI": mi.EI, "z_sim": mi.z_sim, "p_sim": mi.p_sim,
        "k": k, "n": len(gdf), "permutations": permutations,
        "result": mi,
    }


def plot_moran_scatter(moran_result, ax: plt.Axes | None = None) -> plt.Figure:
    """Moran scatterplot: hodnota obce (x) vs. prostorově zpožděný průměr sousedů (y)."""
    set_style()
    from libpysal.weights import lag_spatial

    y = moran_result.z
    ylag = lag_spatial(moran_result.w, y)

    if ax is None:
        fig, ax = plt.subplots(figsize=config.FIGSIZE_SQUARE)
    else:
        fig = ax.figure

    ax.scatter(y, ylag, color=config.PRIMARY_COLOR, alpha=0.35, s=14, linewidths=0)
    b, a = np.polyfit(y, ylag, 1)
    x_line = np.linspace(y.min(), y.max(), 100)
    ax.plot(x_line, a + b * x_line, color=config.ACCENT_COLOR, lw=2, label=f"I = {moran_result.I:.3f}")
    ax.axhline(0, color="#999", lw=0.8)
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_xlabel("Label obce (centrovaný, 1=plurál/0=singulár)")
    ax.set_ylabel("Prostorově zpožděný průměr sousedů")
    ax.set_title(f"Moran scatterplot (p_sim = {moran_result.p_sim:.4f}, {moran_result.permutations} permutací)")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def moran_local_test(df: pd.DataFrame, column: str = "wiki_number",
                      k: int = 6, permutations: int = 999, seed: int = 42):
    """LISA (Local Indicators of Spatial Association) — KDE lokálních shluků.

    Doplněk ke globálnímu `moran_binary_test()`: globální test řekne "shlukuje
    se to", LISA řekne KDE. Vrací (gdf, esda.Moran_Local) — gdf má navíc
    sloupce `lisa_q` (kvadrant: 1=HH, 2=LH, 3=LL, 4=HL) a `lisa_p` (permutační
    p-value pro daný bod), gdf řazený stejně jako výsledek Moran_Local.
    """
    from esda.moran import Moran_Local

    d = df[df[column].isin(["singular", "plural"])].copy()
    gdf, w = build_knn_weights(d, k=k)
    w.transform = "r"
    y = (gdf[column] == "plural").astype(float).values

    ml = Moran_Local(y, w, permutations=permutations, seed=seed)
    gdf["lisa_q"] = ml.q
    gdf["lisa_p"] = ml.p_sim
    return gdf, ml


def plot_lisa_map(gdf: pd.DataFrame, alpha: float = 0.05, ax: plt.Axes | None = None) -> plt.Figure:
    """Mapa significantních LISA shluků (výstup `moran_local_test`).

    HH (plurál obklopený plurálem) a LL (singulár obklopený singulárem) jsou
    "typické" shluky — plná PRIMARY/NEUTRAL barva. HL/LH (obec jiného labelu
    než sousedi) jsou prostorové anomálie — ACCENT, stejná sémantika jako
    jinde v projektu (skutečná neshoda/výjimka, ne jen odlišení). Body s
    p >= alpha (nevýznamné) jsou tenké šedé tečky v pozadí.
    """
    set_style()
    _LISA_LABELS = {1: "HH (plurál shluk)", 2: "LH (anomálie)", 3: "LL (singulár shluk)", 4: "HL (anomálie)"}
    _LISA_COLORS = {1: config.PRIMARY_COLOR, 2: config.ACCENT_COLOR,
                     3: config.NEUTRAL_COLOR, 4: config.ACCENT_COLOR}

    if ax is None:
        fig, ax = plt.subplots(figsize=config.FIGSIZE_SQUARE)
    else:
        fig = ax.figure

    sig = gdf[gdf["lisa_p"] < alpha]
    nonsig = gdf[gdf["lisa_p"] >= alpha]

    ax.scatter(nonsig.geometry.x, nonsig.geometry.y, color="#ccc", s=6, alpha=0.4, linewidths=0,
               label=f"Nevýznamné (p ≥ {alpha})")
    for q, label in _LISA_LABELS.items():
        m = sig[sig["lisa_q"] == q]
        if len(m):
            ax.scatter(m.geometry.x, m.geometry.y, color=_LISA_COLORS[q], s=22, alpha=0.85,
                       linewidths=0, label=f"{label} (n={len(m)})")

    ax.set_title(f"LISA — lokální shluky sg/pl (p < {alpha})")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    sns.despine(ax=ax, left=True, bottom=True)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 10. Neznámé labely vs. velikost obce — notebook 09
# ---------------------------------------------------------------------------

def test_unknown_population(df: pd.DataFrame, column: str = "wiki_number") -> dict:
    """Statistický test: souvisí pravděpodobnost 'unknown' s populací obce?

    Dva doplňkové testy na stejných datech:
    - Mann-Whitney U (nonparametrický, robustní k šikmému rozdělení populace,
      nevyžaduje žádnou funkční formu vztahu) — porovnává rozdělení populace
      unknown vs. klasifikovaných obcí.
    - Logistická regrese is_unknown ~ log10(populace) (statsmodels) —
      kvantifikuje směr a velikost efektu (odds ratio) a dává p-value/CI.

    Obce s wiki_number == 'not_found'/'error' jsou vyloučeny (jiná kategorie
    chybovosti než 'unknown' — viz CLAUDE.md).
    """
    import statsmodels.api as sm

    d = df[df[column].isin(["singular", "plural", "unknown"])].dropna(subset=["population_total"])
    d = d[d["population_total"] > 0].copy()
    d["is_unknown"] = (d[column] == "unknown").astype(int)

    pop_unknown = d.loc[d["is_unknown"] == 1, "population_total"]
    pop_classified = d.loc[d["is_unknown"] == 0, "population_total"]
    u_stat, u_p = stats.mannwhitneyu(pop_unknown, pop_classified, alternative="less")

    x = np.log10(d["population_total"])
    X = sm.add_constant(x)
    logit = sm.Logit(d["is_unknown"], X).fit(disp=False)
    coef = logit.params.iloc[1]
    ci_low, ci_high = logit.conf_int().iloc[1]

    return {
        "n": len(d), "n_unknown": int(d["is_unknown"].sum()),
        "median_pop_unknown": pop_unknown.median(), "median_pop_classified": pop_classified.median(),
        "mannwhitney_u": u_stat, "mannwhitney_p": u_p,
        "logit_coef": coef, "logit_p": logit.pvalues.iloc[1],
        "logit_odds_ratio_per_10x_pop": np.exp(coef),
        "logit_ci_odds_ratio": (np.exp(ci_low), np.exp(ci_high)),
        "logit_pseudo_r2": logit.prsquared,
        "result": logit,
    }


def plot_unknown_population_logistic(df: pd.DataFrame, column: str = "wiki_number",
                                      ax: plt.Axes | None = None) -> tuple[plt.Figure, object]:
    """Scatter y=0/1 (klasifikováno/unknown) × log populace + logistická regresní křivka."""
    set_style()
    import statsmodels.api as sm

    d = df[df[column].isin(["singular", "plural", "unknown"])].dropna(subset=["population_total"])
    d = d[d["population_total"] > 0].copy()
    d["is_unknown"] = (d[column] == "unknown").astype(int)

    x_raw = np.log10(d["population_total"])
    X = sm.add_constant(x_raw)
    result = sm.Logit(d["is_unknown"], X).fit(disp=False)

    if ax is None:
        fig, ax = plt.subplots(figsize=config.FIGSIZE_DEFAULT)
    else:
        fig = ax.figure

    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.03, 0.03, size=len(d))
    labels = {0: ("Klasifikováno", config.PRIMARY_COLOR), 1: ("Neznámé", COLORS["unknown"])}
    for val, (label, color) in labels.items():
        mask = d["is_unknown"] == val
        ax.scatter(x_raw[mask], d["is_unknown"][mask] + jitter[mask],
                    color=color, alpha=0.20, s=9, linewidths=0, label=label)

    x_line = np.linspace(x_raw.min(), x_raw.max(), 300)
    X_line = sm.add_constant(x_line)
    ax.plot(x_line, result.predict(X_line), color="#222", lw=2.2, label="Logistická regrese")

    ax.set_xlabel("log₁₀(počet obyvatel)")
    ax.set_ylabel("")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Klasifikováno", "Neznámé"])

    coef = result.params.iloc[1]
    pval = result.pvalues.iloc[1]
    ax.set_title(f"Logistická regrese ({_SOURCE_LABELS.get(column, column)}): populace → neznámý label\n"
                 f"β = {coef:.3f}, OR (na 10× populaci) = {np.exp(coef):.3f}, p = {pval:.2e}")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig, result


# ---------------------------------------------------------------------------
# 10b. Kde na ose populace leží unknown — srovnání všech 3 labelů najednou
#
# test_unknown_population() se ptá "P(unknown | populace)" a slučuje
# singular+plural do jedné "klasifikované" skupiny — vhodné pro odds ratio,
# ale plurál (1410 obcí) tam čítá stejně jako singulár (233), takže případný
# rozdíl mezi singulár a plurál se v tom "zprůměruje" pryč. Otázka "kde na
# diskrétní ose populace leží unknown, ve srovnání se VŠEMI třemi labely
# zvlášť" potřebuje 3skupinové srovnání, ne 2skupinové.
# ---------------------------------------------------------------------------

_LABEL_ORDER = ["singular", "plural", "unknown"]
_LABEL_TITLES = {"singular": "Singulár", "plural": "Plurál", "unknown": "Neznámé"}


def _population_by_label(df: pd.DataFrame, column: str = "wiki_number") -> pd.DataFrame:
    d = df[df[column].isin(_LABEL_ORDER)].dropna(subset=["population_total"])
    return d[d["population_total"] > 0].copy()


def plot_population_by_label(df: pd.DataFrame, column: str = "wiki_number",
                              kind: str = "box", ax: plt.Axes | None = None) -> plt.Figure:
    """Box nebo violin plot: log(populace) pro všechny 3 labely vedle sebe.

    kind="box" (výchozí) ukazuje medián/kvartily/outliery přehledně a čitelně
    i při silně nevyrovnaných velikostech skupin (1410 vs. 233 vs. 163).
    kind="violin" navíc ukazuje tvar celého rozdělení (např. bimodalitu).
    """
    set_style()
    d = _population_by_label(df, column)
    d["log_pop"] = np.log10(d["population_total"])

    if ax is None:
        fig, ax = plt.subplots(figsize=config.FIGSIZE_SQUARE)
    else:
        fig = ax.figure

    palette = {c: COLORS[c] for c in _LABEL_ORDER}
    plot_fn = sns.violinplot if kind == "violin" else sns.boxplot
    kwargs = {"cut": 0} if kind == "violin" else {"showfliers": True, "width": 0.5}
    plot_fn(data=d, x=column, y="log_pop", order=_LABEL_ORDER, hue=column, palette=palette,
            legend=False, ax=ax, **kwargs)

    ns = d[column].value_counts()
    ax.set_xticklabels([f"{_LABEL_TITLES[c]}\n(n={ns.get(c, 0)})" for c in _LABEL_ORDER])
    ax.set_xlabel("")
    ax.set_ylabel("log₁₀(počet obyvatel)")
    kind_label = "Violin" if kind == "violin" else "Box"
    ax.set_title(f"{kind_label} plot: populace dle labelu ({_SOURCE_LABELS.get(column, column)})")
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_population_density_by_label(df: pd.DataFrame, column: str = "wiki_number",
                                      ax: plt.Axes | None = None) -> plt.Figure:
    """Překrývající se hustoty (KDE) log(populace) pro všechny 3 labely.

    Přímo odpovídá na "je unknown jen dole na ose populace, nebo je
    rozprostřené po celé škále jako plurál/singulár?" — pokud je křivka
    unknown posunutá doleva vůči plurálu/singuláru, jsou malé obce
    nadreprezentované; pokud kopíruje jejich tvar, rozmístění je podobné.
    """
    set_style()
    d = _population_by_label(df, column)
    d["log_pop"] = np.log10(d["population_total"])

    if ax is None:
        fig, ax = plt.subplots(figsize=config.FIGSIZE_WIDE)
    else:
        fig = ax.figure

    for c in _LABEL_ORDER:
        vals = d.loc[d[column] == c, "log_pop"]
        sns.kdeplot(vals, ax=ax, color=COLORS[c], fill=True, alpha=0.25, lw=2,
                    label=f"{_LABEL_TITLES[c]} (n={len(vals)})", cut=0)
        ax.axvline(vals.median(), color=COLORS[c], lw=1.2, ls="--", alpha=0.8)

    ax.set_xlabel("log₁₀(počet obyvatel)  (čárkovaně: mediány)")
    ax.set_ylabel("Hustota")
    ax.set_title(f"Rozdělení populace dle labelu ({_SOURCE_LABELS.get(column, column)})")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_population_ecdf_by_label(df: pd.DataFrame, column: str = "wiki_number",
                                   ax: plt.Axes | None = None) -> plt.Figure:
    """Empirická distribuční funkce (ECDF) log(populace) pro všechny 3 labely.

    ECDF dělá stochastickou dominanci vizuálně přímočarou: pokud křivka
    unknown leží nad ostatními (dřív dosáhne vyšších percentilů při nižší
    populaci), unknown obce jsou systematicky menší — přesně to, co testuje
    Mann-Whitney U v `test_unknown_population()`, jen jako obrázek místo čísla.
    """
    set_style()
    d = _population_by_label(df, column)
    d["log_pop"] = np.log10(d["population_total"])

    if ax is None:
        fig, ax = plt.subplots(figsize=config.FIGSIZE_WIDE)
    else:
        fig = ax.figure

    for c in _LABEL_ORDER:
        vals = d.loc[d[column] == c, "log_pop"]
        sns.ecdfplot(vals, ax=ax, color=COLORS[c], lw=2.2, label=f"{_LABEL_TITLES[c]} (n={len(vals)})")

    ax.set_xlabel("log₁₀(počet obyvatel)")
    ax.set_ylabel("Kumulativní podíl obcí")
    ax.set_title(f"ECDF: populace dle labelu ({_SOURCE_LABELS.get(column, column)})")
    ax.legend(frameon=False, loc="lower right")
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def test_population_by_label(df: pd.DataFrame, column: str = "wiki_number") -> dict:
    """Kruskal-Wallis test (3+ skupin) + párové Mann-Whitney s Bonferroniho korekcí.

    Kruskal-Wallis je nonparametrická obdoba ANOVA — testuje, zda se
    rozdělení populace liší napříč všemi třemi labely najednou (omnibus test),
    bez předpokladu normality (populace je silně šikmá). Pokud vyjde
    významný, následné párové Mann-Whitney testy (singular-plural,
    singular-unknown, plural-unknown) ukážou, KTERÉ dvojice se liší — s
    Bonferroniho korekcí (×3 srovnání), aby se neinflovala chyba I. druhu.
    """
    d = _population_by_label(df, column)
    groups = {c: d.loc[d[column] == c, "population_total"] for c in _LABEL_ORDER}

    h_stat, kw_p = stats.kruskal(*groups.values())

    pairs = [("singular", "plural"), ("singular", "unknown"), ("plural", "unknown")]
    pairwise = {}
    for a, b in pairs:
        u_stat, p = stats.mannwhitneyu(groups[a], groups[b], alternative="two-sided")
        pairwise[f"{a}_vs_{b}"] = {
            "u": u_stat, "p_raw": p, "p_bonferroni": min(p * len(pairs), 1.0),
            "median_a": groups[a].median(), "median_b": groups[b].median(),
        }

    return {
        "n": len(d), "medians": {c: groups[c].median() for c in _LABEL_ORDER},
        "kruskal_h": h_stat, "kruskal_p": kw_p,
        "pairwise": pairwise,
    }

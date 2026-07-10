"""Visualization functions for GeoGram -ice municipality analysis.

All public functions accept a DataFrame and an optional Axes, returning the Axes.
Notebooks should only call these functions — no plotting logic lives in notebooks.

Colour palette: Wong (2011) colorblind-safe palette, standard in academic publishing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------------
# Global theme and palette
# ---------------------------------------------------------------------------

sns.set_theme(style="whitegrid", font_scale=1.05)

# Wong (2011) colorblind-safe palette
COLORS = {
    "plural":        "#0173b2",   # blue
    "singular":      "#de8f05",   # amber
    "unknown":       "#949494",   # grey
    "both":          "#029e73",   # teal
    # IJP agreement categories
    "match":         "#029e73",   # teal  – both sources agree
    "mismatch":      "#d55e00",   # vermillion – sources disagree
    "wiki_only":     "#0173b2",   # blue  – classified only by Wikipedia
    "ujc_only":      "#cc78bc",   # lavender – classified only by IJP
    "both_unknown":  "#949494",   # grey  – unknown in both sources
}

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


def _classified(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["wiki_number"].isin(["singular", "plural", "both"])]


# ---------------------------------------------------------------------------
# 1. Overview: distribuce sg/pl/unknown
# ---------------------------------------------------------------------------

def plot_overview(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Horizontal bar chart with absolute counts and percentages."""
    counts = df["wiki_number"].value_counts()
    order = [c for c in ["plural", "singular", "both", "unknown", "not_found"] if c in counts.index]
    labels = {"plural": "Plurál", "singular": "Singulár", "both": "Obojí",
              "unknown": "Neznámé", "not_found": "Nenalezeno"}

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3))

    total = len(df)
    palette = [COLORS.get(c, "#aaa") for c in order]
    sns.barplot(
        x=[counts[c] for c in order],
        y=[labels.get(c, c) for c in order],
        palette=palette, orient="h", ax=ax,
    )
    for i, c in enumerate(order):
        n = counts[c]
        ax.text(n + 15, i, f"{n} ({n/total:.1%})", va="center", fontsize=9)

    ax.set_xlabel("Počet obcí")
    ax.set_title("Distribuce gramatického čísla obcí -ice\n(zdroj: Wikipedia)")
    ax.set_xlim(0, max(counts) * 1.20)
    return ax


# ---------------------------------------------------------------------------
# 2. Stacked bar po krajích
# ---------------------------------------------------------------------------

def plot_by_region(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Horizontal stacked bar: počet sg/pl per kraj, seřazeno dle % plurálu."""
    d = df[df["wiki_number"].isin(["singular", "plural"])].copy()
    pivot = (
        d.groupby(["region_name", "wiki_number"])
        .size()
        .unstack(fill_value=0)
    )
    pivot["total"] = pivot.sum(axis=1)
    pivot["pct_pl"] = pivot.get("plural", 0) / pivot["total"]
    pivot = pivot.sort_values("pct_pl", ascending=True)
    short = [_REGION_SHORT.get(r, r) for r in pivot.index]

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

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
    ax.set_title("Singulár vs. plurál dle kraje\n(seřazeno vzestupně dle podílu plurálu)")
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    sns.despine(ax=ax, left=True, bottom=False)
    return ax


# ---------------------------------------------------------------------------
# 3. Stacked bar Čechy / Morava / Vysočina
# ---------------------------------------------------------------------------

def plot_by_land(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Stacked bar: sg vs. pl pro Čechy / Morava+Slezsko / Vysočina."""
    d = add_land_column(df)
    d = d[d["wiki_number"].isin(["singular", "plural"])]
    pivot = (
        d.groupby(["land", "wiki_number"])
        .size()
        .unstack(fill_value=0)
    )
    pivot["total"] = pivot.sum(axis=1)
    pivot["pct_pl"] = pivot.get("plural", 0) / pivot["total"]
    pivot = pivot.sort_values("pct_pl")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))

    bottom = np.zeros(len(pivot))
    for cat, label in [("plural", "Plurál"), ("singular", "Singulár")]:
        vals = pivot.get(cat, pd.Series(0, index=pivot.index)).values
        ax.bar(pivot.index, vals, bottom=bottom, label=label, color=COLORS[cat])
        bottom += vals

    for i, (_, row) in enumerate(pivot.iterrows()):
        ax.text(i, row["total"] + 5, f"{row['pct_pl']:.0%}",
                ha="center", fontsize=10, fontweight="bold", color="#222")

    ax.set_ylabel("Počet obcí")
    ax.set_title("Singulár vs. plurál\ndle historické země")
    ax.legend()
    sns.despine(ax=ax)
    return ax


# ---------------------------------------------------------------------------
# 4. Logistická regrese: populace → sg/pl
# ---------------------------------------------------------------------------

def plot_population_logistic(
    df: pd.DataFrame,
    ax: plt.Axes | None = None,
    log_scale: bool = True,
) -> tuple[plt.Axes, object]:
    """Scatter y=0/1 (sg/pl) × populace + logistická regresní křivka.

    Returns (ax, statsmodels LogitResults).
    """
    import statsmodels.api as sm

    d = df[df["wiki_number"].isin(["singular", "plural"])].dropna(subset=["population_total"])
    d = d[d["population_total"] > 0].copy()
    d["y"] = (d["wiki_number"] == "plural").astype(int)

    x_raw = np.log10(d["population_total"]) if log_scale else d["population_total"]
    X = sm.add_constant(x_raw)
    result = sm.Logit(d["y"], X).fit(disp=False)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.03, 0.03, size=len(d))
    for cat in ["singular", "plural"]:
        mask = d["wiki_number"] == cat
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
    ax.set_title(f"Logistická regrese: populace → sg/pl\nβ = {coef:.3f}, p = {pval:.2e}")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    return ax, result


# ---------------------------------------------------------------------------
# 5. Violin plot: distribuce populace dle sg/pl
# ---------------------------------------------------------------------------

def plot_population_violin(
    df: pd.DataFrame,
    ax: plt.Axes | None = None,
    log_scale: bool = True,
) -> plt.Axes:
    """Violin plot: populace_total rozdělená dle wiki_number."""
    d = df[df["wiki_number"].isin(["singular", "plural"])].dropna(subset=["population_total"])
    d = d[d["population_total"] > 0].copy()
    d["pop_plot"] = np.log10(d["population_total"]) if log_scale else d["population_total"]
    ylabel = "log₁₀(počet obyvatel)" if log_scale else "Počet obyvatel"

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    order = ["singular", "plural"]
    sns.violinplot(
        data=d, x="wiki_number", y="pop_plot", order=order,
        palette={k: COLORS[k] for k in order},
        inner="box", density_norm="width", ax=ax,
    )
    ax.set_xticklabels(["Singulár", "Plurál"])
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title("Distribuce populace: singulár vs. plurál")

    for i, cat in enumerate(order):
        med = d.loc[d["wiki_number"] == cat, "pop_plot"].median()
        n = (d["wiki_number"] == cat).sum()
        label = f"n={n}\nmed={10**med:,.0f}" if log_scale else f"n={n}"
        ax.text(i, ax.get_ylim()[1] * 0.99, label, ha="center", va="top", fontsize=8)

    sns.despine(ax=ax)
    return ax


# ---------------------------------------------------------------------------
# 6. Unknown analýza: velikost nerozpoznaných obcí
# ---------------------------------------------------------------------------

def plot_missing_analysis(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Histogramy: populace klasifikovaných vs. neznámých obcí."""
    classified = df[df["wiki_number"].isin(["singular", "plural"]) & (df["population_total"] > 0)]
    unknown = df[(df["wiki_number"] == "unknown") & (df["population_total"] > 0)]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

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
    ax.set_title("Velikost obcí: klasifikované vs. nerozpoznané")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    return ax


# ---------------------------------------------------------------------------
# 7. Statistické testy
# ---------------------------------------------------------------------------

def chi2_land_test(df: pd.DataFrame) -> dict:
    """Chi² test: je distribuce sg/pl nezávislá na historické zemi?"""
    d = add_land_column(df)[lambda x: x["wiki_number"].isin(["singular", "plural"])]
    ct = pd.crosstab(d["land"], d["wiki_number"])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    return {"chi2": chi2, "p": p, "dof": dof, "contingency": ct}


def chi2_region_test(df: pd.DataFrame) -> dict:
    """Chi² test: distribuce sg/pl vs. kraj."""
    d = df[df["wiki_number"].isin(["singular", "plural"])]
    ct = pd.crosstab(d["region_name"], d["wiki_number"])
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


def plot_agreement_summary(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Sloupcový graf kategorií shody Wikipedia vs. IJP."""
    if "agreement" not in df.columns:
        df = classify_agreement(df)

    counts = df["agreement"].value_counts()
    order = [c for c in _AGREE_ORDER if c in counts.index]
    total = len(df)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

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
    return ax


def plot_agreement_heatmap(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Matice záměn (Wikipedia × IJP) jako seaborn heatmap."""
    if "agreement" not in df.columns:
        df = classify_agreement(df)

    KNOWN = ["singular", "plural", "both"]
    d = df[df["wiki_number"].isin(KNOWN) & df["ujc_number"].isin(KNOWN)]
    ct = pd.crosstab(
        d["wiki_number"].rename("Wikipedia"),
        d["ujc_number"].rename("IJP ÚJČ"),
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        ct, annot=True, fmt="d", cmap="Blues",
        linewidths=0.5, ax=ax, cbar_kws={"label": "Počet obcí"},
    )
    ax.set_title("Matice záměn: Wikipedia vs. IJP ÚJČ\n(jen klasifikované v obou)")
    return ax


def plot_agreement_by_region(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Stacked bar: kategorie shody po krajích."""
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
        _, ax = plt.subplots(figsize=(10, 5.5))

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
    return ax


def plot_mismatch_details(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Bar chart neshodujících se obcí: co říká Wikipedia vs. IJP."""
    if "agreement" not in df.columns:
        df = classify_agreement(df)

    mis = df[df["agreement"] == "mismatch"].copy()
    if mis.empty:
        if ax is None:
            _, ax = plt.subplots()
        ax.text(0.5, 0.5, "Žádné neshody", ha="center", va="center", transform=ax.transAxes)
        return ax

    label = mis["wiki_number"] + " → " + mis["ujc_number"]
    counts = label.value_counts()

    if ax is None:
        _, ax = plt.subplots(figsize=(6, max(3, len(counts) * 0.5)))

    sns.barplot(x=counts.values, y=counts.index, color=COLORS["mismatch"], ax=ax)
    ax.set_xlabel("Počet obcí")
    ax.set_title(f"Neshody Wikipedia → IJP (n={len(mis)})")
    sns.despine(ax=ax, left=True)
    return ax


# ---------------------------------------------------------------------------
# 9. Interaktivní mapa (vyžaduje lat/lon sloupce z Wikipedia coordinates API)
# ---------------------------------------------------------------------------

def plot_map_folium(df: pd.DataFrame, lat_col: str = "latitude", lon_col: str = "longitude"):
    """Folium mapa s body obcí obarvených dle wiki_number.

    Vyžaduje sloupce latitude/longitude (z municipalities_ice_integrated.csv).
    Vrací folium.Map — v notebooku display(m), nebo m.save('mapa.html').
    """
    import folium

    d = df.dropna(subset=[lat_col, lon_col])
    m = folium.Map(location=[49.8, 15.5], zoom_start=7, tiles="CartoDB positron")

    for _, row in d.iterrows():
        color = COLORS.get(row.get("wiki_number", "unknown"), "#aaa")
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=4, color=color, fill=True, fill_color=color, fill_opacity=0.75,
            popup=(f"<b>{row.get('name','')}</b><br>"
                   f"{row.get('wiki_number','')} | {int(row.get('population_total',0))} obyv."),
        ).add_to(m)

    legend = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:999;background:white;
                padding:10px 14px;border:1px solid #ccc;border-radius:6px;font-size:13px;
                box-shadow:2px 2px 6px rgba(0,0,0,0.15)">
      <b>Gramatické číslo</b><br>
      <span style="color:#0173b2;">&#9679;</span> Plurál<br>
      <span style="color:#de8f05;">&#9679;</span> Singulár<br>
      <span style="color:#949494;">&#9679;</span> Neznámé
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    return m

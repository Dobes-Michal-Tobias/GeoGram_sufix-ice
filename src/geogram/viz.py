"""Visualization functions for GeoGram -ice municipality analysis.

All public functions build and return their own `matplotlib.Figure` (unless an
`ax` is passed in for composition into a larger grid). Notebooks should only
call these functions and save the result via `save_fig()` — no plotting logic
or styling decisions live in notebooks.

Barevná paleta a rozměry grafů se řídí `src/geogram/config.py` (3 sémantické
role: PRIMARY / ACCENT / NEUTRAL), stejná konvence jako v PeriodSim.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

from . import config
from .morphology import SUFFIX_ORDER, SUFFIX_TITLES, suffix_ending

# ---------------------------------------------------------------------------
# Global theme and palette
# ---------------------------------------------------------------------------


_fonts_registered = False


def _register_fonts() -> None:
    """Zaregistruje lokální IBM Plex .ttf (assets/fonts/) do matplotlib font manageru.

    Fonty nejsou nainstalované jako OS fonty (stejně jako na webu jsou jen přes
    Google Fonts <link>) — matplotlib je bez tohohle kroku nenajde a tiše
    spadne na DejaVu Sans, což je přesně ten "šev" mezi grafy a zbytkem webu.
    """
    global _fonts_registered
    if _fonts_registered:
        return
    import matplotlib.font_manager as fm
    for path in config.FONTS_DIR.glob("*.ttf"):
        fm.fontManager.addfont(str(path))
    _fonts_registered = True


_current_bg = "white"


def set_style(bg: str | None = None) -> None:
    """Nastaví seaborn theme + house style (fonty, mřížka, barvy) dle config.py.

    Volej jednou na začátku notebooku (a je idempotentní — každá plot_* funkce
    ji volá znovu pro sebe, aby fungovala i nezávisle na notebooku).

    `bg` je "sticky": `bg=None` (výchozí) znamená "nech to, co bylo nastaveno
    naposledy (nebo white, pokud nic)". Kdyby `bg=None` resetoval na white při
    každém volání, `compose_poster(bg="cream")` by se hned rozbil — každá
    plot_* funkce si totiž volá `set_style()` bez argumentů znovu pro sebe.
    Nastav `bg="cream"`/`bg="white"` explicitně jen tam, kde se pozadí
    skutečně rozhoduje (jednou na začátku notebooku, nebo uvnitř
    `compose_poster()`).
    """
    global _current_bg
    if bg is not None:
        _current_bg = bg
    bg = _current_bg

    _register_fonts()
    sns.set_theme(style=config.STYLE, palette=config.PALETTE, font_scale=config.FONT_SCALE)

    bg_color = config.BG_WHITE if bg == "white" else config.BG_CREAM
    grid_color = config.GRID_ON_WHITE if bg == "white" else config.GRID_ON_CREAM

    plt.rcParams.update({
        "font.family": config.FONT_SANS,
        "text.color": config.INK_COLOR,
        "axes.titlecolor": config.INK_COLOR,
        "axes.labelcolor": config.INK_COLOR,
        "xtick.color": config.MUTED_COLOR,
        "ytick.color": config.MUTED_COLOR,
        "xtick.labelcolor": config.MUTED_COLOR,
        "ytick.labelcolor": config.MUTED_COLOR,
        "grid.color": grid_color,
        "grid.linewidth": 0.9,
        "axes.edgecolor": grid_color,
        "legend.frameon": False,
        "legend.labelcolor": config.INK_COLOR,
        "figure.facecolor": bg_color,
        "axes.facecolor": bg_color,
        "savefig.facecolor": bg_color,
    })


def _lollipop_height(n: int) -> float:
    """Adaptivní výška horizontálního grafu podle počtu kategorií.

    3 kategorie nepotřebují stejnou plochu jako 15 — bez tohohle byly krátké
    srovnávací grafy (Čechy/Morava, 3 sufixální skupiny...) opticky stejně
    "těžké" jako grafy s desítkami řádků.
    """
    return max(config.FIGSIZE_MIN_H, config.ROW_HEIGHT * n + 1.0)


def _hlollipop(ax: plt.Axes, labels: list[str], values: list[float],
               color: str | None = None, colors: list[str] | None = None) -> None:
    """Lollipop varianta horizontálního bar chartu: tenká čára + tečka na konci.

    Nahrazuje `ax.barh`/`sns.barplot(orient="h")` všude, kde graf nese jen
    JEDNU hodnotu na kategorii (řazení/magnitude) — plný sloupec tam nic
    navíc neříká, jen zabírá barevnou plochu. Pro skládané/stacked bary
    (part-to-whole) lollipop nedává smysl, tam zůstává `ax.bar`/`ax.barh`
    (jen tenčí, viz `config.BAR_WIDTH`).

    `color` = jedna barva pro všechny položky, `colors` = barva na položku
    (stejná délka jako labels/values) — dej právě jeden z nich.
    """
    y = np.arange(len(labels))
    cols = colors if colors is not None else [color] * len(labels)
    for yi, v, c in zip(y, values, cols):
        ax.hlines(yi, 0, v, color=c, linewidth=config.LOLLIPOP_LW)
        ax.plot(v, yi, "o", color=c, markersize=config.LOLLIPOP_MS, markeredgewidth=0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.7, len(labels) - 0.3)


def _share_lollipop(labels: list[str], pct: np.ndarray, ns: np.ndarray,
                     color, ax: plt.Axes | None = None, figsize: tuple | None = None,
                     xlabel: str = "Podíl plurálu") -> plt.Figure:
    """Sdílený lollipop pro "podíl plurálu podle skupiny" grafy.

    Nahrazuje skládaný (stacked) bar chart všude, kde je skutečná informace
    JEDNO číslo na skupinu (% plurálu) — stacked bar k tomu navíc kreslí
    absolutní výšku, která tady nic nenese (celkový počet je vedle v textu
    `n=`). `color` = jedna barva pro všechny, nebo seznam (barva na skupinu).
    """
    set_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (6, _lollipop_height(len(labels))))
    else:
        fig = ax.figure

    colors = color if isinstance(color, list) else [color] * len(labels)
    _hlollipop(ax, labels, list(pct), colors=colors)
    for i, (p, n) in enumerate(zip(pct, ns)):
        ax.text(p + 0.02, i, f"{p:.0%}  (n={int(n)})", va="center", fontsize=9)

    ax.set_xlabel(xlabel)
    ax.set_xlim(0, 1.15)
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    return fig


def _binned_logistic(x_raw: pd.Series, y: pd.Series, ax: plt.Axes,
                      dot_color: str | None = None, n_bins: int = 10) -> object:
    """Fitne logistickou regresi a nakreslí ji jako binned-proportion pohled.

    Nahrazuje syrový jittered scatter 0/1 bodů (ten na ploše grafu skoro nic
    neukazuje — stovky překrývajících se bodů natlačené na dvou vodorovných
    pruzích u y=0/1, efekt schovaný v hustotě, ne ve tvaru). Místo toho:
    pozorovaný podíl v kvantilových binech osy x (velikost bodu ~ n v binu)
    + fitovaná křivka s 95% pásem spolehlivosti. Stejná statistika
    (`result`), jen čitelnější obrázek.

    Vrací fitnutý `statsmodels` LogitResults (stejné rozhraní jako předtím).
    """
    import statsmodels.api as sm

    X = sm.add_constant(x_raw)
    result = sm.Logit(y, X).fit(disp=False)

    dot_color = dot_color or config.PRIMARY_COLOR
    n_bins = min(n_bins, x_raw.nunique())
    bins = pd.qcut(x_raw, n_bins, duplicates="drop")
    binned = (
        pd.DataFrame({"x": x_raw.values, "y": y.values})
        .groupby(bins.values, observed=True)
        .agg(x_mean=("x", "mean"), y_mean=("y", "mean"), n=("y", "size"))
    )
    sizes = 18 + 90 * (binned["n"] / binned["n"].max())
    ax.scatter(binned["x_mean"], binned["y_mean"], s=sizes, color=dot_color,
               alpha=0.9, linewidths=0, zorder=3, label="Pozorovaný podíl (bin)")

    x_line = np.linspace(x_raw.min(), x_raw.max(), 200)
    pred = result.get_prediction(sm.add_constant(x_line)).summary_frame(alpha=0.05)
    ax.fill_between(x_line, pred["ci_lower"], pred["ci_upper"],
                     color=dot_color, alpha=0.15, linewidth=0, label="95% CI")
    ax.plot(x_line, pred["predicted"], color=config.INK_COLOR, lw=2, label="Logistická regrese")
    ax.set_ylim(-0.03, 1.03)
    return result


def save_fig(fig: plt.Figure, filename: str, subdir: Path | None = None, fmt: str = "svg") -> Path:
    """Uloží figuru do assets/img/geogram/ (pro publikaci na webu).

    Výchozí formát je SVG — vektor, ostrý na jakémkoli zoomu/DPI, na rozdíl
    od PNG nikdy nerozostří text oproti okolnímu HTML. Použij `fmt="png"`
    jen pro grafy s tisíci překrývajících se bodů (např. LISA mapa) — tam by
    SVG kreslilo každý bod jako samostatný element a zbytečně nafouklo
    velikost souboru.
    """
    target_dir = subdir or config.ASSETS_IMG_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{filename}.{fmt}"
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


# Binární Čechy/Morava rozdělení pro `plot_land_binary()` — na rozdíl od
# `add_land_column()` (celé kraje) rozpouští kraj Vysočina podle OKRESU na
# historickou zemskou hranici, ne podle správního kraje. Zlínský i
# Moravskoslezský kraj počítají jako Morava (slezská část se sem pro
# dvouskupinové srovnání slučuje).
_LAND_BINARY_MAP = {
    "Středočeský kraj": "Čechy", "Jihočeský kraj": "Čechy", "Plzeňský kraj": "Čechy",
    "Karlovarský kraj": "Čechy", "Ústecký kraj": "Čechy", "Liberecký kraj": "Čechy",
    "Královéhradecký kraj": "Čechy", "Pardubický kraj": "Čechy",
    "Jihomoravský kraj": "Morava", "Olomoucký kraj": "Morava",
    "Zlínský kraj": "Morava", "Moravskoslezský kraj": "Morava",
}
_VYSOCINA_DISTRICTS_CECHY = {"Havlíčkův Brod", "Pelhřimov"}


def add_land_column_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'land_binary' column (Čechy / Morava), s Vysočinou rozpuštěnou
    podle okresu (Havlíčkobrodsko a Pelhřimovsko → Čechy, zbytek → Morava) —
    přesnější aproximace historické zemské hranice než celý kraj najednou.
    """
    out = df.copy()
    land = out["region_name"].map(_LAND_BINARY_MAP)
    is_vysocina = out["region_name"] == "Kraj Vysočina"
    land = land.where(~is_vysocina, out["district_name"].isin(_VYSOCINA_DISTRICTS_CECHY).map({True: "Čechy", False: "Morava"}))
    out["land_binary"] = land.fillna("Neznámý")
    return out


# ---------------------------------------------------------------------------
# 1. Overview: distribuce sg/pl/unknown
# ---------------------------------------------------------------------------

def plot_overview(df: pd.DataFrame, column: str = "wiki_number", ax: plt.Axes | None = None) -> plt.Figure:
    """Lollipop chart (tenká čára + tečka) s absolutními počty a procenty.

    column: "wiki_number" (default) nebo "ujc_number" — obě mají stejnou
    sadu kategorií (singular/plural/both/unknown/not_found/error).
    """
    set_style()
    counts = df[column].value_counts()
    order = [c for c in ["plural", "singular", "both", "unknown", "not_found", "error"] if c in counts.index]
    labels = {"plural": "Plurál", "singular": "Singulár", "both": "Obojí",
              "unknown": "Neznámé", "not_found": "Nenalezeno", "error": "Chyba"}

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, _lollipop_height(len(order))))
    else:
        fig = ax.figure

    total = len(df)
    values = [counts[c] for c in order]
    colors = [COLORS.get(c, config.NEUTRAL_COLOR) for c in order]
    _hlollipop(ax, [labels.get(c, c) for c in order], values, colors=colors)
    for i, c in enumerate(order):
        n = counts[c]
        ax.text(n + max(values) * 0.02, i, f"{n} ({n/total:.1%})", va="center", fontsize=9)

    ax.set_xlabel("Počet obcí")
    ax.set_xlim(0, max(counts) * 1.20)
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    return fig


_REGION_TOTAL_CATS = ["plural", "singular", "unknown"]
_REGION_TOTAL_LABELS = {"plural": "Plurál", "singular": "Singulár", "unknown": "Neznámé"}
_REGION_TOTAL_TEXT_COLOR = {"plural": "white", "singular": None, "unknown": None}  # None → config.INK_COLOR


def plot_region_totals(df: pd.DataFrame, column: str = "wiki_number", ax: plt.Axes | None = None) -> plt.Figure:
    """Stacked bar: rozklad plurál/singulár/neznámé podle kraje, seřazeno
    sestupně podle celkového počtu -ice obcí.

    Na rozdíl od `plot_by_region()` (jen sg/pl, řazeno podle % plurálu — jiná
    otázka) tenhle graf odpovídá na "kde je -ice obcí nejvíc" a záměrně
    zahrnuje i "neznámé" — tady jde o objem dat, ne o gramatické číslo, takže
    není důvod tu kategorii schovávat. Číslo v segmentu se píše, jen když je
    segment dost široký, aby se do něj vešlo čitelně; celkový součet je vždy
    u konce sloupce.
    """
    set_style()
    pivot = df.groupby(["region_name", column]).size().unstack(fill_value=0)
    for c in _REGION_TOTAL_CATS:
        if c not in pivot.columns:
            pivot[c] = 0
    pivot["total"] = pivot[_REGION_TOTAL_CATS].sum(axis=1)
    pivot = pivot.sort_values("total")  # ascending: barh kreslí první položku dole → největší nahoře
    short = [_REGION_SHORT.get(r, r) for r in pivot.index]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, _lollipop_height(len(pivot))))
    else:
        fig = ax.figure

    bottom = np.zeros(len(pivot))
    min_label_width = pivot["total"].max() * 0.05
    for cat in _REGION_TOTAL_CATS:
        vals = pivot[cat].values
        ax.barh(short, vals, left=bottom, color=COLORS[cat], label=_REGION_TOTAL_LABELS[cat],
                 height=config.BAR_WIDTH)
        text_color = _REGION_TOTAL_TEXT_COLOR[cat] or config.INK_COLOR
        for i, v in enumerate(vals):
            if v >= min_label_width:
                ax.text(bottom[i] + v / 2, i, str(int(v)), ha="center", va="center",
                         fontsize=7.5, color=text_color)
        bottom += vals

    for i, total in enumerate(pivot["total"].values):
        ax.text(total + pivot["total"].max() * 0.02, i, str(int(total)),
                 va="center", fontsize=9, fontweight="600")

    ax.set_xlabel("Počet obcí -ice")
    ax.set_xlim(0, pivot["total"].max() * 1.15)
    ax.legend(frameon=False, loc="lower right")
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
        fig, ax = plt.subplots(figsize=(config.FIGSIZE_WIDE[0], _lollipop_height(len(pivot))))
    else:
        fig = ax.figure

    bottom = np.zeros(len(pivot))
    for cat, label in [("plural", "Plurál"), ("singular", "Singulár")]:
        vals = pivot.get(cat, pd.Series(0, index=pivot.index)).values
        ax.barh(short, vals, left=bottom, label=label, color=COLORS[cat], height=config.BAR_WIDTH)
        bottom += vals

    # % label on each bar
    for i, (_, row) in enumerate(pivot.iterrows()):
        ax.text(row["total"] + 2, i, f"{row['pct_pl']:.0%}",
                va="center", fontsize=8, color="#444")

    ax.set_xlabel("Počet obcí")
    ax.legend(loc="lower right", frameon=False)
    ax.invert_yaxis()
    sns.despine(ax=ax, left=True, bottom=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Stacked bar Čechy / Morava / Vysočina
# ---------------------------------------------------------------------------

def plot_by_land(df: pd.DataFrame, column: str = "wiki_number", ax: plt.Axes | None = None) -> plt.Figure:
    """Lollipop: podíl plurálu pro Čechy / Morava+Slezsko / Vysočina, seřazeno."""
    d = add_land_column(df)
    d = d[d[column].isin(["singular", "plural"])]
    pivot = d.groupby(["land", column]).size().unstack(fill_value=0)
    pivot["total"] = pivot.sum(axis=1)
    pivot["pct_pl"] = pivot.get("plural", 0) / pivot["total"]
    pivot = pivot.sort_values("pct_pl")

    return _share_lollipop(list(pivot.index), pivot["pct_pl"].values, pivot["total"].values,
                            config.PRIMARY_COLOR, ax=ax, figsize=(6, _lollipop_height(len(pivot))))


def plot_land_binary(df: pd.DataFrame, column: str = "wiki_number", ax: plt.Axes | None = None) -> plt.Figure:
    """Lollipop: podíl plurálu Čechy vs. Morava (binární historická hranice).

    Na rozdíl od `plot_by_land()` (4 skupiny podle celých krajů) je tu kraj
    Vysočina rozpuštěný podle okresu na skutečnou zemskou hranici — viz
    `add_land_column_binary()`.
    """
    d = add_land_column_binary(df)
    d = d[d[column].isin(["singular", "plural"])]
    pivot = d.groupby(["land_binary", column]).size().unstack(fill_value=0)
    pivot["total"] = pivot.sum(axis=1)
    pivot["pct_pl"] = pivot.get("plural", 0) / pivot["total"]
    pivot = pivot.sort_values("pct_pl")

    return _share_lollipop(list(pivot.index), pivot["pct_pl"].values, pivot["total"].values,
                            config.PRIMARY_COLOR, ax=ax, figsize=(6, _lollipop_height(len(pivot))))


# ---------------------------------------------------------------------------
# 4. Logistická regrese: populace → sg/pl
# ---------------------------------------------------------------------------

def plot_population_logistic(
    df: pd.DataFrame,
    column: str = "wiki_number",
    ax: plt.Axes | None = None,
    log_scale: bool = True,
) -> tuple[plt.Figure, object]:
    """Binned-proportion pohled na logistickou regresi populace → sg/pl.

    Returns (fig, statsmodels LogitResults).
    """
    set_style()
    d = df[df[column].isin(["singular", "plural"])].dropna(subset=["population_total"])
    d = d[d["population_total"] > 0].copy()
    d["y"] = (d[column] == "plural").astype(int)
    x_raw = np.log10(d["population_total"]) if log_scale else d["population_total"]

    if ax is None:
        fig, ax = plt.subplots(figsize=config.FIGSIZE_DEFAULT)
    else:
        fig = ax.figure

    result = _binned_logistic(x_raw, d["y"], ax, dot_color=config.PRIMARY_COLOR)

    ax.set_xlabel("log₁₀(počet obyvatel)" if log_scale else "Počet obyvatel")
    ax.set_ylabel("Podíl plurálu")
    ax.legend(frameon=False, loc="lower right")
    sns.despine(ax=ax, left=True)
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

    for i, cat in enumerate(order):
        med = d.loc[d[column] == cat, "pop_plot"].median()
        n = (d[column] == cat).sum()
        label = f"n={n}\nmed={10**med:,.0f}" if log_scale else f"n={n}"
        ax.text(i, ax.get_ylim()[1] * 0.99, label, ha="center", va="top", fontsize=8)

    sns.despine(ax=ax, left=True)
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
    ax.legend(frameon=False)
    sns.despine(ax=ax, left=True)
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
        fig, ax = plt.subplots(figsize=(8, _lollipop_height(len(order))))
    else:
        fig = ax.figure

    values = [counts[c] for c in order]
    colors = [COLORS[c] for c in order]
    _hlollipop(ax, [_AGREE_LABELS[c] for c in order], values, colors=colors)
    for i, c in enumerate(order):
        n = counts[c]
        ax.text(n + max(values) * 0.02, i, f"{n} ({n/total:.1%})", va="center", fontsize=9)

    ax.set_xlabel("Počet obcí")
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
        fig, ax = plt.subplots(figsize=(config.FIGSIZE_WIDE[0], _lollipop_height(len(pivot))))
    else:
        fig = ax.figure

    bottom = np.zeros(len(pivot))
    patches = []
    for cat in order_cols:
        vals = pivot[cat].values
        ax.barh(short, vals, left=bottom, color=COLORS[cat], label=_AGREE_LABELS[cat], height=config.BAR_WIDTH)
        patches.append(mpatches.Patch(color=COLORS[cat], label=_AGREE_LABELS[cat]))
        bottom += vals

    ax.set_xlabel("Počet obcí")
    ax.legend(handles=patches, loc="lower right", fontsize=8, frameon=False)
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
    label_all = mis["wiki_number"] + " → " + mis["ujc_number"] if not mis.empty else pd.Series(dtype=str)
    n_rows = label_all.nunique() if not mis.empty else 1
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, _lollipop_height(n_rows)) if not mis.empty else (6, 3))
    else:
        fig = ax.figure

    if mis.empty:
        ax.text(0.5, 0.5, "Žádné neshody", ha="center", va="center", transform=ax.transAxes)
        return fig

    counts = label_all.value_counts()
    _hlollipop(ax, list(counts.index), list(counts.values), color=COLORS["mismatch"])
    ax.set_xlabel("Počet obcí")
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
#   radius/fill_opacity zmenšeny a stroke je bílý (ne stejná barva jako
#   výplň) — na 1806 bodech na mapě celé ČR se hustě obydlené oblasti jinak
#   slily do neprůhledných modrých/tmavých "kaněk" bez rozlišitelných
#   jednotlivých bodů. Bílý 0.5px obrys dá i slepeným bodům viditelnou
#   hranici, nižší opacity necha proscvítat překryv (víc bodů na místě =
#   o trochu tmavší skvrna, ne rovnou plná barva).
_MAP_STYLE = {
    "plural":    {"color": config.PRIMARY_COLOR,          "radius": 3, "fill": True,  "fill_opacity": 0.65, "weight": 0.5, "stroke_color": "#ffffff"},
    "singular":  {"color": config.CATEGORICAL_PALETTE[2], "radius": 3, "fill": True,  "fill_opacity": 0.70, "weight": 0.5, "stroke_color": "#ffffff"},
    "both":      {"color": config.CATEGORICAL_PALETTE[2], "radius": 3, "fill": True,  "fill_opacity": 0.70, "weight": 0.5, "stroke_color": "#ffffff"},
}
_MAP_STYLE_DEFAULT = {
    "color": config.NEUTRAL_COLOR, "radius": 3, "fill": False, "fill_opacity": 0.0, "weight": 1.2, "stroke_color": config.NEUTRAL_COLOR,
}

# Overrides Leaflet's own chrome to match styles.css: no shadows, no rounded
# corners, thin 1px "--rule"-colored borders, IBM Plex everywhere text
# appears (popups, zoom control, attribution) instead of the browser default.
_MAP_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  .leaflet-container { font-family: 'IBM Plex Sans', Arial, sans-serif; }
  .leaflet-popup-content-wrapper { border-radius: 0; box-shadow: none; border: 1px solid #d8d6ce; }
  .leaflet-popup-content { font-family: 'IBM Plex Sans', Arial, sans-serif; font-size: 13px; color: #14171a; margin: .7em .9em; line-height: 1.5; }
  .leaflet-popup-content b { font-weight: 600; }
  .leaflet-popup-tip-container { display: none; }
  .leaflet-bar { border-radius: 0; box-shadow: none; border: 1px solid #d8d6ce; }
  .leaflet-bar a { border-radius: 0 !important; font-family: 'IBM Plex Mono', monospace; color: #14171a; background-color: #ffffff; }
  .leaflet-bar a:hover { background-color: #f5f4f0; color: #c1440e; }
  .leaflet-control-attribution { font-family: 'IBM Plex Mono', monospace; font-size: 10px; background: rgba(255,255,255,.85); border-top: 1px solid #d8d6ce; padding: 0 .4em; }
  .leaflet-container a { color: #c1440e; }
  .leaflet-container a:hover { color: #c1440e; text-decoration: underline; }
</style>
"""


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
    # "CartoDB positron" (folium's built-in preset) now renders a big
    # "API KEY REQUIRED" watermark over the whole map — Carto restricted
    # anonymous access to their basemap tiles. Esri's World Light Gray Canvas
    # is the closest free, no-key equivalent (same "trivially clean
    # cartography" look — neutral light gray land, no satellite/terrain).
    m = folium.Map(
        location=[49.8, 15.5], zoom_start=7,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ",
    )

    # Leaflet's own chrome (popups, zoom buttons, attribution) is a separate
    # standalone HTML document loaded in an <iframe> — it doesn't inherit the
    # parent page's <link> or CSS, so it needs its own font import and its
    # own override of Leaflet's defaults (rounded corners + drop shadows,
    # both against this site's "no shadows, no rounded corners, thin 1px
    # lines" rule) to actually look like part of michaltobiasdobes.cz.
    m.get_root().header.add_child(folium.Element(_MAP_CSS))

    for _, row in d.iterrows():
        wiki_number = row.get("wiki_number", "unknown")
        style = _MAP_STYLE.get(wiki_number, _MAP_STYLE_DEFAULT)
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=style["radius"], color=style["stroke_color"], weight=style["weight"],
            fill=style["fill"], fill_color=style["color"], fill_opacity=style["fill_opacity"],
            popup=(f"<b>{row.get('name','')}</b><br>"
                   f"{wiki_number} | {int(row.get('population_total',0))} obyv."),
        ).add_to(m)

    # Styl legendy sladěný s designem michaltobiasdobes.cz: bílé pozadí, tenká
    # 1px linka (žádný stín, žádné zaoblené rohy), IBM Plex Sans/Mono — mono
    # verzálky pro popisek, sans pro kategorie (viz styles.css .section-label / .tag).
    legend = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:999;background:#ffffff;
                padding:.7em 1em;border:1px solid #d8d6ce;font-size:13px;
                font-family:'IBM Plex Sans',Arial,sans-serif;color:#14171a;line-height:1.6;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:.04em;
                  text-transform:uppercase;color:#6b6a63;margin-bottom:.35em;">Gramatické číslo</div>
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
    ax.legend(frameon=False)
    sns.despine(ax=ax, left=True)
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


def plot_lisa_map(gdf: pd.DataFrame, alpha: float = 0.05, boundary=None,
                   ax: plt.Axes | None = None) -> plt.Figure:
    """Mapa significantních LISA shluků (výstup `moran_local_test`).

    HH (plurál obklopený plurálem) a LL (singulár obklopený singulárem) jsou
    "typické" shluky — plná PRIMARY/NEUTRAL barva. HL/LH (obec jiného labelu
    než sousedi) jsou prostorové anomálie — ACCENT, stejná sémantika jako
    jinde v projektu (skutečná neshoda/výjimka, ne jen odlišení). Body s
    p >= alpha (nevýznamné) jsou tenké šedé tečky v pozadí.

    `boundary`: volitelný `geopandas.GeoDataFrame` s hranicí ČR
    (`data.load_cz_boundary()`) — vykreslí se jako tenká hairline POD body
    (žádná výplň/barva země, viz STYLE_GUIDE.md §7 "trivially clean" mapy),
    ať je jasné, kde končí republika a kde je jen okraj shluku dat. Reprojektuje
    se automaticky do CRS `gdf` (EPSG:5514), pokud ještě není.
    """
    set_style()
    _LISA_LABELS = {1: "HH (plurál shluk)", 2: "LH (anomálie)", 3: "LL (singulár shluk)", 4: "HL (anomálie)"}
    _LISA_COLORS = {1: config.PRIMARY_COLOR, 2: config.ACCENT_COLOR,
                     3: config.NEUTRAL_COLOR, 4: config.ACCENT_COLOR}

    if ax is None:
        fig, ax = plt.subplots(figsize=config.FIGSIZE_SQUARE)
    else:
        fig = ax.figure

    if boundary is not None:
        b = boundary if boundary.crs == gdf.crs else boundary.to_crs(gdf.crs)
        b.boundary.plot(ax=ax, color=config.MUTED_COLOR, linewidth=0.8, alpha=0.6, zorder=1)

    sig = gdf[gdf["lisa_p"] < alpha]
    nonsig = gdf[gdf["lisa_p"] >= alpha]

    ax.scatter(nonsig.geometry.x, nonsig.geometry.y, color="#ccc", s=6, alpha=0.4, linewidths=0,
               zorder=2, label=f"Nevýznamné (p ≥ {alpha})")
    for q, label in _LISA_LABELS.items():
        m = sig[sig["lisa_q"] == q]
        if len(m):
            ax.scatter(m.geometry.x, m.geometry.y, color=_LISA_COLORS[q], s=22, alpha=0.85,
                       linewidths=0, zorder=3, label=f"{label} (n={len(m)})")

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
    """Binned-proportion pohled na logistickou regresi populace → 'unknown' label."""
    set_style()
    d = df[df[column].isin(["singular", "plural", "unknown"])].dropna(subset=["population_total"])
    d = d[d["population_total"] > 0].copy()
    d["is_unknown"] = (d[column] == "unknown").astype(int)
    x_raw = np.log10(d["population_total"])

    if ax is None:
        fig, ax = plt.subplots(figsize=config.FIGSIZE_DEFAULT)
    else:
        fig = ax.figure

    result = _binned_logistic(x_raw, d["is_unknown"], ax, dot_color=COLORS["unknown"])

    ax.set_xlabel("log₁₀(počet obyvatel)")
    ax.set_ylabel("Podíl s labelem 'unknown'")
    ax.legend(frameon=False, loc="upper right")
    sns.despine(ax=ax, left=True)
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
    sns.despine(ax=ax, left=True)
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
    ax.legend(frameon=False)
    sns.despine(ax=ax, left=True)
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
    ax.legend(frameon=False, loc="lower right")
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    return fig


def _kruskal_pairwise(groups: dict[str, pd.Series]) -> dict:
    """Kruskal-Wallis omnibus test (3+ skupin) + párové Mann-Whitney s Bonferroniho korekcí.

    Sdílená implementace pro libovolnou spojitou proměnnou srovnávanou napříč
    labely singular/plural/unknown (populace v `test_population_by_label`,
    prostorové skóre v `test_neighbor_share_by_label`). Kruskal-Wallis je
    nonparametrická obdoba ANOVA — netestuje žádný konkrétní pár, jen "liší
    se rozdělení napříč skupinami vůbec". Teprve pokud vyjde významný, mají
    smysl párové Mann-Whitney testy — s Bonferroniho korekcí (×počet párů),
    aby násobné testování neinflovalo chybu I. druhu.
    """
    order = [c for c in _LABEL_ORDER if c in groups]
    h_stat, kw_p = stats.kruskal(*[groups[c] for c in order])

    pairs = [(a, b) for i, a in enumerate(order) for b in order[i + 1:]]
    pairwise = {}
    for a, b in pairs:
        u_stat, p = stats.mannwhitneyu(groups[a], groups[b], alternative="two-sided")
        pairwise[f"{a}_vs_{b}"] = {
            "u": u_stat, "p_raw": p, "p_bonferroni": min(p * len(pairs), 1.0),
            "median_a": groups[a].median(), "median_b": groups[b].median(),
        }

    return {
        "n": sum(len(g) for g in groups.values()),
        "medians": {c: groups[c].median() for c in order},
        "kruskal_h": h_stat, "kruskal_p": kw_p,
        "pairwise": pairwise,
    }


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
    return _kruskal_pairwise(groups)


# ---------------------------------------------------------------------------
# 11. Je unknown prostorově spojené s regiony, kde převažuje singulár?
#     (notebook 10) — propojuje spatial segregaci z nb08 s otázkou
#     "je unknown náhodné, nebo se kupí tam, kde je singulár?" z nb09.
# ---------------------------------------------------------------------------

def chi2_unknown_land_test(df: pd.DataFrame, column: str = "wiki_number") -> dict:
    """Chi² test: liší se podíl 'unknown' mezi Čechy/Morava/Morava+Slezsko/Vysočina?

    Rychlý, hrubozrnný test na úrovni regionu — analogie `chi2_land_test()`,
    ale místo sg/pl testuje unknown vs. klasifikováno (sg+pl sloučené).
    """
    d = add_land_column(df)
    d = d[d[column].isin(["singular", "plural", "unknown"])].copy()
    d["status"] = np.where(d[column] == "unknown", "unknown", "klasifikováno")
    ct = pd.crosstab(d["land"], d["status"])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    return {"chi2": chi2, "p": p, "dof": dof, "contingency": ct}


def plot_unknown_share_by_land(df: pd.DataFrame, column: str = "wiki_number",
                                ax: plt.Axes | None = None) -> plt.Figure:
    """Bar chart: podíl obcí s labelem 'unknown' v každé historické zemi."""
    set_style()
    d = add_land_column(df)
    d = d[d[column].isin(["singular", "plural", "unknown"])]
    share = d.groupby("land")[column].apply(lambda s: (s == "unknown").mean()).sort_values()

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, _lollipop_height(len(share))))
    else:
        fig = ax.figure

    _hlollipop(ax, list(share.index), list(share.values), color=COLORS["unknown"])
    for i, v in enumerate(share.values):
        ax.text(v + 0.002, i, f"{v:.1%}", va="center", fontsize=9)

    ax.set_xlabel("Podíl obcí s labelem 'unknown'")
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    return fig


def plural_neighbor_share(df: pd.DataFrame, column: str = "wiki_number", k: int = 6) -> pd.DataFrame:
    """Pro každou obec: podíl plurálu mezi jejími k nejbližšími KLASIFIKOVANÝMI sousedy.

    k-NN graf (`build_knn_weights`) se staví přes VŠECHNY obce (singular/
    plural/unknown) — unknown obce musí být v grafu jako uzly, aby měly
    vlastní skóre. Ale do samotného skóre (podílu plurálu) se počítají jen
    sousedé s definovaným labelem (singular/plural) — unknown soused se
    nezapočítává, jinak by skóre bylo kruhové (unknown by "vysvětlovalo"
    unknown).

    Vrací gdf se sloupci `plural_neighbor_share` (NaN, pokud obec nemá mezi
    k nejbližšími sousedy ani jednoho klasifikovaného — řídké, ale možné na
    okraji hustoty bodů) a `n_classified_neighbors`.
    """
    d = df[df[column].isin(_LABEL_ORDER)].copy()
    gdf, w = build_knn_weights(d, k=k)

    is_plural = (gdf[column] == "plural").values
    is_classified = gdf[column].isin(["singular", "plural"]).values

    shares = np.full(len(gdf), np.nan)
    n_classified = np.zeros(len(gdf), dtype=int)
    for i, neighbors in w.neighbors.items():
        classified = [j for j in neighbors if is_classified[j]]
        n_classified[i] = len(classified)
        if classified:
            shares[i] = np.mean(is_plural[classified])

    gdf["plural_neighbor_share"] = shares
    gdf["n_classified_neighbors"] = n_classified
    return gdf


def plot_neighbor_share_by_label(df_with_share: pd.DataFrame, column: str = "wiki_number",
                                  kind: str = "box", ax: plt.Axes | None = None) -> plt.Figure:
    """Box/violin: podíl plurálu mezi klasifikovanými sousedy, dle vlastního labelu obce.

    Vstup je výstup `plural_neighbor_share()` (musí mít sloupec
    `plural_neighbor_share`), ne syrový df.
    """
    set_style()
    d = df_with_share.dropna(subset=["plural_neighbor_share"])

    if ax is None:
        fig, ax = plt.subplots(figsize=config.FIGSIZE_SQUARE)
    else:
        fig = ax.figure

    palette = {c: COLORS[c] for c in _LABEL_ORDER}
    plot_fn = sns.violinplot if kind == "violin" else sns.boxplot
    kwargs = {"cut": 0} if kind == "violin" else {"showfliers": True, "width": 0.5}
    plot_fn(data=d, x=column, y="plural_neighbor_share", order=_LABEL_ORDER, hue=column,
            palette=palette, legend=False, ax=ax, **kwargs)

    ns = d[column].value_counts()
    ax.set_xticklabels([f"{_LABEL_TITLES[c]}\n(n={ns.get(c, 0)})" for c in _LABEL_ORDER])
    ax.set_xlabel("")
    ax.set_ylabel("Podíl plurálu mezi k nejbližšími\nklasifikovanými sousedy")
    kind_label = "Violin" if kind == "violin" else "Box"
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    return fig


def test_neighbor_share_by_label(df_with_share: pd.DataFrame, column: str = "wiki_number") -> dict:
    """Kruskal-Wallis + párové Mann-Whitney na `plural_neighbor_share` dle labelu.

    Stejná logika jako `test_population_by_label()`, jiná proměnná: místo
    "jak velká je obec" testuje "jak moc plurálové je její okolí". Pokud je
    skóre u unknown obcí systematicky nižší než u plurálu (a bližší
    singuláru), jsou unknown obce prostorově vtlačené do singulárových
    kapes — tj. `unknown` koreluje s geografií nezávisle na tom, co ukázal
    binární chi² test na úrovni celého regionu.
    """
    d = df_with_share.dropna(subset=["plural_neighbor_share"])
    groups = {c: d.loc[d[column] == c, "plural_neighbor_share"] for c in _LABEL_ORDER}
    return _kruskal_pairwise(groups)


# ---------------------------------------------------------------------------
# 12. Sufixální morfologie vs. gramatické číslo (notebook 11)
#
# Vstupní df musí mít sloupec `suffix_group` (viz `morphology.add_suffix_column`).
# ---------------------------------------------------------------------------

def plot_suffix_overview(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Figure:
    """Bar chart: počet obcí v každé sufixální skupině.

    Šířka plátna (8") je záměrně stejná jako u `plot_by_suffix()` — obě
    zobrazují stejné 3 skupiny hned pod sebou na stránce; různá šířka by je
    vykreslila v jiném měřítku a jedna by vedle druhé působila "přiblíženě".
    """
    set_style()
    counts = df["suffix_group"].value_counts()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, _lollipop_height(len(SUFFIX_ORDER))))
    else:
        fig = ax.figure

    total = len(df)
    values = [counts.get(c, 0) for c in SUFFIX_ORDER]
    colors = [config.CATEGORICAL_PALETTE[i] for i in range(len(SUFFIX_ORDER))]
    _hlollipop(ax, [SUFFIX_TITLES[c] for c in SUFFIX_ORDER], values, colors=colors)
    for i, c in enumerate(SUFFIX_ORDER):
        n = counts.get(c, 0)
        ax.text(n + counts.max() * 0.02, i, f"{n} ({n/total:.1%})", va="center", fontsize=9)

    ax.set_xlabel("Počet obcí")
    ax.set_xlim(0, counts.max() * 1.20)
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    return fig


def plot_by_suffix(df: pd.DataFrame, column: str = "wiki_number", ax: plt.Axes | None = None) -> plt.Figure:
    """Lollipop: podíl plurálu v každé sufixální skupině (analogie plot_by_land).

    Pořadí i barvy skupin jsou stejné jako v `plot_suffix_overview()` (ne
    seřazené podle podílu) — obě grafy vedle sebe na stránce mají stejné
    kategorie na stejném místě, snáz se mezi nimi přepíná pohled.
    """
    d = df[df[column].isin(["singular", "plural"])]
    pivot = d.groupby(["suffix_group", column]).size().unstack(fill_value=0)
    pivot = pivot.reindex(SUFFIX_ORDER)
    pivot["total"] = pivot.sum(axis=1)
    pivot["pct_pl"] = pivot.get("plural", 0) / pivot["total"]

    colors = [config.CATEGORICAL_PALETTE[i] for i in range(len(SUFFIX_ORDER))]
    return _share_lollipop([SUFFIX_TITLES[c] for c in pivot.index], pivot["pct_pl"].values,
                            pivot["total"].values, colors, ax=ax,
                            figsize=(8, _lollipop_height(len(pivot))))


def chi2_suffix_test(df: pd.DataFrame, column: str = "wiki_number") -> dict:
    """Chi² test: je distribuce sg/pl nezávislá na sufixální skupině?"""
    d = df[df[column].isin(["singular", "plural"])]
    ct = pd.crosstab(d["suffix_group"], d[column]).reindex(SUFFIX_ORDER)
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    return {"chi2": chi2, "p": p, "dof": dof, "contingency": ct}


def chi2_unknown_suffix_test(df: pd.DataFrame, column: str = "wiki_number") -> dict:
    """Chi² test: liší se podíl 'unknown' mezi sufixálními skupinami?

    Kontrola konfundu symetrická k `chi2_unknown_land_test()` z nb10 — kdyby
    unknown korelovalo se sufixem, mohlo by to zkreslovat i test sg/pl vs.
    sufix (méně dat = méně jistoty zrovna v jedné skupině).
    """
    d = df[df[column].isin(["singular", "plural", "unknown"])].copy()
    d["status"] = np.where(d[column] == "unknown", "unknown", "klasifikováno")
    ct = pd.crosstab(d["suffix_group"], d["status"]).reindex(SUFFIX_ORDER)
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    return {"chi2": chi2, "p": p, "dof": dof, "contingency": ct}


def chi2_suffix_land_test(df: pd.DataFrame) -> dict:
    """Chi² test: je sufixální skupina rozmístěná nezávisle na historické zemi?

    Kontrola konfundu: pokud by např. -ovice bylo silně vázané na jeden
    region, efekt sufixu z `chi2_suffix_test()` by se mohl částečně
    překrývat s regionálním efektem z nb07/nb08.
    """
    d = add_land_column(df)
    ct = pd.crosstab(d["suffix_group"], d["land"]).reindex(SUFFIX_ORDER)
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    return {"chi2": chi2, "p": p, "dof": dof, "contingency": ct}


def chi2_land_test_by_suffix(df: pd.DataFrame, column: str = "wiki_number") -> dict:
    """Stratifikovaný test: přežije efekt historické země (nb07/nb08), když se
    kontroluje na sufixální skupinu?

    Pro každou sufixální skupinu zvlášť spočítá chi² sg/pl vs. land. Pokud
    efekt země zmizí uvnitř skupin, byl z větší části jen odrazem toho, že
    sufixální skupiny samy nejsou rovnoměrně rozmístěné (viz
    `chi2_suffix_land_test`). `low_count_warning` upozorní, když má
    kontingenční tabulka očekávané četnosti < 5 (chi² aproximace je pak
    nespolehlivá — typicky u -ovice, kde je singulár extrémně vzácný).
    """
    d = add_land_column(df)
    d = d[d[column].isin(["singular", "plural"])]
    results = {}
    for grp in SUFFIX_ORDER:
        sub = d[d["suffix_group"] == grp]
        ct = pd.crosstab(sub["land"], sub[column])
        chi2, p, dof, expected = stats.chi2_contingency(ct)
        results[grp] = {
            "n": len(sub), "chi2": chi2, "p": p, "dof": dof, "contingency": ct,
            "low_count_warning": bool((expected < 5).any()),
        }
    return results


def plot_suffix_detail(df: pd.DataFrame, column: str = "wiki_number",
                        min_n: int = 15, ax: plt.Axes | None = None) -> plt.Figure:
    """Popisný (ne testovaný) graf: % plurálu pro jednotlivé 5písmenné koncovky.

    Jemnější rozlišení než 3 hlavní skupiny — zajímavé pro morfologický
    detail, ale skupiny jsou malé a početné, takže formální test by měl
    málo síly a hodně příležitostí k falešné pozitivitě. Jen popisné.

    Barva je záměrně INK (tmavá), ne PRIMARY steelblue jako u otestovaných
    zjištění jinde v projektu — vizuální signál "tohle je exploratorní pohled,
    ne otestovaný nález", ne porušení datové palety (ta vyhrazuje jen
    terakotu striktně; steelblue je konvence pro hlavní/testovanou řadu).
    """
    set_style()
    d = df[df[column].isin(["singular", "plural"])].copy()
    d["end5"] = d["name"].map(suffix_ending)
    g = d.groupby("end5")[column].agg(n="count", pct_plural=lambda s: (s == "plural").mean())
    g = g[g["n"] >= min_n].sort_values("pct_plural")

    if ax is None:
        fig, ax = plt.subplots(figsize=(config.FIGSIZE_WIDE[0], _lollipop_height(len(g))))
    else:
        fig = ax.figure

    _hlollipop(ax, list(g.index), list(g["pct_plural"]), color=config.INK_COLOR)
    for i, (end5, row) in enumerate(g.iterrows()):
        ax.text(row["pct_plural"] + 0.01, i, f"{row['pct_plural']:.0%} (n={row['n']:.0f})",
                va="center", fontsize=8)

    ax.set_xlabel("Podíl plurálu")
    ax.set_xlim(0, 1.15)
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 13. Multivariátní syntéza: sufix + populace + region v jednom modelu (nb12)
#
# Vstupní df musí mít sloupec `suffix_group` (viz `morphology.add_suffix_column`).
# ---------------------------------------------------------------------------

_MODEL_SPECS = [
    ("populace",                  "y ~ log_pop"),
    ("region",                    "y ~ C(land)"),
    ("sufix",                     "y ~ C(suffix_group)"),
    ("sufix + populace",          "y ~ C(suffix_group) + log_pop"),
    ("sufix + populace + region", "y ~ C(suffix_group) + log_pop + C(land)"),
]


def fit_nested_logit_models(df: pd.DataFrame, column: str = "wiki_number") -> dict[str, object]:
    """Nafituje 5 (vnořených) logistických modelů sg/pl na stejném vzorku.

    Cíl: kvantifikovat, kolik variance sg/pl vysvětlí sufix samotný oproti
    populaci/regionu, a jestli populace/region přidají cokoli NAD RÁMEC
    sufixu. `-ovice` (98 % plurál, nb11) v datech nezpůsobuje perfektní
    separaci (11 protipříkladů existuje) — model konverguje bez regularizace.

    Vrací dict {název_modelu: statsmodels LogitResults}, v pořadí ze
    `_MODEL_SPECS` (populace / region / sufix / sufix+populace /
    sufix+populace+region) — použij pro pseudo-R² progresi a LR testy.
    """
    import statsmodels.formula.api as smf

    d = df[df[column].isin(["singular", "plural"])].dropna(subset=["population_total"]).copy()
    d = d[d["population_total"] > 0]
    d = add_land_column(d)
    d["y"] = (d[column] == "plural").astype(int)
    d["log_pop"] = np.log10(d["population_total"])

    return {name: smf.logit(formula, data=d).fit(disp=0) for name, formula in _MODEL_SPECS}


def lr_test(model_small, model_big) -> dict:
    """Likelihood-ratio test mezi dvěma vnořenými modely (menší je podmnožina většího).

    Testuje, jestli přidané proměnné modelu `model_big` vysvětlují
    statisticky významně víc, než by se čekalo náhodou — na rozdíl od
    pouhého srovnání pseudo-R² (které vždy roste přidáním proměnných, ať
    užitečných nebo ne).
    """
    lr_stat = 2 * (model_big.llf - model_small.llf)
    df_diff = model_big.df_model - model_small.df_model
    p = stats.chi2.sf(lr_stat, df_diff)
    return {"lr": lr_stat, "df": df_diff, "p": p}


def plot_pseudo_r2_progression(models: dict[str, object], ax: plt.Axes | None = None) -> plt.Figure:
    """Bar chart: McFadden pseudo-R² pro každý model z `fit_nested_logit_models()`."""
    set_style()
    names = list(models.keys())
    r2 = [models[n].prsquared for n in names]

    if ax is None:
        fig, ax = plt.subplots(figsize=(config.FIGSIZE_WIDE[0], _lollipop_height(len(names))))
    else:
        fig = ax.figure

    _hlollipop(ax, names, r2, color=config.PRIMARY_COLOR)
    for i, v in enumerate(r2):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)

    ax.set_xlabel("Pseudo-R² (McFadden)")
    ax.invert_yaxis()
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    return fig


def _pretty_coef_name(raw: str) -> str:
    m = re.match(r"C\((\w+)\)\[T\.(.+)\]", raw)
    if raw == "log_pop":
        return "log₁₀(populace)"
    if raw == "Intercept":
        return "Intercept"
    if m:
        factor, level = m.groups()
        if factor == "suffix_group":
            return f"{SUFFIX_TITLES.get(level, level)}  (vs. {SUFFIX_TITLES['bare_ice']})"
        if factor == "land":
            return f"{level}  (vs. Morava)"
    return raw


def plot_model_coefficients(model, ax: plt.Axes | None = None) -> plt.Figure:
    """Forest plot: koeficienty finálního modelu (log-odds) s 95% CI, bez interceptu.

    Barva sama nekóduje "riziko" (ACCENT se v projektu vyhrazuje jen pro
    to) — všechny koeficienty jsou PRIMARY, nula (žádný efekt) je
    referenční čárkovaná čára.
    """
    set_style()
    params = model.params.drop("Intercept")
    conf = model.conf_int().drop("Intercept")
    order = params.reindex(params.abs().sort_values().index).index

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, _lollipop_height(len(order))))
    else:
        fig = ax.figure

    for i, name in enumerate(order):
        coef = params[name]
        lo, hi = conf.loc[name]
        ax.errorbar(coef, i, xerr=[[coef - lo], [hi - coef]], fmt="o",
                     color=config.PRIMARY_COLOR, capsize=3, markersize=6, lw=1.5)

    ax.axvline(0, color="#999", lw=1, ls="--")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([_pretty_coef_name(n) for n in order])
    ax.set_xlabel("Koeficient (log-odds) → plurál, s 95% CI")
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 14. Doplňkové analýzy (notebook 13): rank-size (Zipf), hustota -ice obcí
#     dle rozlohy kraje, citlivost prostorových testů na volbě k.
# ---------------------------------------------------------------------------

def plot_rank_size(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Figure:
    """Rank-size (Zipfův) graf: populace obce vs. pořadí podle velikosti, log-log.

    Čistě popisný pohled, ne statistický test — proto INK_COLOR pro samotná
    data, ne PRIMARY (viz STYLE_GUIDE.md §1). Referenční čárkovaná čára se
    sklonem -1 (ideální Zipfovo rozdělení, P(rank) = P(1)/rank) je NEUTRAL —
    je to srovnávací referenční tvar, ne nález ani riziko, takže ACCENT by tu
    byl sémanticky špatně (ten je vyhrazený pro anomálie/prahy jinde v projektu).

    Bod-na-obec graf s ~1 800 body → ulož jako PNG (`save_fig(fig, ..., fmt="png")`),
    SVG by tu kreslilo každý bod jako samostatný element zbytečně navíc.
    """
    set_style()
    d = df.dropna(subset=["population_total"])
    d = d[d["population_total"] > 0].sort_values("population_total", ascending=False)
    rank = np.arange(1, len(d) + 1)
    pop = d["population_total"].values

    if ax is None:
        fig, ax = plt.subplots(figsize=config.FIGSIZE_DEFAULT)
    else:
        fig = ax.figure

    ax.scatter(rank, pop, s=10, color=config.INK_COLOR, alpha=0.5, linewidths=0, label="Obce na -ice")
    ref = pop[0] / rank
    ax.plot(rank, ref, color=config.NEUTRAL_COLOR, lw=config.BASELINE_LW, ls=config.BASELINE_LS,
            label="Ideální Zipf (sklon = -1)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Pořadí podle velikosti (log)")
    ax.set_ylabel("Počet obyvatel (log)")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def test_zipf_law(df: pd.DataFrame, n_bootstrap: int = 200, seed: int = 42) -> dict:
    """Férové ověření Zipfova zákona (Clauset, Shalizi & Newman 2009), balík `powerlaw`.

    Naivní OLS regrese na log-log rank-size grafu (`plot_rank_size`) skoro
    vždy vrátí vysoké R² — to podle kritiky Piantadosiho (2014) nic neříká
    o tom, jestli je mocninné rozdělení skutečně nejlepší popis dat. Správný
    postup: MLE odhad exponentu s KS-optimálním prahem `x_min` (mocninné
    rozdělení typicky neplatí pro celá data, jen pro chvost nad prahem),
    Monte Carlo bootstrap na "dá se to vůbec zamítnout" a Vuongův LRT proti
    log-normálnímu rozdělení — nejčastější alternativě k mocninnému rozdělení
    u velikostí měst (Eeckhout 2004).

    Vrací dict:
      `fit` — `powerlaw.Fit` objekt (pro `plot_zipf_ccdf`)
      `alpha`, `xmin`, `n_tail`, `sigma` — MLE odhad exponentu a KS-optimální práh
      `bootstrap_p` — GOF p-hodnota (CSN 2009 §4); p ≥ 0,1 → mocninné rozdělení
        NAD `xmin` nelze na této hladině zamítnout
      `lognormal_R`, `lognormal_p` — R > 0 → mocninné je lepší fit, R < 0 →
        log-normální; p < 0,05 → rozdíl je statisticky významný
    """
    import powerlaw

    freqs = df["population_total"].dropna()
    freqs = freqs[freqs > 0].values

    fit = powerlaw.Fit(freqs, discrete=True, verbose=False)
    ks_emp = fit.power_law.D

    np.random.seed(seed)
    ks_syn = []
    for _ in range(n_bootstrap):
        synthetic = fit.power_law.generate_random(len(fit.data), estimate_discrete=True)
        try:
            syn_fit = powerlaw.Fit(synthetic, discrete=True, verbose=False, xmin=fit.xmin)
            ks_syn.append(syn_fit.power_law.D)
        except Exception:
            continue
    ks_syn = np.array(ks_syn)
    bootstrap_p = float(np.mean(ks_syn >= ks_emp)) if len(ks_syn) else float("nan")

    lognormal_R, lognormal_p = fit.distribution_compare("power_law", "lognormal", normalized_ratio=True)

    return {
        "fit": fit,
        "alpha": fit.power_law.alpha, "xmin": fit.xmin,
        "n_tail": int(fit.n), "sigma": fit.power_law.sigma,
        "bootstrap_p": bootstrap_p, "n_bootstrap": len(ks_syn),
        "lognormal_R": lognormal_R, "lognormal_p": lognormal_p,
    }


def plot_zipf_ccdf(fit, ax: plt.Axes | None = None) -> plt.Figure:
    """CCDF diagnostický graf: empirická data vs. fitované mocninné a
    log-normální rozdělení, log-log, jen nad KS-optimálním `fit.xmin`.

    Standardní diagnostika pro "fair" Zipf testing (viz `test_zipf_law`) —
    na rozdíl od rank-size grafu (`plot_rank_size`, celá data, naivní pohled)
    tenhle graf ukazuje přesně tu část dat a přesně ten model, který byl
    statisticky testovaný. Empirická data jsou INK (popisná), mocninný fit
    PRIMARY (hlavní testovaný model), log-normální NEUTRAL (srovnávací
    alternativa) — pokud se PRIMARY a NEUTRAL čára prakticky překrývají,
    je to vizuální potvrzení toho, co říká `lognormal_R` blízké nule.
    """
    set_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=config.FIGSIZE_DEFAULT)
    else:
        fig = ax.figure

    fit.plot_ccdf(ax=ax, color=config.INK_COLOR, linewidth=0, marker="o",
                  markersize=4, alpha=0.5, label="Obce (nad xmin)")
    fit.power_law.plot_ccdf(ax=ax, color=config.PRIMARY_COLOR, linewidth=2.2,
                             label=f"Mocninné (α = {fit.power_law.alpha:.2f})")
    fit.lognormal.plot_ccdf(ax=ax, color=config.NEUTRAL_COLOR, linewidth=2.2,
                             linestyle="--", label="Log-normální")

    ax.set_xlabel(f"Počet obyvatel, log (≥ xmin = {fit.xmin:.0f})")
    ax.set_ylabel("P(X ≥ x), log")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_density_by_kraj(df: pd.DataFrame, area_df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Figure:
    """Lollipop: počet obcí na -ice na 1 000 km² plochy kraje.

    Surový počet obcí na kraj (`plot_region_totals`) zavádí — Středočeský kraj
    vede částečně jen proto, že je plošně největší. Normalizace rozlohou
    (`data.load_kraj_areas()`, zdroj Wikidata/ČSÚ) odpovídá na jinou otázku:
    kde je -ice obcí poměrově nejvíc vzhledem k velikosti kraje, ne kde jich
    je nejvíc v absolutních číslech.
    """
    set_style()
    counts = df["region_name"].value_counts().rename("n_obci")
    d = area_df.set_index("kraj").join(counts, how="inner")
    d["density"] = d["n_obci"] / d["area_km2"] * 1000
    d = d.sort_values("density")
    short = [_REGION_SHORT.get(r, r) for r in d.index]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, _lollipop_height(len(d))))
    else:
        fig = ax.figure

    _hlollipop(ax, short, list(d["density"]), color=config.PRIMARY_COLOR)
    for i, v in enumerate(d["density"]):
        ax.text(v + d["density"].max() * 0.02, i, f"{v:.1f}", va="center", fontsize=9)

    ax.set_xlabel("Obcí na -ice na 1 000 km² plochy kraje")
    ax.set_xlim(0, d["density"].max() * 1.15)
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    return fig


def k_sensitivity_test(df: pd.DataFrame, column: str = "wiki_number",
                        k_values: Iterable[int] = range(4, 21, 2),
                        permutations: int = 999, seed: int = 42) -> pd.DataFrame:
    """Přepočítá globální Moran's I přes rozsah k (velikost k-NN sousedství).

    Ověřuje, že hlavní prostorový nález (`moran_binary_test`, k=6) není
    artefakt jedné konkrétní volby "kolik sousedů se počítá" — pokud I
    zůstává v podobném řádu a se stejným znaménkem napříč k=4..20, závěr
    "sousední obce sdílejí gramatické číslo víc, než by čekala náhoda" je
    robustní na tenhle modelovací parametr, ne artefakt k=6.

    Méně permutací než hlavní test (999 vs. 9999 v `moran_binary_test`) —
    tady jde o směr/řád efektu napříč k, ne o přesnou p-hodnotu jednoho k;
    9 hodnot k × 9999 permutací by bylo zbytečně pomalé pro tuhle otázku.
    """
    rows = []
    for k in k_values:
        mi = moran_binary_test(df, column=column, k=k, permutations=permutations, seed=seed)
        rows.append({"k": k, "moran_I": mi["I"], "p_sim": mi["p_sim"]})
    return pd.DataFrame(rows)


def plot_k_sensitivity(result_df: pd.DataFrame, chosen_k: int = 6, ax: plt.Axes | None = None) -> plt.Figure:
    """Scree-like graf: Moran's I napříč hodnotami k, se zvýrazněným zvoleným k.

    Čára+tečky jsou INK/PRIMARY (popisný průběh), zvolené k je zvýrazněné
    ACCENT — ne proto, že by bylo "riziko", ale je to jediné k, na kterém
    stojí zbytek článku, takže si zaslouží stejnou vizuální váhu jako
    upozornění jinde v projektu.
    """
    set_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.5))
    else:
        fig = ax.figure

    ax.plot(result_df["k"], result_df["moran_I"], color=config.INK_COLOR, lw=1.6, zorder=2)
    ax.scatter(result_df["k"], result_df["moran_I"], color=config.PRIMARY_COLOR, s=36,
               zorder=3, linewidths=0)
    chosen = result_df[result_df["k"] == chosen_k]
    if not chosen.empty:
        ax.scatter(chosen["k"], chosen["moran_I"], color=config.ACCENT_COLOR, s=90, zorder=4,
                   linewidths=0, label=f"použité k = {chosen_k}")
        ax.legend(frameon=False)

    ax.set_xlabel("k (počet nejbližších sousedů)")
    ax.set_ylabel("Moran's I")
    ax.set_xticks(list(result_df["k"]))
    ax.set_ylim(0, max(result_df["moran_I"].max() * 1.2, 0.05))
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 15. Poster composer — JEDINÉ místo, kam se titulek smí propálit do obrázku.
#
# Grafy vložené do článku na webu titulek/podtitulek uvnitř obrázku NEMAJÍ
# mít (mobilní responsivita, čtečky/SEO, editace bez re-run Pythonu — titulek
# patří do <h2>/<figcaption> v HTML). compose_poster() je pro opačný případ:
# samostatný obrázek, co cestuje sám o sobě (LinkedIn, X, slajd, PDF) a musí
# nést titulek jako součást pixelů. Sdílená verze i s STYLE_GUIDE.md je v
# research-template/modules/dataviz_style/ — tohle je zrcadlená kopie
# přizpůsobená na COLORS/config tohoto projektu.
# ---------------------------------------------------------------------------

def compose_poster(
    plot_fns,
    title: str,
    subtitle: str | None = None,
    source: str | None = None,
    layout: str = "row",
    bg: str = "cream",
    figsize: tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Poskládá 1-2 grafy + editorial titulkový blok do jednoho samostatného obrázku.

    `plot_fns`: jeden nebo dva callables `f(ax) -> None`, každý nakreslí
    kompletní graf na dané `ax` — sedí přímo na konvenci téhle knihovny (každá
    `plot_*` funkce už bere `ax=`). Pro funkci, co potřebuje víc argumentů,
    použij `lambda ax: plot_x(df, ax=ax)`.

    Typografie: serif titulek (`config.FONT_SERIF`), sans vše ostatní, mono
    zdroj dole — viz STYLE_GUIDE.md v research-template, sekce "Poster mode".
    Nepoužívej tohle na grafy, co jdou do HTML článku.
    """
    previous_bg = _current_bg
    set_style(bg=bg)
    n = len(plot_fns)
    if n not in (1, 2):
        raise ValueError("compose_poster podporuje jen 1 nebo 2 grafy")

    fig = plt.figure(figsize=figsize)
    bg_color = config.BG_WHITE if bg == "white" else config.BG_CREAM
    fig.patch.set_facecolor(bg_color)

    bottom = 0.16
    top_margin = 0.72 if subtitle else 0.78
    if n == 1:
        axes = [fig.add_axes([0.08, bottom, 0.88, top_margin - bottom])]
    elif layout == "row":
        gap = 0.12  # dost místa na datové popisky u pravého okraje levého grafu
        w = (0.92 - gap) / 2
        axes = [
            fig.add_axes([0.06, bottom, w, top_margin - bottom]),
            fig.add_axes([0.06 + w + gap, bottom, w, top_margin - bottom]),
        ]
    else:
        gap = 0.10
        h = (top_margin - bottom - gap) / 2
        axes = [
            fig.add_axes([0.08, bottom + h + gap, 0.86, h]),
            fig.add_axes([0.08, bottom, 0.86, h]),
        ]

    for fn, ax in zip(plot_fns, axes):
        ax.set_facecolor(bg_color)
        fn(ax)

    fig.text(0.06, 0.95, title, fontfamily=config.FONT_SERIF, fontsize=20,
              fontweight="bold", color=config.INK_COLOR)
    if subtitle:
        fig.text(0.06, 0.87, subtitle, fontfamily=config.FONT_SANS, fontsize=12,
                  color=config.MUTED_COLOR)
    if source:
        fig.text(0.06, 0.02, source, fontfamily=config.FONT_MONO, fontsize=8,
                  color=config.MUTED_COLOR)

    set_style(bg=previous_bg)  # neprosakovat pozadí postera do dalších buněk
    return fig

# GeoGram — CLAUDE.md

## Co je projekt

Výzkumný projekt: 1 906 českých obcí s koncovkou `-ice`, cíl je zjistit zda jsou
gramaticky **singulár nebo plurál** (pomnožná jména) a jak jsou prostorově rozloženy.

Výzkumné otázky: podíl sg/pl, geografická distribuce (Čechy vs. Morava),
korelace s demografií, etymologické vzory, srovnání IJP ÚJČ vs. Wikipedia.

## Struktura projektu

```
src/geogram/          ← Python balíček (logika zde, ne v noteboocích)
  data.py             ← load_all_municipalities(), filter_suffix_ice()
  ingest.py           ← integrace RÚIAN + ČSÚ
  ujc.py              ← IJP ÚJČ parser (sg/pl z morfol. tabulky)
  wikipedia.py        ← Wikipedia disambiguace + extrakce sg/pl z intro textu
  ujc_scrape.py       ← robustní IJP scraper s cache surového HTML
  morphology.py       ← sufixální skupiny (-ovice / -nice / holé -ice)
  viz.py              ← grafy + statistické testy (chi², join-count, LISA, logit)
  web_export.py       ← JSON tabulka + statistiky pro článek na webu

data/raw/             ← RÚIAN CSVs (UI_OBEC, UI_OKRES, UI_VUSC) + ČSÚ Excel
data/processed/       ← výstupy (gitignored — generuj spuštěním notebooků)

notebooks/
  01  intro
  02  filtrace -ice obcí
  03  test IJP parseru (4 obce)
  04  integrace dat → municipalities_ice_integrated.csv (vč. souřadnic)
  05  batch IJP ÚJČ klasifikace → ice_grammar_ujc.csv
  06  batch Wikipedia klasifikace → ice_grammar_wiki.csv
  07  re-klasifikace unknown (intro + lokativní fallback), statistiky, mapa
  08–13  prostorová segregace, unknown vs. populace/geografie, sufixy,
         výjimky + syntéza, doplňkové analýzy
```

## Aktuální stav (2026-10-31)

Zdroj gramatického čísla je **Wikipedia** (IJP ÚJČ odloženo — server blokuje
po ~13 dotazech, viz gotchas; uživatel se rozhodl zůstat u Wikipedie).
Odborný článek se nechystá, výstupem je článek na webu (repo `personalWeb`,
`projekty/obce-na-ice.html`).

- **1 906 obcí** (`filter_suffix_ice` přes `ice_core()` — vč. víceslovných
  názvů „Lipnice nad Sázavou“, spojených obcí a římských číslic; sloupce
  `ice_head`, `name_type`). Do 2026-10 bylo chybně 1 806.
- `ice_grammar_wiki_v2.csv`: 1 507 plurál, 251 singulár, 148 unknown
- Pipeline: nb04 → nb06 (navazuje, dotazuje jen nové/unknown) → nb07 → nb08–13
- **Aktualizace webu:** grafy z `assets/img/geogram/` (10 souborů, viz `<img>`
  v článku) + `data/processed/mapa_ice.html` → `personalWeb/assets/geogram/mapa-ice.html`
  + `python -m geogram.web_export` (tabulka + statistiky + blok `text` s čísly
  citovanými v próze).
- `ujc_scrape.py` — robustní IJP scraper s HTML cache (`data/raw/ujc_html/`),
  zatím staženo jen 13 hesel.

## Kritické gotchas — přečti PŘED prací na IJP

### ÚJČ server rate-limiting
Server `prirucka.ujc.cas.cz` zpracovává **jeden dotaz z IP najednou**.
Rychlé dotazy způsobí odpověď "server je přetížen" (bez tabulky → `not_found`).

Implementace to detekuje (`_is_overloaded()`) a opakuje s backoffem (5→10→20→40s).
Ale když je IP tvrdě blokována → dostaneš `ReadTimeout` i po backoffu.

**Příznaky blokace:** notebook 03 (4 dotazy) vyhodí `ReadTimeout`.
**Řešení:** počkej přes noc. Pak znovu spusť s `SLEEP = 5.0`.

Bypass "nechci čekat" na stránce = JavaScript odkaz, nelze zavolat z Pythonu.

### Kategorie výsledků (ujc.py)
- `singular` / `plural` / `both` — klasifikováno
- `not_found` — stránka bez morfologické tabulky (throttling NEBO heslo v IJP neexistuje)
- `unknown` — tabulka nalezena, hodnoty nerozlišitelné
- `error` — síťová/HTTP výjimka

Malé vesnice (<500 obyvatel) skutečně v IJP ÚJČ chybí → `not_found` je pro ně legitimní.

### Wikipedia — diakritika
Wikipedia search funguje pouze s diakritickými názvy. Vždy použij sloupec `name` z CSV.

## Konvence

- Logika patří do `src/geogram/`, notebooky jsou jen orchestrace a vizualizace
- Data se necommitují (`data/` je v `.gitignore`)
- Commity: malé, logické, bez Co-Authored-By
- Virtuální prostředí: `.venv/` (aktivuj před spuštěním)
- Progress bary: `tqdm.notebook.tqdm` pro všechny batch operace

## Next steps po dokončení klasifikace

1. Mail na ÚJČ pro speciální přístup k IJP (server dosud blokuje IP)
2. Notebook 07 rozšířit o srovnání IJP vs. Wikipedia, až budou IJP data (funkce ve `viz.py` už připravené: `classify_agreement`, `plot_agreement_*`)

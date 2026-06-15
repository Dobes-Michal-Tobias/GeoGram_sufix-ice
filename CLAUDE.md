# GeoGram — CLAUDE.md

## Co je projekt

Výzkumný projekt: 1 806 českých obcí s koncovkou `-ice`, cíl je zjistit zda jsou
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

data/raw/             ← RÚIAN CSVs (UI_OBEC, UI_OKRES, UI_VUSC) + ČSÚ Excel
data/processed/       ← výstupy (gitignored — generuj spuštěním notebooků)

notebooks/
  01  intro
  02  filtrace -ice obcí
  03  test IJP parseru (4 obce)
  04  integrace dat → municipalities_ice_integrated.csv
  05  batch IJP ÚJČ klasifikace → ice_grammar_ujc.csv
  06  batch Wikipedia klasifikace → ice_grammar_wiki.csv
```

## Aktuální stav (2026-06-15)

### Hotovo a funkční
- `data/processed/municipalities_ice_integrated.csv` — 1 806 obcí s demografií
- `src/geogram/ujc.py` — parser funguje, má retry logiku pro server overload
- `src/geogram/wikipedia.py` — disambiguace (3 strategie) + regex extrakce sg/pl
- Notebooky 01–06 jsou opraveny a připraveny ke spuštění

### Co je potřeba ještě spustit

**Notebook 05 (IJP ÚJČ):**
- `data/processed/ice_grammar_ujc.csv` existuje ALE data jsou vadná
- Při posledním běhu server ÚJČ throttloval IP → 684/700 výsledků je `not_found`
- **Postup:** ověř že server odpovídá (notebook 03), pak spusť nb05 s `SLEEP_BETWEEN_REQUESTS = 5.0`
- Nb05 automaticky přeskočí správné výsledky (sg/pl/both) a re-procesuje jen not_found/error

**Notebook 06 (Wikipedia):**
- Nebyl ještě spuštěn vůbec
- Odhadovaný čas: ~15 minut (Wikipedia API je benevolentní, 0.5s/dotaz)

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

1. Notebook 07: srovnání IJP vs. Wikipedia (shoda, neshody, pokrytí)
2. Notebook 08: mapa ČR s body -ice obcí, barvené dle sg/pl (geopandas / folium)
3. Statistická analýza: t-test/chi2 sg vs. pl podle populace, průměrného věku, kraje

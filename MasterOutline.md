# GeoGram — Master Outline

## Co jsme dosud vytvořili
- Projektová struktura (src/, notebooks/, data/raw, data/processed)
- Počáteční soubory: `.gitignore`, `README.md`, `requirements.txt`, `pyproject.toml`
- Python package `geogram` se základními moduly:
  - `data.py` – načítání a základní filtrace
  - `ingest.py` – integrace RÚIAN + ČSÚ
  - `ujc.py` – parser IJP ÚJČ (singulár/plurál)
  - `wikipedia.py` – helper pro stažení perexu z Wikipedie
  - `__main__.py` – jednoduché CLI: `python -m geogram input.csv`
- Notebooky:
  - `notebooks/01-project-setup.ipynb` — úvod
  - `notebooks/02-filter-ice.ipynb` — filtrování `-ice`
  - `notebooks/03-ujc-analysis.ipynb` — test IJP extrakce
  - `notebooks/04-data-integration.ipynb` — integrace a základní statistiky
- Data: RÚIAN CSV (`UI_*.csv`) a ČSÚ excel se sbírkou počtů obyvatel.

## Hlavní cíl projektu
Zmapovat morfologicko-geografický jev: obce v ČR s koncovkou `-ice` a zjistit, zda jsou gramaticky vnímány spíše jako singulár nebo plurál, a jak jsou prostorově rozloženy.

## Výzkumné otázky (RQ)
1. RQ1: Jaký podíl obcí končících na `-ice` je gramaticky singulár vs. plurál podle IJP ÚJČ a Wikipedia perexu? (kolik přesně a jaká je shoda mezi zdroji)
2. RQ2: Jak se `-ice` obce geograficky rozkládají — převaha v Čechách vs. na Moravě, nebo jiné regionální shluky?
3. RQ3: Existují korelace mezi gramatickým číslem (sg/pl) a demografickými atributy (velikost obce, průměrný věk)?
4. RQ4: Vyskytují se lexikálně/etymologické vzory (např. typ sufixu, původ názvu) spojené s počtem (plurál/pomnožné forma)?
5. RQ5: Jak robustní je metoda založená na perexu Wikipedie oproti autoritě IJP ÚJČ (přesnost, chybovost, chybějící hesla)?

## Plán práce (konkrétní kroky)
1. Dokončení a čištění ingesovaných dat
   - Dolepit chybějící mapování kódů → názvy okresů/krajů (hotovo)
   - Doplnit populaci (hotovo)
   - Přidat souřadnice (body) z RÚIAN nebo jiného otevřeného zdroje (další krok)
2. Linguistická klasifikace
   - Paralelní přístupy: IJP ÚJČ parser + Wikipedie perex + fallback UDPipe
   - Normalizace výsledků, rozhodovací pravidla pro konflikty
3. Analýza a vizualizace
   - Mapové vizualizace (GeoPandas / folium / kepler) — heatmapy a shluky
   - Statistická analýza rozdílů (t-test/ANOVA/chi2 podle proměnných)
4. Zpracování výstupu
   - Publikovat `data/processed/ice_grammar_ujc.csv` a `ice_grammar_wiki.csv`
   - Vytvořit notebooky s grafy a mapami pro článek/blog
5. Reproducibilita a publikace
   - Upravit `README.md` se způsobem reprodukce
   - Připravení repozitáře pro veřejné sdílení (malé commity, žádní spoluautoři)

## Krátkodobé úkoly (to be done next)
- [ ] Doházet souřadnice (RÚIAN geodata) a připojit do `data/processed`
- [ ] Dokončit škálování IJP parseru pro všechny 1 806 obcí
- [ ] Spustit Wikipedii perex + UDPipe na stejné množině a vyhodnotit shodu
- [ ] Vytvořit mapu ČR s body `-ice` obcí a barevným rozlišením sg/pl

## Distribuce výsledků / očekávané artefakty
- `data/processed/municipalities_ice_integrated.csv` (hotovo)
- `data/processed/ice_grammar_ujc.csv`
- `data/processed/ice_grammar_wiki.csv`
- Notebooks: vizualizace, srovnání IJP vs. Wiki, závěry

## Git & publikace
- Budu dělat více malých commitů (logicky dle změn): scaffold → ingestion → NLP → notebooks → výsledky
- Doporučení: nastavit `origin` (GitHub remote) a `git push --set-upstream origin main` ručně; neprovedu push bez tvého souhlasu a bez nastavení remote.

## Poznámky k etice a citacím
- Data RÚIAN/ČSÚ jsou otevřená — vždy uveď zdroj.
- Pokud použijeme data z Wikipedie nebo IJP ÚJČ, cituj zdroje a ověřuj oprávnění u IJP.

---

Chceš, abych teď:
- a) připojil souřadnice z RÚIAN (automaticky), nebo
- b) spustil plný IJP parser nad všemi `-ice` obcemi a uložil výstup?

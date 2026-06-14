# GeoGram: Analýza českých obcí na sufix *-ice*

Projekt zkoumá české obce končící na *-ice* z pohledu gramatického čísla a geografického rozložení.

## Cíl

- získat kompletní seznam obcí v ČR
- vyfiltrovat ty, jejichž název končí na *-ice*
- analyzovat, zda jsou gramaticky vnímány jako singulár nebo plurál
- porovnat výsledky z Wikipedie a z jazykové příručky IJP ÚJČ
- vizualizovat prostorové rozložení v ČR

## Návrh pipeline

1. stáhnout data o obcích (ČSÚ / RÚIAN / otevřená data)
2. filtrovat názvy končící na *-ice*
3. pro každý název zkusit získat gramatické číslo z IJP ÚJČ
4. alternativně ověřit pomocí první věty z Wikipedie a UDPipe
5. spojit výsledky s geometrií a obyvatelstvem
6. hledat prostorové vzory a případné výjimky

## Struktura

- `src/geogram/` – Python modul s funkcemi pro data, scraping a analýzu
- `notebooks/` – Jupyter notebooky pro prozkoumání dat a vizualizaci
- `data/raw/` – surová data ke stažení
- `data/processed/` – připravená data pro analýzu

## Další kroky

- vytvořit virtuální prostředí
- nainstalovat závislosti z `requirements.txt`
- nainstalovat projekt v editable módu:

```powershell
python -m pip install -e .
```

- stáhnout nebo připravit dataset obcí a uložit ho jako `data/raw/municipalities.csv`
- spustit první filtraci `-ice` pomocí modulu:

```powershell
python -m geogram data/raw/municipalities.csv
```

- otevřít notebook `notebooks/02-filter-ice.ipynb` pro interaktivní analýzu
- otevřít notebook `notebooks/03-ujc-analysis.ipynb` pro první IJP ÚJČ klasifikaci

## English summary

GeoGram is a small spatial humanities project exploring Czech municipalities ending in *-ice*. The first stage is to collect the full list of municipalities, then classify them as singular or plural using authoritative language resources (IJP ÚJČ) and Wikipedia perex verbs. The goal is to compare grammatical patterns with geography.

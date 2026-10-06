"""Robust, resumable scraper for IJP ÚJČ with an on-disk HTML cache.

Design (lessons from the first failed batch run, see CLAUDE.md):

* The server handles **one request per IP at a time** — we stay strictly
  sequential, with a polite delay + jitter between requests.
* Every *valid* response (entry found / entry genuinely missing) is cached as
  raw HTML in ``data/raw/ujc_html/``. Re-running skips cached names, and the
  parser can be re-run offline at any time without touching the server.
* The page state is classified explicitly instead of "no table → not_found":
  ``found`` (declension table), ``absent`` ("Zadaný výraz … nebyl nalezen"),
  ``overloaded`` (queue page — never cached, retried with backoff) and
  ``unexpected`` (anything else — cached under a separate suffix for review).
* Circuit breaker: repeated timeouts → long pause; if the server keeps
  refusing, the run stops cleanly and can be resumed later.

Usage::

    python -m geogram.ujc_scrape              # scrape + parse
    python -m geogram.ujc_scrape --parse-only # re-parse cached HTML only
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from . import config
from .ujc import BOTH, NOT_FOUND, PLURAL, SINGULAR, UNKNOWN, build_ujc_url

log = logging.getLogger("geogram.ujc_scrape")

CACHE_DIR = config.RAW_DIR / "ujc_html"
INPUT_CSV = config.PROCESSED_DIR / "municipalities_ice_integrated.csv"
OUTPUT_CSV = config.PROCESSED_DIR / "ice_grammar_ujc.csv"

USER_AGENT = "GeoGram research scraper (Czech -ice toponyms; 1 req / 5 s)"

# Page states
FOUND = "found"
ABSENT = "absent"
OVERLOADED = "overloaded"
UNEXPECTED = "unexpected"

_OVERLOAD_SIGNALS = ("je ještě vyhodnocován", "server je přetížen", "is still evalu")
# Raw HTML uses entities: Zadaný výraz &bdquo;X&ldquo; nebyl nalezen.
_ABSENT_RE = re.compile(r"Zadaný výraz.{0,300}?nebyl nalezen", re.DOTALL)
# Only the entry notes count — sidebar links also mention "Zeměpisná jména pomnožná"
_POMNOZNE_RE = re.compile(r"\bpomnožn")


class ServerBlocked(RuntimeError):
    """Raised when the circuit breaker gives up — resume the run later."""


# ---------------------------------------------------------------------------
# Page classification + parsing (pure functions, no network)
# ---------------------------------------------------------------------------

def classify_page(html: str) -> str:
    if any(s in html for s in _OVERLOAD_SIGNALS):
        return OVERLOADED
    if 'class="para"' in html or "class='para'" in html:
        return FOUND
    if _ABSENT_RE.search(html):
        return ABSENT
    return UNEXPECTED


def _table_number(table) -> str:
    """Classify one ``table.para``: which column (sg / pl) has word forms."""
    rows = table.find_all("tr")
    if len(rows) < 2:
        return UNKNOWN
    header = [c.get_text(strip=True).lower() for c in rows[0].find_all(["th", "td"])]
    sg_i = next((i for i, c in enumerate(header) if "jednotn" in c), 1)
    pl_i = next((i for i, c in enumerate(header) if "mno" in c and "slo" in c), 2)

    has_sg = has_pl = False
    for row in rows[1:]:
        cells = row.find_all(["th", "td"])
        if len(cells) <= max(sg_i, pl_i):
            continue
        has_sg |= bool(re.sub(r"\d+", "", cells[sg_i].get_text()).strip())
        has_pl |= bool(re.sub(r"\d+", "", cells[pl_i].get_text()).strip())

    if has_sg and has_pl:
        return BOTH
    if has_sg:
        return SINGULAR
    if has_pl:
        return PLURAL
    return UNKNOWN


@dataclass
class ParsedEntry:
    page_state: str
    ujc_number: str          # final label: singular / plural / both / unknown / not_found
    n_tables: int            # >1 = homonymous entries on one page
    table_numbers: str       # per-table labels joined by "|"
    note_pomnozne: bool      # entry notes say "pomnožné"
    consistent: bool         # table and note agree (False → manual check)


def parse_page(html: str) -> ParsedEntry:
    state = classify_page(html)
    if state != FOUND:
        return ParsedEntry(state, NOT_FOUND, 0, "", False, True)

    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table", class_="para")
    labels = [_table_number(t) for t in tables]
    notes = [p.get_text(" ") for p in soup.find_all("p", class_="polozky")
             if p.get_text().strip().startswith("poznámky k heslu")]
    pomnozne = any(_POMNOZNE_RE.search(n) for n in notes)

    distinct = set(labels) - {UNKNOWN}
    if len(distinct) == 1:
        number = distinct.pop()
    elif len(distinct) > 1:
        number = BOTH
    else:
        number = UNKNOWN

    # "pomnožné" note must coincide with a plural-only table (and vice versa)
    consistent = (number == PLURAL) == pomnozne or number == BOTH
    return ParsedEntry(state, number, len(tables), "|".join(labels), pomnozne, consistent)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def cache_path(name: str, cache_dir: Path = CACHE_DIR, suffix: str = ".html") -> Path:
    # Filenames from hash: Czech diacritics + spaces are awkward on Windows
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{digest}{suffix}"


def is_cached(name: str, cache_dir: Path = CACHE_DIR) -> bool:
    return cache_path(name, cache_dir).exists()


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

@dataclass
class ScrapeConfig:
    delay: float = 5.0                # base pause between requests [s]
    jitter: float = 1.5               # + uniform(0, jitter)
    overload_backoff: tuple = (30, 60, 120, 300)
    timeout: tuple = (10, 45)         # (connect, read)
    max_consecutive_failures: int = 4  # timeouts/5xx in a row → long pause
    long_pause: float = 15 * 60
    max_long_pauses: int = 3          # then give up (ServerBlocked)
    max_overload_wait: float = 20 * 60  # total overload wait per term, then give up


class UjcScraper:
    def __init__(self, cfg: ScrapeConfig | None = None, cache_dir: Path = CACHE_DIR):
        self.cfg = cfg or ScrapeConfig()
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "cs"})
        self._fail_streak = 0
        self._long_pauses = 0
        self.stats = {FOUND: 0, ABSENT: 0, UNEXPECTED: 0, "overload_hits": 0, "net_errors": 0}

    def _sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def _polite_pause(self) -> None:
        self._sleep(self.cfg.delay + random.uniform(0, self.cfg.jitter))

    def _register_failure(self, reason: str) -> None:
        self.stats["net_errors"] += 1
        self._fail_streak += 1
        log.warning("network failure #%d in a row: %s", self._fail_streak, reason)
        if self._fail_streak < self.cfg.max_consecutive_failures:
            self._sleep(self.cfg.delay * 2 ** self._fail_streak)
            return
        if self._long_pauses >= self.cfg.max_long_pauses:
            raise ServerBlocked(
                f"server unreachable after {self._long_pauses} long pauses — "
                "IP is probably blocked, resume later (cache is kept)"
            )
        self._long_pauses += 1
        log.warning("circuit breaker: pausing %.0f min (%d/%d)",
                    self.cfg.long_pause / 60, self._long_pauses, self.cfg.max_long_pauses)
        self._sleep(self.cfg.long_pause)
        self._fail_streak = 0

    def fetch(self, name: str) -> str:
        """Fetch one term until a cacheable state is reached; cache and return its state."""
        url = build_ujc_url(name)
        overload_attempt = 0
        overload_waited = 0.0
        while True:
            try:
                resp = self.session.get(url, timeout=self.cfg.timeout)
                if resp.status_code >= 500 or resp.status_code == 429:
                    self._register_failure(f"HTTP {resp.status_code}")
                    continue
                resp.raise_for_status()
            except (requests.Timeout, requests.ConnectionError) as exc:
                self._register_failure(type(exc).__name__)
                continue

            self._fail_streak = 0
            resp.encoding = "utf-8"  # page declares utf-8; header may not
            html = resp.text
            state = classify_page(html)

            if state == OVERLOADED:
                self.stats["overload_hits"] += 1
                if overload_waited >= self.cfg.max_overload_wait:
                    raise ServerBlocked(
                        f"still overloaded on {name!r} after {overload_waited / 60:.0f} min — "
                        "rate limit window not over, resume later (cache is kept)"
                    )
                wait =self.cfg.overload_backoff[min(overload_attempt, len(self.cfg.overload_backoff) - 1)]
                overload_attempt += 1
                overload_waited += wait
                log.info("overloaded on %r, waiting %ss", name, wait)
                self._sleep(wait)
                continue

            suffix = ".html" if state in (FOUND, ABSENT) else ".unexpected.html"
            cache_path(name, self.cache_dir, suffix).write_text(html, encoding="utf-8")
            self.stats[state] += 1
            return state

    def scrape(self, names, progress=None) -> dict:
        """Scrape all names not yet in cache. ``progress`` wraps the iterable (tqdm)."""
        todo = [n for n in dict.fromkeys(names) if not is_cached(n, self.cache_dir)]
        log.info("%d unique names, %d left to fetch", len(set(names)), len(todo))
        it = progress(todo) if progress else todo
        for name in it:
            state = self.fetch(name)
            if hasattr(it, "set_postfix"):
                it.set_postfix(last=name[:16], state=state,
                               overloads=self.stats["overload_hits"], err=self.stats["net_errors"])
            self._polite_pause()
        return self.stats


# ---------------------------------------------------------------------------
# Parse cache → DataFrame
# ---------------------------------------------------------------------------

def parse_cache(df: pd.DataFrame, cache_dir: Path = CACHE_DIR, name_col: str = "name") -> pd.DataFrame:
    """Attach IJP results to ``df`` (one row per municipality) from cached HTML."""
    rows = {}
    for name in df[name_col].unique():
        path = cache_path(name, cache_dir)
        if path.exists():
            entry = parse_page(path.read_text(encoding="utf-8"))
            rows[name] = {"ujc_number": entry.ujc_number, "ujc_page_state": entry.page_state,
                          "ujc_n_tables": entry.n_tables, "ujc_table_numbers": entry.table_numbers,
                          "ujc_note_pomnozne": entry.note_pomnozne, "ujc_consistent": entry.consistent}
        else:
            rows[name] = {"ujc_number": "not_scraped", "ujc_page_state": "missing"}
    res = pd.DataFrame.from_dict(rows, orient="index").rename_axis(name_col).reset_index()
    return df.merge(res, on=name_col, how="left")


def main() -> None:
    from tqdm import tqdm

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parse-only", action="store_true", help="skip network, only parse cache")
    ap.add_argument("--delay", type=float, default=ScrapeConfig.delay)
    ap.add_argument("--limit", type=int, default=None, help="scrape only first N names (test run)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(config.PROCESSED_DIR / "ujc_scrape.log", encoding="utf-8"),
                                  logging.StreamHandler()])

    df = pd.read_csv(INPUT_CSV)
    # Largest municipalities first: most likely to be in IJP, best early signal
    names = df.sort_values("population_total", ascending=False)["name"].drop_duplicates().tolist()
    if args.limit:
        names = names[: args.limit]

    if not args.parse_only:
        scraper = UjcScraper(ScrapeConfig(delay=args.delay))
        try:
            stats = scraper.scrape(names, progress=lambda x: tqdm(x, unit="slovo", dynamic_ncols=True))
            log.info("scrape finished: %s", stats)
        except ServerBlocked as exc:
            log.error("%s | stats so far: %s", exc, scraper.stats)

    out = parse_cache(df)
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    log.info("saved %s\n%s", OUTPUT_CSV, out["ujc_number"].value_counts().to_string())


if __name__ == "__main__":
    main()

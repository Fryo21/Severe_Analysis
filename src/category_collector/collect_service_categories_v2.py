from __future__ import annotations

import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "raw-data" / "market-entry" / "service-categories"
RAW_DIR.mkdir(parents=True, exist_ok=True)

RAW_OUTPUT = RAW_DIR / "service_categories_raw_v2.csv"
SUMMARY_OUTPUT = RAW_DIR / "service_categories_platform_summary_v2.csv"
DIAGNOSTICS_OUTPUT = RAW_DIR / "service_category_collection_diagnostics_v2.csv"

TIMEOUT = 30
DEFAULT_DELAY = 1.5
TREATWELL_DELAY = 5.0

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/150.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-GB,en;q=0.9",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

@dataclass
class ServiceRow:
    platform: str
    raw_service: str
    geography_scope: str
    source_url: str
    source_type: str
    collection_date: str
    coverage_note: str

NAV_NOISE = {
    "", "home", "about", "about us", "contact", "contact us",
    "login", "log in", "sign up", "join", "help", "privacy",
    "terms", "careers", "blog", "news", "cookie policy",
    "skip to navigation", "skip to search", "skip to content",
    "skip to footer", "see all", "view all", "more",
    "popular services", "popular categories", "all services",
    "all trades", "all categories", "business", "pricing",
    "reviews", "advice", "advice centre",
}

BROAD_NOISE = {
    "events & entertainers", "health & wellness", "house & home",
    "lessons & training", "professional services", "other",
}

LOCATION_WORDS = {
    "london", "england", "scotland", "wales", "northern ireland",
    "greater london", "edinburgh", "manchester", "glasgow", "leeds",
    "birmingham", "liverpool", "bristol", "australia", "sydney",
    "melbourne", "brisbane", "perth",
}

def today():
    return datetime.now(timezone.utc).date().isoformat()

def normalise_space(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()

def clean_service_name(value):
    text = normalise_space(value)
    text = re.sub(r"\s+near me$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+in\s+[A-Za-z .'-]+$", "", text, flags=re.IGNORECASE)
    return normalise_space(text)

def is_valid_service(value):
    text = clean_service_name(value)
    if not text:
        return False

    lower = text.casefold()

    if lower in NAV_NOISE or lower in BROAD_NOISE or lower in LOCATION_WORDS:
        return False
    if len(text) < 2 or len(text) > 100:
        return False
    if re.fullmatch(r"[A-Z]", text) or re.fullmatch(r"\d+", text):
        return False

    banned = [
        "cookie", "privacy", "terms and conditions",
        "join as a professional", "find a local trusted trader",
    ]
    return not any(x in lower for x in banned)

def unique_clean(values):
    seen = set()
    result = []

    for value in values:
        cleaned = clean_service_name(value)
        if not is_valid_service(cleaned):
            continue

        key = cleaned.casefold()
        if key in seen:
            continue

        seen.add(key)
        result.append(cleaned)

    return sorted(result, key=str.casefold)

def fetch_html(url, delay=DEFAULT_DELAY):
    time.sleep(delay)
    response = SESSION.get(url, timeout=TIMEOUT, allow_redirects=True)
    response.raise_for_status()
    return response.text

def soup_from_url(url, delay=DEFAULT_DELAY):
    return BeautifulSoup(fetch_html(url, delay), "html.parser")

def rows_from_values(platform, values, source_url, source_type, coverage_note):
    return [
        ServiceRow(
            platform=platform,
            raw_service=value,
            geography_scope="London",
            source_url=source_url,
            source_type=source_type,
            collection_date=today(),
            coverage_note=coverage_note,
        )
        for value in unique_clean(values)
    ]

def dedupe_rows(rows):
    seen = set()
    result = []

    for row in rows:
        key = (row.platform.casefold(), row.raw_service.casefold())
        if key in seen:
            continue
        seen.add(key)
        result.append(row)

    return result

def collect_checkatrade():
    rows = []

    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        url = f"https://www.checkatrade.com/sitemap/locations/London/{letter}"

        try:
            soup = soup_from_url(url)
        except Exception:
            continue

        values = []

        for a in soup.find_all("a", href=True):
            text = normalise_space(a.get_text(" ", strip=True))
            href = urljoin(url, a["href"])
            path = urlparse(href).path.casefold()

            if "/sitemap/locations/london/" in path:
                continue

            if is_valid_service(text):
                values.append(text)

        rows.extend(rows_from_values(
            "Checkatrade",
            values,
            url,
            "london_trade_sitemap",
            "London A-Z trade/service sitemap.",
        ))

    return dedupe_rows(rows)

def collect_mybuilder():
    url = "https://www.mybuilder.com/trades"
    soup = soup_from_url(url)
    values = []

    for a in soup.find_all("a", href=True):
        text = normalise_space(a.get_text(" ", strip=True))
        href = urljoin(url, a["href"])
        path = urlparse(href).path.casefold()

        if "/trades/" in path or "/find-trades/" in path:
            values.append(text)

    return rows_from_values(
        "MyBuilder", values, url, "trade_directory",
        "Public trades/professions page."
    )

def collect_rated_people():
    rows = []

    for letter in "abcdefghijklmnopqrstuvwxyz":
        url = f"https://www.ratedpeople.com/sitemap/trades/{letter}"

        try:
            soup = soup_from_url(url)
        except Exception:
            continue

        values = []

        for a in soup.find_all("a", href=True):
            text = normalise_space(a.get_text(" ", strip=True))
            href = urljoin(url, a["href"])
            path = urlparse(href).path.casefold()

            if "/sitemap/trades/" in path:
                continue

            if is_valid_service(text):
                values.append(text)

        if not values:
            values = [li.get_text(" ", strip=True) for li in soup.find_all("li")]

        rows.extend(rows_from_values(
            "Rated People",
            values,
            url,
            "alphabetical_trade_sitemap",
            "Public A-Z trade and skill sitemap.",
        ))

    return dedupe_rows(rows)

def collect_trustatrader():
    url = "https://www.trustatrader.com/trades"
    soup = soup_from_url(url)
    values = []

    for a in soup.find_all("a", href=True):
        text = normalise_space(a.get_text(" ", strip=True))
        href = urljoin(url, a["href"])
        path = urlparse(href).path.casefold()

        if "/trades/" in path and is_valid_service(text):
            values.append(text)

    return rows_from_values(
        "TrustATrader", values, url, "trade_directory",
        "Public trades directory."
    )

def collect_yell():
    url = "https://www.yell.com/"

    try:
        soup = soup_from_url(url)
    except Exception:
        return []

    values = []

    for a in soup.find_all("a", href=True):
        text = normalise_space(a.get_text(" ", strip=True))
        href = urljoin(url, a["href"])
        path = urlparse(href).path.casefold()

        if path.startswith("/s/") or path.startswith("/l/"):
            values.append(text)

    return rows_from_values(
        "Yell", values, url, "public_category_links",
        "Partial public category coverage; validate before analysis."
    )

def collect_bark():
    url = "https://www.bark.com/en/gb/services/"
    soup = soup_from_url(url)
    values = []

    for a in soup.find_all("a", href=True):
        text = normalise_space(a.get_text(" ", strip=True))
        href = urljoin(url, a["href"])
        path = urlparse(href).path.casefold().rstrip("/")

        if not path.startswith("/en/gb/"):
            continue
        if path in {"/en/gb", "/en/gb/services"}:
            continue
        if is_valid_service(text):
            values.append(text)

    return rows_from_values(
        "Bark", values, url, "all_services_page",
        "Bark UK Explore All Services page."
    )

def collect_airtasker():
    url = "https://www.airtasker.com/uk/services/"
    soup = soup_from_url(url)
    values = []

    for a in soup.find_all("a", href=True):
        text = normalise_space(a.get_text(" ", strip=True))
        href = urljoin(url, a["href"])
        path = urlparse(href).path.casefold().rstrip("/")

        if not path.startswith("/uk/services/"):
            continue
        if path == "/uk/services":
            continue
        if is_valid_service(text):
            values.append(text)

    return rows_from_values(
        "Airtasker", values, url, "all_services_page",
        "Airtasker UK service taxonomy."
    )

def collect_fresha():
    urls = [
        "https://www.fresha.com/en-GB/pricing",
        "https://www.fresha.com/en-GB/for-business/features/marketplace",
    ]

    target_terms = [
        "Hair Salon", "Nail Salon", "Barber", "Waxing Salon",
        "Medspa", "Eyebrow Bar", "Massage Salon", "Spa", "Fitness",
        "Personal Trainer", "Therapy Centre", "Tattooing & Piercing",
        "Tanning Studio", "Beauty Salon", "Physical Therapy",
        "Pet Grooming",
    ]

    values = []

    for url in urls:
        try:
            text = soup_from_url(url).get_text(" ", strip=True)
        except Exception:
            continue

        for term in target_terms:
            if re.search(
                rf"\b{re.escape(term)}s?\b",
                text,
                flags=re.IGNORECASE,
            ):
                values.append(term)

    return rows_from_values(
        "Fresha", values, urls[0], "uk_service_verticals",
        "UK public service/business verticals; broad coverage only."
    )

def collect_treatwell():
    url = "https://www.treatwell.co.uk/"
    soup = soup_from_url(url, TREATWELL_DELAY)
    values = []

    for a in soup.find_all("a", href=True):
        text = normalise_space(a.get_text(" ", strip=True))
        href = urljoin(url, a["href"])
        path = urlparse(href).path.casefold()

        if "/places/treatment-" in path and is_valid_service(text):
            values.append(text)

    return rows_from_values(
        "Treatwell", values, url, "treatment_taxonomy",
        "Public treatment-category links from UK marketplace."
    )

def collect_booksy():
    url = "https://booksy.com/en-gb/"
    soup = soup_from_url(url)
    values = []

    for a in soup.find_all("a", href=True):
        text = normalise_space(a.get_text(" ", strip=True))
        href = urljoin(url, a["href"])
        path = urlparse(href).path

        if re.match(r"^/en-gb/s/[^/]+/?$", path, flags=re.IGNORECASE):
            values.append(text)

    return rows_from_values(
        "Booksy", values, url, "service_category_page",
        "Booksy UK public marketplace service categories."
    )
def collect_myjobquote():
    url = "https://www.myjobquote.co.uk/sitemap"
    soup = soup_from_url(url)

    values = []

    for a in soup.find_all("a", href=True):
        text = normalise_space(a.get_text(" ", strip=True))
        href = urljoin(url, a["href"])
        path = urlparse(href).path.casefold()

        if (
            "/tradespeople/" in path
            or "/trades/" in path
        ):
            if is_valid_service(text):
                values.append(text)

    return rows_from_values(
        "MyJobQuote",
        values,
        url,
        "public_trade_sitemap",
        "Public trade/service taxonomy from MyJobQuote.",
    )

COLLECTORS = [
    ("Checkatrade", collect_checkatrade),
    ("MyBuilder", collect_mybuilder),
    ("Rated People", collect_rated_people),
    ("TrustATrader", collect_trustatrader),
    ("Yell", collect_yell),
    ("Bark", collect_bark),
    ("Airtasker", collect_airtasker),
    ("Fresha", collect_fresha),
    ("Treatwell", collect_treatwell),
    ("Booksy", collect_booksy),
    ("MyJobQuote", collect_myjobquote),
]

def main():
    all_rows = []
    diagnostics = []

    print("=" * 78)
    print("SEVERSE SERVICE CATEGORY COLLECTOR V2")
    print("=" * 78)

    for platform, collector in COLLECTORS:
        print(f"\nCollecting: {platform}")

        try:
            rows = dedupe_rows(collector())
            all_rows.extend(rows)

            diagnostics.append({
                "platform": platform,
                "status": "success" if rows else "no_rows",
                "services_collected": len(rows),
                "error": "",
                "collection_date": today(),
            })

            print(f"  Collected {len(rows):,} service/category labels")

        except Exception as exc:
            diagnostics.append({
                "platform": platform,
                "status": "failed",
                "services_collected": 0,
                "error": str(exc),
                "collection_date": today(),
            })

            print(f"  FAILED: {exc}")

    all_rows = dedupe_rows(all_rows)

    raw_df = pd.DataFrame([asdict(row) for row in all_rows])

    if raw_df.empty:
        raise RuntimeError("No service categories were collected.")

    raw_df.to_csv(RAW_OUTPUT, index=False)

    summary_df = (
        raw_df
        .groupby("platform", as_index=False)
        .agg(
            service_labels_collected=("raw_service", "nunique"),
            source_types=(
                "source_type",
                lambda x: " | ".join(sorted(set(x))),
            ),
            coverage_notes=(
                "coverage_note",
                lambda x: " | ".join(sorted(set(x))),
            ),
        )
        .sort_values("service_labels_collected", ascending=False)
    )

    summary_df.to_csv(SUMMARY_OUTPUT, index=False)
    pd.DataFrame(diagnostics).to_csv(DIAGNOSTICS_OUTPUT, index=False)

    print("\n" + "=" * 78)
    print("V2 COLLECTION COMPLETE")
    print("=" * 78)

    print(f"\nTotal platform/service rows: {len(raw_df):,}")
    print(f"Unique raw service labels: {raw_df['raw_service'].nunique():,}")

    print("\nBy platform:\n")
    print(
        summary_df[
            ["platform", "service_labels_collected", "source_types"]
        ].to_string(index=False)
    )

    print("\nFiles written:")
    print(f"  {RAW_OUTPUT}")
    print(f"  {SUMMARY_OUTPUT}")
    print(f"  {DIAGNOSTICS_OUTPUT}")

    print("\nV1 files were not overwritten.")
    print(
        "\nNEXT STEP: quality-check V2, then standardise "
        "raw service names into canonical service categories."
    )

if __name__ == "__main__":
    main()

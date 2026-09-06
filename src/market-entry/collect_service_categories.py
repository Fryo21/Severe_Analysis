from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "raw-data"
    / "market-entry"
    / "service-categories"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_OUTPUT = OUTPUT_DIR / "service_categories_raw.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "service_categories_platform_summary.csv"
DIAGNOSTICS_OUTPUT = OUTPUT_DIR / "service_category_collection_diagnostics.csv"

TIMEOUT = 30
DEFAULT_DELAY_SECONDS = 1.5
TREATWELL_DELAY_SECONDS = 5.0

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
    source_url: str
    source_type: str
    collection_date: str
    coverage_note: str = ""


GENERIC_TEXT = {
    "",
    "home",
    "services",
    "service",
    "all services",
    "all trades",
    "all categories",
    "see all",
    "view all",
    "more",
    "more...",
    "search",
    "find trades",
    "find a trade",
    "popular services",
    "popular categories",
    "current popular trades",
    "other",
    "gift card",
    "lookbook",
    "the treatment files",
    "business",
    "business services",
    "house & home",
    "health & wellness",
    "health & wellbeing",
    "lessons & training",
    "events & entertainers",
}

CITY_WORDS = {
    "london",
    "manchester",
    "birmingham",
    "glasgow",
    "edinburgh",
    "leeds",
    "liverpool",
    "bristol",
    "sheffield",
    "cardiff",
    "nottingham",
    "coventry",
    "brighton",
    "bradford",
    "portsmouth",
}


def today():
    return datetime.now(timezone.utc).date().isoformat()


def normalise_space(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def clean_service_name(value):
    text = normalise_space(value)
    text = re.sub(
        r"\s+in\s+(London|UK|United Kingdom)$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\s+near me$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return normalise_space(text)


def looks_like_service_name(value):
    text = clean_service_name(value)

    if not text:
        return False

    lower = text.lower()

    if lower in GENERIC_TEXT or lower in CITY_WORDS:
        return False

    if len(text) < 2 or len(text) > 100:
        return False

    if re.fullmatch(r"[a-zA-Z]", text) or re.fullmatch(r"\d+", text):
        return False

    banned = [
        "cookie",
        "privacy",
        "terms",
        "login",
        "log in",
        "sign up",
        "contact us",
    ]

    return not any(token in lower for token in banned)


def unique_clean(values):
    seen = set()
    result = []

    for value in values:
        value = clean_service_name(value)

        if not looks_like_service_name(value):
            continue

        key = value.casefold()

        if key in seen:
            continue

        seen.add(key)
        result.append(value)

    return sorted(result, key=str.casefold)


def fetch_html(url, delay_seconds=DEFAULT_DELAY_SECONDS):
    time.sleep(delay_seconds)

    response = SESSION.get(url, timeout=TIMEOUT)
    response.raise_for_status()

    return response.text


def soup_from_url(url, delay_seconds=DEFAULT_DELAY_SECONDS):
    return BeautifulSoup(
        fetch_html(url, delay_seconds),
        "html.parser",
    )


def rows_from_values(
    platform,
    values,
    source_url,
    source_type,
    coverage_note,
):
    return [
        ServiceRow(
            platform=platform,
            raw_service=value,
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
        key = (
            row.platform.casefold(),
            row.raw_service.casefold(),
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(row)

    return result


def extract_links_by_path(
    soup,
    base_url,
    path_predicate: Callable[[str], bool],
):
    values = []

    for a in soup.find_all("a", href=True):
        text = normalise_space(a.get_text(" ", strip=True))
        href = urljoin(base_url, a["href"])
        path = urlparse(href).path

        if path_predicate(path) and looks_like_service_name(text):
            values.append(text)

    return unique_clean(values)


def collect_checkatrade():
    url = "https://join.checkatrade.com/what-you-get/"
    soup = soup_from_url(url)

    values = []

    for node in soup.find_all(["li", "option", "a"]):
        text = normalise_space(node.get_text(" ", strip=True))
        if looks_like_service_name(text):
            values.append(text)

    return rows_from_values(
        "Checkatrade",
        values,
        url,
        "public_category_page",
        "Checkatrade public category/onboarding page.",
    )


def collect_mybuilder():
    url = "https://www.mybuilder.com/trades"
    soup = soup_from_url(url)

    values = extract_links_by_path(
        soup,
        url,
        lambda path:
        "/trades/" in path
        or "/find-trades/" in path,
    )

    if not values:
        values = [
            li.get_text(" ", strip=True)
            for li in soup.find_all("li")
        ]

    return rows_from_values(
        "MyBuilder",
        values,
        url,
        "public_trade_directory",
        "MyBuilder public trades/professions page.",
    )


def collect_rated_people():
    rows = []

    for letter in "abcdefghijklmnopqrstuvwxyz":
        url = (
            "https://www.ratedpeople.com/"
            f"sitemap/trades/{letter}"
        )

        try:
            soup = soup_from_url(url)
        except Exception as exc:
            print(f"  Rated People {letter}: {exc}")
            continue

        values = [
            li.get_text(" ", strip=True)
            for li in soup.find_all("li")
        ]

        rows.extend(
            rows_from_values(
                "Rated People",
                values,
                url,
                "alphabetical_trade_sitemap",
                "Rated People public trade-and-skill sitemap.",
            )
        )

    return dedupe_rows(rows)


def collect_trustatrader():
    url = "https://www.trustatrader.com/trades"
    soup = soup_from_url(url)

    values = extract_links_by_path(
        soup,
        url,
        lambda path:
        "/trades/" in path,
    )

    return rows_from_values(
        "TrustATrader",
        values,
        url,
        "public_trade_directory",
        "TrustATrader public trades directory.",
    )


def collect_yell():
    url = "https://www.yell.com/"
    soup = soup_from_url(url)

    values = extract_links_by_path(
        soup,
        url,
        lambda path:
        path.startswith("/s/")
        or path.startswith("/l/"),
    )

    if not values:
        values = [
            li.get_text(" ", strip=True)
            for li in soup.find_all("li")
        ]

    return rows_from_values(
        "Yell",
        values,
        url,
        "public_category_taxonomy",
        (
            "Public Yell category labels available from its pages. "
            "Not claimed to represent Yell's complete internal classification."
        ),
    )


def collect_bark():
    url = "https://www.bark.com/en/gb/services/"
    soup = soup_from_url(url)

    values = extract_links_by_path(
        soup,
        url,
        lambda path:
        path.startswith("/en/gb/")
        and path.rstrip("/")
        not in {"/en/gb", "/en/gb/services"},
    )

    return rows_from_values(
        "Bark",
        values,
        url,
        "public_all_services_page",
        "Bark UK Explore All Services page.",
    )


def collect_airtasker():
    url = "https://www.airtasker.com/uk/services/"
    soup = soup_from_url(url)

    values = extract_links_by_path(
        soup,
        url,
        lambda path:
        path.startswith("/uk/services/")
        and path.rstrip("/") != "/uk/services",
    )

    return rows_from_values(
        "Airtasker",
        values,
        url,
        "public_all_services_page",
        (
            "Airtasker UK services taxonomy. "
            "Recent task counts will be collected separately as demand data."
        ),
    )


def collect_fresha():
    urls = [
        "https://www.fresha.com/en-GB/pricing",
        "https://www.fresha.com/en-GB/for-business/features/marketplace",
    ]

    values = []

    target_terms = [
        "Hair Salon",
        "Nail Salon",
        "Barber",
        "Waxing Salon",
        "Medspa",
        "Eyebrow Bar",
        "Massage Salon",
        "Spa",
        "Fitness",
        "Personal Trainer",
        "Salon",
        "Therapy Centre",
        "Tattooing & Piercing",
        "Tanning Studio",
        "Beauty Salon",
        "Physical Therapy",
    ]

    for url in urls:
        try:
            text = soup_from_url(url).get_text(" ", strip=True)
        except Exception as exc:
            print(f"  Fresha {url}: {exc}")
            continue

        for term in target_terms:
            if re.search(
                rf"\b{re.escape(term)}s?\b",
                text,
                flags=re.IGNORECASE,
            ):
                values.append(term)

    return rows_from_values(
        "Fresha",
        values,
        urls[0],
        "public_service_verticals",
        (
            "Fresha public UK service/business verticals. "
            "Detailed treatments will be added later for shortlisted verticals."
        ),
    )


def collect_treatwell():
    homepage = "https://www.treatwell.co.uk/"
    soup = soup_from_url(
        homepage,
        TREATWELL_DELAY_SECONDS,
    )

    values = []

    broad_links = []

    for a in soup.find_all("a", href=True):
        text = normalise_space(a.get_text(" ", strip=True))
        href = urljoin(homepage, a["href"])

        if (
            "/places/treatment-" in urlparse(href).path
            and looks_like_service_name(text)
        ):
            broad_links.append((text, href))
            values.append(text)

    seen_urls = []

    for _, href in broad_links:
        if href not in seen_urls:
            seen_urls.append(href)

    for href in seen_urls[:12]:
        try:
            sub_soup = soup_from_url(
                href,
                TREATWELL_DELAY_SECONDS,
            )
        except Exception as exc:
            print(f"  Treatwell {href}: {exc}")
            continue

        for a in sub_soup.find_all("a", href=True):
            text = normalise_space(a.get_text(" ", strip=True))
            target = urljoin(href, a["href"])

            if (
                "/places/treatment-" in urlparse(target).path
                and looks_like_service_name(text)
            ):
                values.append(text)

    return rows_from_values(
        "Treatwell",
        values,
        homepage,
        "public_treatment_taxonomy",
        "Treatwell public treatment-family and treatment-menu links.",
    )


def collect_booksy():
    url = "https://booksy.com/en-gb/"
    soup = soup_from_url(url)

    values = extract_links_by_path(
        soup,
        url,
        lambda path:
        bool(
            re.match(
                r"^/en-gb/s/[^/]+/?$",
                path,
                flags=re.IGNORECASE,
            )
        ),
    )

    return rows_from_values(
        "Booksy",
        values,
        url,
        "public_service_categories",
        "Booksy UK public marketplace service categories.",
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
]


def main():
    all_rows = []
    diagnostics = []

    print("=" * 72)
    print("SEVERSE SERVICE CATEGORY COLLECTION")
    print("=" * 72)

    for platform, collector in COLLECTORS:
        print(f"\nCollecting: {platform}")

        try:
            rows = dedupe_rows(collector())
            all_rows.extend(rows)

            diagnostics.append(
                {
                    "platform": platform,
                    "status": "success",
                    "services_collected": len(rows),
                    "error": "",
                    "collection_date": today(),
                }
            )

            print(
                f"  Collected {len(rows):,} service/category labels"
            )

        except Exception as exc:
            diagnostics.append(
                {
                    "platform": platform,
                    "status": "failed",
                    "services_collected": 0,
                    "error": str(exc),
                    "collection_date": today(),
                }
            )

            print(f"  FAILED: {exc}")

    all_rows = dedupe_rows(all_rows)

    raw_df = pd.DataFrame(
        [row.__dict__ for row in all_rows]
    )

    if raw_df.empty:
        raise RuntimeError(
            "No service categories were collected."
        )

    raw_df.to_csv(
        RAW_OUTPUT,
        index=False,
    )

    summary_df = (
        raw_df
        .groupby("platform", as_index=False)
        .agg(
            service_labels_collected=(
                "raw_service",
                "nunique",
            ),
            source_types=(
                "source_type",
                lambda x:
                " | ".join(
                    sorted(set(x))
                ),
            ),
        )
        .sort_values(
            "service_labels_collected",
            ascending=False,
        )
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    pd.DataFrame(
        diagnostics
    ).to_csv(
        DIAGNOSTICS_OUTPUT,
        index=False,
    )

    print("\n" + "=" * 72)
    print("COLLECTION COMPLETE")
    print("=" * 72)

    print(
        f"\nTotal platform/service rows: {len(raw_df):,}"
    )

    print(
        f"Unique raw service labels: "
        f"{raw_df['raw_service'].nunique():,}"
    )

    print("\nBy platform:")
    print(summary_df.to_string(index=False))

    print("\nFiles written:")
    print(f"  {RAW_OUTPUT}")
    print(f"  {SUMMARY_OUTPUT}")
    print(f"  {DIAGNOSTICS_OUTPUT}")

    print(
        "\nNEXT STEP: standardise raw service names "
        "into canonical service categories before ranking anything."
    )


if __name__ == "__main__":
    main()

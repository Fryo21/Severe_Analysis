from __future__ import annotations

import re
import time
from dataclasses import dataclass, asdict
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
    / "source-discovery"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_MAP_OUTPUT = OUTPUT_DIR / "platform_source_map.csv"
DISCOVERY_OUTPUT = OUTPUT_DIR / "platform_source_discovery_details.csv"

TIMEOUT = 20
REQUEST_DELAY_SECONDS = 1.5

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

PLATFORMS = {
    "Checkatrade": {
        "base_url": "https://www.checkatrade.com/",
        "candidate_pages": [
            "https://www.checkatrade.com/sitemap",
            "https://www.checkatrade.com/sitemap/locations/London/A",
        ],
    },
    "MyBuilder": {
        "base_url": "https://www.mybuilder.com/",
        "candidate_pages": [
            "https://www.mybuilder.com/trades",
        ],
    },
    "Rated People": {
        "base_url": "https://www.ratedpeople.com/",
        "candidate_pages": [
            "https://www.ratedpeople.com/sitemap/trades/a",
            "https://www.ratedpeople.com/sitemap/trades/b",
        ],
    },
    "TrustATrader": {
        "base_url": "https://www.trustatrader.com/",
        "candidate_pages": [
            "https://www.trustatrader.com/trades",
        ],
    },
    "Yell": {
        "base_url": "https://www.yell.com/",
        "candidate_pages": [
            "https://www.yell.com/",
        ],
    },
    "Bark": {
        "base_url": "https://www.bark.com/en/gb/",
        "candidate_pages": [
            "https://www.bark.com/en/gb/services/",
        ],
    },
    "Airtasker": {
        "base_url": "https://www.airtasker.com/uk/",
        "candidate_pages": [
            "https://www.airtasker.com/uk/services/",
        ],
    },
    "Fresha": {
        "base_url": "https://www.fresha.com/",
        "candidate_pages": [
            "https://www.fresha.com/en-GB/pricing",
            "https://www.fresha.com/en-GB/for-business/features/marketplace",
        ],
    },
    "Treatwell": {
        "base_url": "https://www.treatwell.co.uk/",
        "candidate_pages": [
            "https://www.treatwell.co.uk/",
        ],
    },
    "Booksy": {
        "base_url": "https://booksy.com/en-gb/",
        "candidate_pages": [
            "https://booksy.com/en-gb/",
        ],
    },
}

SERVICE_WORDS = {
    "plumber", "plumbing", "electrician", "cleaner", "cleaning",
    "gardener", "gardening", "roofer", "roofing", "builder", "handyman",
    "locksmith", "carpenter", "painting", "decorator", "massage", "hair",
    "barber", "beauty", "nails", "waxing", "spa", "fitness", "therapy",
    "tattoo", "repair", "installation", "removal", "moving", "assembly",
    "pet", "grooming",
}

NAVIGATION_WORDS = {
    "home", "about", "contact", "login", "log in", "sign up", "join",
    "help", "privacy", "terms", "blog", "careers", "news", "cookie",
    "business", "pricing", "reviews", "advice",
}


@dataclass
class DiscoveryResult:
    platform: str
    source_type: str
    source_url: str
    status_code: int | None
    reachable: bool
    service_like_links: int
    jsonld_blocks: int
    sitemap_detected: bool
    sample_services: str
    score: float
    confidence: str
    notes: str


def today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def normalise_space(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def fetch(url: str) -> requests.Response:
    time.sleep(REQUEST_DELAY_SECONDS)
    return SESSION.get(
        url,
        timeout=TIMEOUT,
        allow_redirects=True,
    )


def robots_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    return f"{parsed.scheme}://{parsed.netloc}/robots.txt"


def discover_sitemaps(base_url: str) -> list[str]:
    try:
        response = fetch(robots_url(base_url))
        if response.status_code != 200:
            return []
        results = []
        for line in response.text.splitlines():
            if line.lower().startswith("sitemap:"):
                value = line.split(":", 1)[1].strip()
                if value:
                    results.append(value)
        return list(dict.fromkeys(results))
    except Exception:
        return []


def looks_like_service_text(text: str) -> bool:
    text = normalise_space(text)
    if not text:
        return False

    lower = text.casefold()

    if len(text) < 3 or len(text) > 90:
        return False

    if lower in NAVIGATION_WORDS:
        return False

    if re.fullmatch(r"[A-Z]", text):
        return False

    if re.fullmatch(r"\d+", text):
        return False

    return True


def service_signal_score(text: str, href: str) -> int:
    text_lower = text.casefold()
    href_lower = href.casefold()

    score = 0

    if any(word in text_lower for word in SERVICE_WORDS):
        score += 2

    if any(
        token in href_lower
        for token in (
            "/service",
            "/services",
            "/trade",
            "/trades",
            "/treatment",
            "/category",
            "/categories",
            "/profession",
            "/skill",
        )
    ):
        score += 2

    if "/s/" in href_lower:
        score += 1

    if text_lower in NAVIGATION_WORDS:
        score -= 2

    return score


def extract_service_like_links(
    soup: BeautifulSoup,
    base_url: str,
) -> list[tuple[str, str, int]]:
    rows = []
    seen = set()

    for anchor in soup.find_all("a", href=True):
        text = normalise_space(anchor.get_text(" ", strip=True))

        if not looks_like_service_text(text):
            continue

        href = urljoin(base_url, anchor["href"])
        score = service_signal_score(text, href)

        if score <= 0:
            continue

        key = (text.casefold(), href)
        if key in seen:
            continue

        seen.add(key)
        rows.append((text, href, score))

    return sorted(rows, key=lambda row: row[2], reverse=True)


def count_jsonld(soup: BeautifulSoup) -> int:
    return len(
        soup.find_all(
            "script",
            attrs={
                "type": re.compile(
                    r"application/ld\+json",
                    flags=re.IGNORECASE,
                )
            },
        )
    )


def inspect_page(
    platform: str,
    source_url: str,
    source_type: str,
) -> DiscoveryResult:
    try:
        response = fetch(source_url)
        status_code = response.status_code

        if not response.ok:
            return DiscoveryResult(
                platform=platform,
                source_type=source_type,
                source_url=source_url,
                status_code=status_code,
                reachable=False,
                service_like_links=0,
                jsonld_blocks=0,
                sitemap_detected=False,
                sample_services="",
                score=0,
                confidence="LOW",
                notes=f"HTTP {status_code}",
            )

        soup = BeautifulSoup(response.text, "html.parser")
        links = extract_service_like_links(soup, source_url)
        jsonld_blocks = count_jsonld(soup)

        sitemap_detected = (
            "sitemap" in source_url.casefold()
            or bool(re.search(r"sitemap", response.text, re.IGNORECASE))
        )

        service_count = len(links)
        score = 0.0

        if service_count >= 100:
            score += 5
        elif service_count >= 40:
            score += 4
        elif service_count >= 15:
            score += 3
        elif service_count >= 5:
            score += 2
        elif service_count > 0:
            score += 1

        if sitemap_detected:
            score += 2

        if jsonld_blocks > 0:
            score += 1

        if any(
            token in source_url.casefold()
            for token in ("/services", "/trades", "/sitemap", "/categories")
        ):
            score += 2

        if score >= 7:
            confidence = "HIGH"
        elif score >= 4:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        sample_services = " | ".join(text for text, _, _ in links[:10])

        return DiscoveryResult(
            platform=platform,
            source_type=source_type,
            source_url=source_url,
            status_code=status_code,
            reachable=True,
            service_like_links=service_count,
            jsonld_blocks=jsonld_blocks,
            sitemap_detected=sitemap_detected,
            sample_services=sample_services,
            score=score,
            confidence=confidence,
            notes="",
        )

    except Exception as exc:
        return DiscoveryResult(
            platform=platform,
            source_type=source_type,
            source_url=source_url,
            status_code=None,
            reachable=False,
            service_like_links=0,
            jsonld_blocks=0,
            sitemap_detected=False,
            sample_services="",
            score=0,
            confidence="LOW",
            notes=str(exc),
        )


def inspect_platform(
    platform: str,
    config: dict,
) -> list[DiscoveryResult]:
    results = []

    for sitemap_url in discover_sitemaps(config["base_url"])[:5]:
        results.append(
            inspect_page(
                platform,
                sitemap_url,
                "robots_sitemap",
            )
        )

    for url in config["candidate_pages"]:
        if "sitemap" in url.casefold():
            source_type = "sitemap_page"
        elif "service" in url.casefold():
            source_type = "service_page"
        elif "trade" in url.casefold():
            source_type = "trade_page"
        elif "treatment" in url.casefold():
            source_type = "treatment_page"
        else:
            source_type = "candidate_page"

        results.append(
            inspect_page(
                platform,
                url,
                source_type,
            )
        )

    if config["base_url"] not in config["candidate_pages"]:
        results.append(
            inspect_page(
                platform,
                config["base_url"],
                "homepage_fallback",
            )
        )

    return results


def choose_best_source(
    results: Iterable[DiscoveryResult],
) -> DiscoveryResult | None:
    usable = [row for row in results if row.reachable]

    if not usable:
        return None

    return sorted(
        usable,
        key=lambda row: (
            row.score,
            row.service_like_links,
            row.jsonld_blocks,
        ),
        reverse=True,
    )[0]


def main():
    all_results = []
    source_map_rows = []

    print("=" * 78)
    print("SEVERSE PLATFORM SOURCE DISCOVERY MAPPER")
    print("=" * 78)

    for platform, config in PLATFORMS.items():
        print(f"\nInspecting: {platform}")

        platform_results = inspect_platform(platform, config)
        all_results.extend(platform_results)

        best = choose_best_source(platform_results)

        if best is None:
            source_map_rows.append(
                {
                    "platform": platform,
                    "best_source_type": "",
                    "best_source_url": "",
                    "service_like_links": 0,
                    "jsonld_blocks": 0,
                    "score": 0,
                    "confidence": "LOW",
                    "status": "NO_USABLE_SOURCE",
                    "sample_services": "",
                    "collection_date": today(),
                }
            )
            print("  No usable source found")
            continue

        status = (
            "READY_FOR_V2"
            if best.confidence in {"HIGH", "MEDIUM"}
            and best.service_like_links > 0
            else "NEEDS_CUSTOM_REVIEW"
        )

        source_map_rows.append(
            {
                "platform": platform,
                "best_source_type": best.source_type,
                "best_source_url": best.source_url,
                "service_like_links": best.service_like_links,
                "jsonld_blocks": best.jsonld_blocks,
                "score": best.score,
                "confidence": best.confidence,
                "status": status,
                "sample_services": best.sample_services,
                "collection_date": today(),
            }
        )

        print(f"  Best source: {best.source_type}")
        print(f"  URL: {best.source_url}")
        print(f"  Service-like links: {best.service_like_links}")
        print(f"  Confidence: {best.confidence}")

        if best.sample_services:
            print(f"  Sample: {best.sample_services}")

    details_df = pd.DataFrame(
        [asdict(result) for result in all_results]
    )

    source_map_df = pd.DataFrame(source_map_rows)

    details_df.to_csv(DISCOVERY_OUTPUT, index=False)
    source_map_df.to_csv(SOURCE_MAP_OUTPUT, index=False)

    print("\n" + "=" * 78)
    print("DISCOVERY COMPLETE")
    print("=" * 78)

    print("\nBest source per platform:\n")

    display_columns = [
        "platform",
        "best_source_type",
        "service_like_links",
        "confidence",
        "status",
    ]

    print(
        source_map_df[
            display_columns
        ].to_string(index=False)
    )

    print("\nFiles written:")
    print(f"  {SOURCE_MAP_OUTPUT}")
    print(f"  {DISCOVERY_OUTPUT}")

    print(
        "\nNEXT STEP: review platform_source_map.csv. "
        "READY_FOR_V2 platforms can move to extraction. "
        "NEEDS_CUSTOM_REVIEW platforms need a platform-specific rule."
    )


if __name__ == "__main__":
    main()

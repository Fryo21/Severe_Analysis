import re
import time
import random
import hashlib
import pandas as pd

from playwright.sync_api import sync_playwright


COMPETITOR = "Bark"
DOMAIN = "bark.com"

BASE_URL = f"https://uk.trustpilot.com/review/{DOMAIN}"

MAX_REVIEWS = 200

OUTPUT_FILE = "bark_trustpilot_reviews.csv"


def get_text(element, selectors):
    """
    Try multiple selectors and return the first text found.
    """

    for selector in selectors:
        try:
            locator = element.locator(selector)

            if locator.count() > 0:
                text = locator.first.inner_text().strip()

                if text:
                    return text

        except Exception:
            continue

    return None


def extract_rating(card):
    """
    Extract star rating from image alt text such as:
    'Rated 5 out of 5 stars'
    """

    try:
        images = card.locator("img")

        for i in range(images.count()):
            alt = images.nth(i).get_attribute("alt")

            if alt and "Rated" in alt:
                match = re.search(r"Rated\s+(\d)", alt)

                if match:
                    return int(match.group(1))

    except Exception:
        pass

    return None


def extract_date(card):
    """
    Extract review date.
    """

    try:
        time_element = card.locator("time")

        if time_element.count() > 0:

            datetime_value = time_element.first.get_attribute("datetime")

            if datetime_value:
                return datetime_value

            return time_element.first.inner_text().strip()

    except Exception:
        pass

    return None


def extract_review_url(card):
    """
    Extract the Trustpilot review URL where available.
    """

    try:
        links = card.locator('a[href*="/reviews/"]')

        if links.count() > 0:
            href = links.first.get_attribute("href")

            if href:

                if href.startswith("http"):
                    return href

                return f"https://uk.trustpilot.com{href}"

    except Exception:
        pass

    return None


def create_review_id(review):
    """
    Generate an ID if Trustpilot URL isn't available.
    """

    content = (
        str(review.get("reviewer"))
        + str(review.get("date"))
        + str(review.get("review_text"))
    )

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()[:16]


def scrape_reviews():

    reviews = []

    page_number = 1

    with sync_playwright() as playwright:

        browser = playwright.chromium.launch(
            headless=False
        )

        page = browser.new_page()

        while len(reviews) < MAX_REVIEWS:

            url = f"{BASE_URL}?page={page_number}"

            print(
                f"\nScraping page {page_number}..."
            )

            response = page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=60000
            )

            if response:

                status = response.status

                print(f"HTTP status: {status}")

                if status in [403, 429]:

                    print(
                        "Trustpilot blocked or rate-limited the request."
                    )

                    break

            page.wait_for_timeout(2000)

            # Primary current Trustpilot-style selector

            cards = page.locator(
                '[data-service-review-card-paper]'
            )

            # Fallback

            if cards.count() == 0:

                cards = page.locator("article")

            card_count = cards.count()

            print(
                f"Found {card_count} possible reviews"
            )

            if card_count == 0:

                print(
                    "No reviews found. Trustpilot's page "
                    "structure may have changed."
                )

                break

            new_reviews = 0

            for i in range(card_count):

                card = cards.nth(i)

                reviewer = get_text(
                    card,
                    [
                        '[data-consumer-name-typography]',
                        'span[class*="consumer"]'
                    ]
                )

                title = get_text(
                    card,
                    [
                        '[data-service-review-title-typography]',
                        "h2"
                    ]
                )

                review_text = get_text(
                    card,
                    [
                        '[data-service-review-text-typography]',
                        "p"
                    ]
                )

                rating = extract_rating(card)

                date = extract_date(card)

                review_url = extract_review_url(card)

                # Skip elements that clearly aren't reviews

                if not review_text and not title:
                    continue

                review = {
                    "competitor": COMPETITOR,
                    "domain": DOMAIN,
                    "source": "Trustpilot",
                    "reviewer": reviewer,
                    "rating": rating,
                    "title": title,
                    "review_text": review_text,
                    "date": date,
                    "review_url": review_url,
                }

                review["review_id"] = (
                    review_url
                    if review_url
                    else create_review_id(review)
                )

                existing_ids = {
                    r["review_id"]
                    for r in reviews
                }

                if review["review_id"] not in existing_ids:

                    reviews.append(review)

                    new_reviews += 1

                if len(reviews) >= MAX_REVIEWS:
                    break

            print(
                f"Added {new_reviews} new reviews"
            )

            print(
                f"Total collected: {len(reviews)}"
            )

            if new_reviews == 0:

                print(
                    "No new reviews found. Stopping."
                )

                break

            page_number += 1

            # Be polite to the website

            time.sleep(
                random.uniform(2, 4)
            )

        browser.close()

    return reviews


def save_reviews(reviews):

    df = pd.DataFrame(reviews)

    df = df.drop_duplicates(
        subset=["review_id"]
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print(
        f"\nSaved {len(df)} reviews to {OUTPUT_FILE}"
    )

    print("\nRatings collected:")

    print(
        df["rating"]
        .value_counts(dropna=False)
        .sort_index()
    )


if __name__ == "__main__":

    reviews = scrape_reviews()

    if reviews:

        save_reviews(reviews)

    else:

        print("No reviews collected.")
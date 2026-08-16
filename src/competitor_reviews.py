import os
import re
import time
import requests

from google_play_scraper import reviews, Sort


# ============================================================
# SETTINGS
# ============================================================

# UK reviews
COUNTRY = "gb"
LANGUAGE = "en"

# Number of Google Play reviews to try to collect PER APP.
# Change this if needed.
MAX_GOOGLE_REVIEWS = 1500

# Apple exposes recent reviews through pages.
# Maximum commonly available through this feed is 10 pages.
MAX_APPLE_PAGES = 10

# Where your files will be saved
OUTPUT_FOLDER = "competitor_reviews"


# ============================================================
# APPS TO ANALYSE
# ============================================================

APPS = [
    {
        "name": "Yell Business",
        "google_id": "com.yell.business",
        "apple_id": "1078222002"
    },
    {
        "name": "Checkatrade for Trades",
        "google_id": "com.checkatrade.tradeapp",
        "apple_id": "1498194074"
    },
    {
        "name": "Bark for Professionals",
        "google_id": "com.barkpro",
        "apple_id": "1206370169"
    },
    {
        "name": "Yelp for Business",
        "google_id": "com.yelp.android.biz",
        "apple_id": "936983378"
    }
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_filename(text):
    """
    Convert app names into safe filenames.
    Example:
        'Yell Business' -> 'yell_business'
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def clean_text(value):
    """
    Clean text so reviews are easier to read.
    """
    if value is None:
        return ""

    value = str(value)

    value = value.replace("\r", " ")
    value = value.replace("\n", " ")

    return " ".join(value.split())


def write_review(file, review):
    """
    Write one review in a readable text format.
    """

    file.write("=" * 80 + "\n")

    file.write(f"APP: {review.get('app', '')}\n")
    file.write(f"STORE: {review.get('store', '')}\n")
    file.write(f"COUNTRY: {review.get('country', '')}\n")
    file.write(f"RATING: {review.get('rating', '')}/5\n")
    file.write(f"DATE: {review.get('date', '')}\n")
    file.write(f"VERSION: {review.get('version', '')}\n")
    file.write(f"AUTHOR: {review.get('author', '')}\n")

    title = review.get("title")

    if title:
        file.write(f"TITLE: {title}\n")

    file.write("\nREVIEW:\n")
    file.write(review.get("text", "") + "\n")

    developer_reply = review.get("developer_reply")

    if developer_reply:
        file.write("\nDEVELOPER REPLY:\n")
        file.write(developer_reply + "\n")

    file.write("\n")


# ============================================================
# GOOGLE PLAY
# ============================================================

def download_google_reviews(app_name, app_id):

    print()
    print("=" * 60)
    print(f"GOOGLE PLAY: {app_name}")
    print("=" * 60)

    all_reviews = []

    continuation_token = None

    while len(all_reviews) < MAX_GOOGLE_REVIEWS:

        try:

            remaining = MAX_GOOGLE_REVIEWS - len(all_reviews)

            batch_size = min(200, remaining)

            result, continuation_token = reviews(
                app_id,
                lang=LANGUAGE,
                country=COUNTRY,
                sort=Sort.NEWEST,
                count=batch_size,
                continuation_token=continuation_token
            )

            if not result:
                print("No more Google Play reviews found.")
                break

            for item in result:

                review = {
                    "app": app_name,
                    "store": "Google Play",
                    "country": COUNTRY.upper(),

                    "rating": item.get("score"),

                    "date": str(item.get("at", "")),

                    "version": clean_text(
                        item.get("reviewCreatedVersion")
                    ),

                    "author": clean_text(
                        item.get("userName")
                    ),

                    "title": "",

                    "text": clean_text(
                        item.get("content")
                    ),

                    "developer_reply": clean_text(
                        item.get("replyContent")
                    ),

                    "thumbs_up": item.get("thumbsUpCount"),

                    "review_id": item.get("reviewId")
                }

                # Ignore reviews with no written text
                if review["text"]:
                    all_reviews.append(review)

            print(
                f"Collected {len(all_reviews)} "
                f"Google Play reviews..."
            )

            if continuation_token is None:
                break

            time.sleep(1)

        except Exception as error:

            print(f"Google Play error: {error}")

            break

    return all_reviews


# ============================================================
# APPLE APP STORE
# ============================================================

def download_apple_reviews(app_name, app_id):

    print()
    print("=" * 60)
    print(f"APPLE APP STORE: {app_name}")
    print("=" * 60)

    all_reviews = []

    seen_ids = set()

    for page in range(1, MAX_APPLE_PAGES + 1):

        url = (
            f"https://itunes.apple.com/"
            f"{COUNTRY}/rss/customerreviews/"
            f"page={page}/"
            f"id={app_id}/"
            f"sortby=mostrecent/json"
        )

        try:

            response = requests.get(
                url,
                headers={
                    "User-Agent":
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                },
                timeout=30
            )

            if response.status_code != 200:

                print(
                    f"Apple page {page}: "
                    f"HTTP {response.status_code}"
                )

                continue

            data = response.json()

            entries = (
                data
                .get("feed", {})
                .get("entry", [])
            )

            if not entries:

                print(
                    f"No reviews on Apple page {page}."
                )

                break

            page_reviews = 0

            for item in entries:

                # Apple sometimes includes metadata entries
                # that are not actual reviews.
                if "im:rating" not in item:
                    continue

                review_id = (
                    item
                    .get("id", {})
                    .get("label", "")
                )

                if review_id in seen_ids:
                    continue

                seen_ids.add(review_id)

                rating = (
                    item
                    .get("im:rating", {})
                    .get("label", "")
                )

                review = {

                    "app": app_name,

                    "store": "Apple App Store",

                    "country": COUNTRY.upper(),

                    "rating": rating,

                    "date": clean_text(
                        item
                        .get("updated", {})
                        .get("label", "")
                    ),

                    "version": clean_text(
                        item
                        .get("im:version", {})
                        .get("label", "")
                    ),

                    "author": clean_text(
                        item
                        .get("author", {})
                        .get("name", {})
                        .get("label", "")
                    ),

                    "title": clean_text(
                        item
                        .get("title", {})
                        .get("label", "")
                    ),

                    "text": clean_text(
                        item
                        .get("content", {})
                        .get("label", "")
                    ),

                    "developer_reply": "",

                    "review_id": review_id
                }

                if review["text"]:

                    all_reviews.append(review)

                    page_reviews += 1

            print(
                f"Apple page {page}: "
                f"{page_reviews} reviews "
                f"({len(all_reviews)} total)"
            )

            time.sleep(1)

        except Exception as error:

            print(
                f"Apple page {page} error: {error}"
            )

    return all_reviews


# ============================================================
# SAVE FILES
# ============================================================

def save_reviews(app_name, store_name, reviews):

    filename_base = safe_filename(app_name)

    store_base = safe_filename(store_name)

    filename = os.path.join(
        OUTPUT_FOLDER,
        f"{filename_base}_{store_base}_reviews.txt"
    )

    negative_filename = os.path.join(
        OUTPUT_FOLDER,
        f"{filename_base}_{store_base}_negative_reviews.txt"
    )

    # --------------------------------------------------------
    # ALL REVIEWS
    # --------------------------------------------------------

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            f"{app_name} - {store_name} Reviews\n"
        )

        file.write(
            f"Total reviews collected: {len(reviews)}\n"
        )

        file.write(
            f"Country: {COUNTRY.upper()}\n\n"
        )

        for review in reviews:

            write_review(file, review)

    # --------------------------------------------------------
    # NEGATIVE REVIEWS ONLY
    # --------------------------------------------------------

    negative_reviews = []

    for review in reviews:

        try:

            rating = int(review["rating"])

            if rating <= 3:
                negative_reviews.append(review)

        except:
            pass

    with open(
        negative_filename,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            f"{app_name} - {store_name}\n"
        )

        file.write(
            "NEGATIVE / NEUTRAL REVIEWS "
            "(1-3 STARS)\n"
        )

        file.write(
            f"Total: {len(negative_reviews)}\n\n"
        )

        for review in negative_reviews:

            write_review(file, review)

    print(f"Saved: {filename}")

    print(
        f"Saved negative reviews: "
        f"{negative_filename}"
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    all_reviews_combined = []

    print()
    print("#" * 70)
    print("COMPETITOR APP REVIEW DOWNLOADER")
    print("#" * 70)

    for app in APPS:

        app_name = app["name"]

        # ----------------------------------------------------
        # GOOGLE
        # ----------------------------------------------------

        google_reviews = download_google_reviews(
            app_name,
            app["google_id"]
        )

        save_reviews(
            app_name,
            "Google Play",
            google_reviews
        )

        all_reviews_combined.extend(
            google_reviews
        )

        # ----------------------------------------------------
        # APPLE
        # ----------------------------------------------------

        apple_reviews = download_apple_reviews(
            app_name,
            app["apple_id"]
        )

        save_reviews(
            app_name,
            "Apple App Store",
            apple_reviews
        )

        all_reviews_combined.extend(
            apple_reviews
        )

    # ========================================================
    # COMBINED FILE
    # ========================================================

    combined_filename = os.path.join(
        OUTPUT_FOLDER,
        "ALL_COMPETITOR_REVIEWS.txt"
    )

    with open(
        combined_filename,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "SERVERSE COMPETITOR APP REVIEWS\n"
        )

        file.write(
            "=" * 80 + "\n"
        )

        file.write(
            f"Total reviews: "
            f"{len(all_reviews_combined)}\n\n"
        )

        for review in all_reviews_combined:

            write_review(file, review)

    # ========================================================
    # COMBINED NEGATIVE FILE
    # ========================================================

    negative_combined = []

    for review in all_reviews_combined:

        try:

            if int(review["rating"]) <= 3:
                negative_combined.append(review)

        except:
            pass

    negative_filename = os.path.join(
        OUTPUT_FOLDER,
        "ALL_NEGATIVE_REVIEWS_1_TO_3_STARS.txt"
    )

    with open(
        negative_filename,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "SERVERSE COMPETITOR NEGATIVE REVIEWS\n"
        )

        file.write(
            "1 TO 3 STAR REVIEWS\n"
        )

        file.write(
            "=" * 80 + "\n"
        )

        file.write(
            f"Total negative reviews: "
            f"{len(negative_combined)}\n\n"
        )

        for review in negative_combined:

            write_review(file, review)

    print()
    print("#" * 70)
    print("FINISHED")
    print("#" * 70)

    print(
        f"Total reviews collected: "
        f"{len(all_reviews_combined)}"
    )

    print(
        f"1-3 star reviews: "
        f"{len(negative_combined)}"
    )

    print()
    print(
        f"Files are inside the folder: "
        f"{OUTPUT_FOLDER}"
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()
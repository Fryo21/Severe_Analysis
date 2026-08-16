import os
import time
import hashlib

import pandas as pd

from llm_sentiment import classify_sentiment
from llm_category import categorize_review


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "analysis/negative_reviews_3y.csv"

OUTPUT_FOLDER = "analysis"

OUTPUT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "sentiment_reviews_3y.csv"
)

ERROR_FILE = os.path.join(
    OUTPUT_FOLDER,
    "sentiment_pipeline_errors.csv"
)

MAX_REVIEWS = None

REQUEST_DELAY = 0.2

MAX_RETRIES = 3


# ============================================================
# CREATE UNIQUE REVIEW KEY
# ============================================================

def create_review_key(row):

    raw = "|".join([
        str(row.get("competitor", "")),
        str(row.get("store", "")),
        str(row.get("review_date", "")),
        str(row.get("author", "")),
        str(row.get("review", ""))
    ])

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


# ============================================================
# SAVE PIPELINE
# ============================================================

def save_dataframe(df):

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# SAVE ERROR
# ============================================================

def save_error(
    row,
    stage,
    error
):

    error_row = pd.DataFrame([
        {
            "review_key":
                row.get("review_key", ""),

            "competitor":
                row.get("competitor", ""),

            "review":
                row.get("review", ""),

            "stage":
                stage,

            "error":
                str(error)
        }
    ])

    file_exists = os.path.exists(
        ERROR_FILE
    )

    error_row.to_csv(
        ERROR_FILE,
        mode="a",
        header=not file_exists,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    # ========================================================
    # EXISTING PIPELINE OUTPUT
    # ========================================================

    if os.path.exists(
        OUTPUT_FILE
    ):

        print()
        print(
            "Existing pipeline output found."
        )

        print(
            f"Loading: {OUTPUT_FILE}"
        )

        df = pd.read_csv(
            OUTPUT_FILE
        )

        return df

    # ========================================================
    # NO EXISTING OUTPUT
    # ========================================================

    print()
    print(
        "No existing sentiment output found."
    )

    print(
        "Loading original review dataset."
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    # --------------------------------------------------------
    # CLEAN REVIEW
    # --------------------------------------------------------

    if "review" not in df.columns:

        raise ValueError(
            "Input file does not contain "
            "a 'review' column."
        )

    df["review"] = (
        df["review"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[
        df["review"] != ""
    ].copy()

    # --------------------------------------------------------
    # CREATE KEYS
    # --------------------------------------------------------

    if "review_key" not in df.columns:

        df["review_key"] = df.apply(
            create_review_key,
            axis=1
        )

    return df


# ============================================================
# SENTIMENT STAGE
# ============================================================

def run_sentiment_stage(df):

    print()
    print("=" * 70)
    print("STAGE 1 - SENTIMENT")
    print("=" * 70)

    # --------------------------------------------------------
    # CREATE COLUMN IF MISSING
    # --------------------------------------------------------

    if "sentiment" not in df.columns:

        df["sentiment"] = pd.NA

    # --------------------------------------------------------
    # FIND REVIEWS WITHOUT SENTIMENT
    # --------------------------------------------------------

    missing_sentiment = (
        df["sentiment"].isna()
        |
        (
            df["sentiment"]
            .astype(str)
            .str.strip()
            == ""
        )
    )

    remaining = df[
        missing_sentiment
    ]

    if MAX_REVIEWS is not None:

        remaining = remaining.head(
            MAX_REVIEWS
        )

    total = len(
        remaining
    )

    # ========================================================
    # SENTIMENT ALREADY COMPLETE
    # ========================================================

    if total == 0:

        print()
        print(
            "Sentiment already completed."
        )

        print(
            "Skipping sentiment LLM."
        )

        return df

    print()
    print(
        f"Reviews requiring sentiment: {total}"
    )

    # ========================================================
    # PROCESS SENTIMENT
    # ========================================================

    for number, (index, row) in enumerate(
        remaining.iterrows(),
        start=1
    ):

        review = str(
            row["review"]
        )

        competitor = row.get(
            "competitor",
            "Unknown"
        )

        print(
            f"[{number}/{total}] "
            f"{competitor}"
        )

        success = False
        last_error = None

        # ----------------------------------------------------
        # RETRIES
        # ----------------------------------------------------

        for attempt in range(
            1,
            MAX_RETRIES + 1
        ):

            try:

                sentiment = (
                    classify_sentiment(
                        review
                    )
                )

                df.at[
                    index,
                    "sentiment"
                ] = sentiment

                success = True

                print(
                    f"    Sentiment: "
                    f"{sentiment}"
                )

                break

            except Exception as error:

                last_error = error

                print(
                    f"    Attempt "
                    f"{attempt}/"
                    f"{MAX_RETRIES} "
                    f"failed: {error}"
                )

                if attempt < MAX_RETRIES:

                    time.sleep(
                        attempt * 2
                    )

        # ----------------------------------------------------
        # SAVE FAILURE
        # ----------------------------------------------------

        if not success:

            save_error(
                row,
                "sentiment",
                last_error
            )

            print(
                "    Sentiment failed"
            )

        # ----------------------------------------------------
        # SAVE PROGRESS
        # ----------------------------------------------------

        save_dataframe(
            df
        )

        time.sleep(
            REQUEST_DELAY
        )

    return df


# ============================================================
# RESTORE CATEGORY LIST
# ============================================================

def load_existing_categories(df):

    categories = []

    if (
        "category_id" not in df.columns
        or
        "category_name" not in df.columns
    ):

        return categories

    existing = df[
        df["category_id"].notna()
        &
        df["category_name"].notna()
    ][
        [
            "category_id",
            "category_name"
        ]
    ].drop_duplicates()

    for _, row in existing.iterrows():

        try:

            category_id = int(
                float(
                    row["category_id"]
                )
            )

        except (ValueError, TypeError):

            continue

        category_name = str(
            row["category_name"]
        ).strip()

        if not category_name:
            continue

        categories.append(
            {
                "category_id":
                    category_id,

                "category_name":
                    category_name
            }
        )

    # --------------------------------------------------------
    # REMOVE DUPLICATE IDS
    # --------------------------------------------------------

    unique = {}

    for category in categories:

        unique[
            category["category_id"]
        ] = category

    categories = list(
        unique.values()
    )

    categories.sort(
        key=lambda item:
        item["category_id"]
    )

    return categories


# ============================================================
# CATEGORY STAGE
# ============================================================

def run_category_stage(df):

    print()
    print("=" * 70)
    print("STAGE 2 - CATEGORY")
    print("=" * 70)

    # --------------------------------------------------------
    # CREATE CATEGORY COLUMNS
    # --------------------------------------------------------

    if "category_id" not in df.columns:

        df["category_id"] = pd.NA

    if "category_name" not in df.columns:

        df["category_name"] = pd.NA

    # ========================================================
    # RESTORE EXISTING CATEGORY LIST
    # ========================================================

    categories = load_existing_categories(
        df
    )

    print()
    print(
        f"Existing categories: "
        f"{len(categories)}"
    )

    if categories:

        for category in categories:

            print(
                f"    "
                f"{category['category_id']} - "
                f"{category['category_name']}"
            )

    # ========================================================
    # FIND UNCATEGORISED REVIEWS
    # ========================================================

    missing_category = (
        df["category_id"].isna()
        |
        df["category_name"].isna()
        |
        (
            df["category_name"]
            .astype(str)
            .str.strip()
            == ""
        )
    )

    # Only categorize reviews that successfully
    # received sentiment.
    valid_sentiment = (
        df["sentiment"].notna()
        &
        (
            df["sentiment"]
            .astype(str)
            .str.strip()
            != ""
        )
    )

    remaining = df[
        missing_category
        &
        valid_sentiment
    ]

    if MAX_REVIEWS is not None:

        remaining = remaining.head(
            MAX_REVIEWS
        )

    total = len(
        remaining
    )

    # ========================================================
    # CATEGORY ALREADY COMPLETE
    # ========================================================

    if total == 0:

        print()
        print(
            "All reviews already categorized."
        )

        return df

    print()
    print(
        f"Reviews requiring categories: "
        f"{total}"
    )

    print()

    # ========================================================
    # PROCESS CATEGORY
    # ========================================================

    for number, (index, row) in enumerate(
        remaining.iterrows(),
        start=1
    ):

        review = str(
            row["review"]
        )

        competitor = row.get(
            "competitor",
            "Unknown"
        )

        print(
            f"[{number}/{total}] "
            f"{competitor}"
        )

        success = False
        last_error = None

        # ----------------------------------------------------
        # RETRIES
        # ----------------------------------------------------

        for attempt in range(
            1,
            MAX_RETRIES + 1
        ):

            try:

                result = (
                    categorize_review(
                        review,
                        categories
                    )
                )

                # --------------------------------------------
                # UPDATE MASTER CATEGORY LIST
                # --------------------------------------------

                categories = result[
                    "categories"
                ]

                # --------------------------------------------
                # SAVE CATEGORY RESULT
                # --------------------------------------------

                df.at[
                    index,
                    "category_id"
                ] = result[
                    "category_id"
                ]

                df.at[
                    index,
                    "category_name"
                ] = result[
                    "category_name"
                ]

                success = True

                print(
                    f"    Category: "
                    f"{result['category_id']} - "
                    f"{result['category_name']}"
                )

                if result[
                    "created_new_category"
                ]:

                    print(
                        "    New category created"
                    )

                break

            except Exception as error:

                last_error = error

                print(
                    f"    Attempt "
                    f"{attempt}/"
                    f"{MAX_RETRIES} "
                    f"failed: {error}"
                )

                if attempt < MAX_RETRIES:

                    time.sleep(
                        attempt * 2
                    )

        # ----------------------------------------------------
        # SAVE FAILURE
        # ----------------------------------------------------

        if not success:

            save_error(
                row,
                "category",
                last_error
            )

            print(
                "    Categorization failed"
            )

        # ----------------------------------------------------
        # SAVE AFTER EVERY REVIEW
        # ----------------------------------------------------

        save_dataframe(
            df
        )

        time.sleep(
            REQUEST_DELAY
        )

    return df


# ============================================================
# DISPLAY FINAL SUMMARY
# ============================================================

def display_summary(df):

    print()
    print("=" * 70)
    print("FINAL PIPELINE SUMMARY")
    print("=" * 70)

    print(
        f"Total reviews: "
        f"{len(df)}"
    )

    if "sentiment" in df.columns:

        print()
        print("SENTIMENT")
        print("-" * 40)

        print(
            df["sentiment"]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    if "category_id" in df.columns:

        categories = (
            df[
                [
                    "category_id",
                    "category_name"
                ]
            ]
            .dropna()
            .drop_duplicates()
            .sort_values(
                "category_id"
            )
        )

        print()
        print(
            f"Total categories: "
            f"{len(categories)}"
        )

        print()
        print("CATEGORY LIST")
        print("-" * 40)

        for _, row in categories.iterrows():

            print(
                f"{int(float(row['category_id']))}: "
                f"{row['category_name']}"
            )


# ============================================================
# FULL PIPELINE
# ============================================================

def run_pipeline():

    print()
    print("=" * 70)
    print(
        "SERVERSE - SENTIMENT & CATEGORY PIPELINE"
    )
    print("=" * 70)

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    # --------------------------------------------------------
    # CHECK API KEY
    # --------------------------------------------------------

    if not os.getenv(
        "OPENAI_API_KEY"
    ):

        raise EnvironmentError(
            "OPENAI_API_KEY is not configured."
        )

    # ========================================================
    # LOAD DATA
    # ========================================================

    df = load_data()

    print()
    print(
        f"Reviews loaded: {len(df)}"
    )

    # ========================================================
    # STAGE 1
    # ========================================================

    df = run_sentiment_stage(
        df
    )

    # ========================================================
    # STAGE 2
    # ========================================================

    df = run_category_stage(
        df
    )

    # ========================================================
    # FINAL SAVE
    # ========================================================

    save_dataframe(
        df
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    display_summary(
        df
    )

    print()
    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    print(
        f"Final output: "
        f"{OUTPUT_FILE}"
    )

    return df


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    run_pipeline()
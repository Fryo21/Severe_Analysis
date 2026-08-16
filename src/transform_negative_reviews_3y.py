import os
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "transformed_reviews/reviews_last_3_years.csv"

OUTPUT_FOLDER = "analysis"

OUTPUT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "negative_reviews_3y.csv"
)


# ============================================================
# TRANSFORM NEGATIVE REVIEWS
# ============================================================

def transform_negative_reviews():

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    print("=" * 70)
    print("SERVERSE - NEGATIVE REVIEW TRANSFORMATION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load reviews from the last 3 years
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE
    )

    print(f"\nTotal reviews loaded: {len(df)}")

    # --------------------------------------------------------
    # 2. Make sure rating is numeric
    # --------------------------------------------------------

    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # 3. Remove rows without a valid rating
    # --------------------------------------------------------

    df = df.dropna(
        subset=["rating"]
    )

    # --------------------------------------------------------
    # 4. Keep only 1-3 star reviews
    # --------------------------------------------------------

    negative_df = df[
        df["rating"].between(
            1,
            3,
            inclusive="both"
        )
    ].copy()

    # --------------------------------------------------------
    # 5. Remove reviews with no text
    # --------------------------------------------------------

    if "review" in negative_df.columns:

        negative_df["review"] = (
            negative_df["review"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        negative_df = negative_df[
            negative_df["review"] != ""
        ]

    # --------------------------------------------------------
    # 6. Sort reviews
    # --------------------------------------------------------

    if "review_date" in negative_df.columns:

        negative_df["review_date"] = pd.to_datetime(
            negative_df["review_date"],
            errors="coerce"
        )

        negative_df = negative_df.sort_values(
            by=["competitor", "review_date"],
            ascending=[True, False]
        )

    # --------------------------------------------------------
    # 7. Save output
    # --------------------------------------------------------

    negative_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print(
        f"Negative reviews (1-3 stars): "
        f"{len(negative_df)}"
    )

    # --------------------------------------------------------
    # Reviews per competitor
    # --------------------------------------------------------

    if "competitor" in negative_df.columns:

        print()
        print("NEGATIVE REVIEWS PER COMPETITOR")
        print("-" * 40)

        competitor_stats = (
            negative_df
            .groupby("competitor")
            .size()
            .sort_values(ascending=False)
        )

        print(
            competitor_stats.to_string()
        )

    # --------------------------------------------------------
    # Reviews by rating
    # --------------------------------------------------------

    print()
    print("REVIEWS BY STAR RATING")
    print("-" * 40)

    rating_stats = (
        negative_df
        .groupby("rating")
        .size()
        .sort_index()
    )

    print(
        rating_stats.to_string()
    )

    print()
    print("=" * 70)
    print("FILE CREATED")
    print("=" * 70)

    print(OUTPUT_FILE)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    transform_negative_reviews()
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = (
    PROJECT_ROOT
    / "raw-data"
    / "competitior-research"
    / "trustpilot"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed-data"
    / "competitor-research"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ANALYSIS PERIOD
# ============================================================

START_DATE = pd.Timestamp("2024-01-01", tz="UTC")
END_DATE = pd.Timestamp("2026-08-14 23:59:59", tz="UTC")


# ============================================================
# IDENTIFY COMPETITOR
# ============================================================

def identify_competitor(filename):
    filename = filename.lower()

    if "checkatrade" in filename:
        return "Checkatrade"

    if "bark" in filename:
        return "Bark"

    if "yell" in filename:
        return "Yell"

    if "yelp" in filename:
        return "Yelp"

    return "Unknown"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    csv_files = list(RAW_DATA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in:\n{RAW_DATA_DIR}"
        )

    datasets = []

    for file_path in csv_files:
        competitor = identify_competitor(file_path.name)

        print(f"Loading {competitor}...")

        df = pd.read_csv(
            file_path,
            low_memory=False
        )

        required_columns = [
            "reviewId",
            "rating",
            "publishedDate",
            "text"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            print(
                f"Skipping {competitor}. "
                f"Missing columns: {missing_columns}"
            )
            continue

        df = df[required_columns].copy()

        df["competitor"] = competitor

        datasets.append(df)

    if not datasets:
        raise ValueError(
            "No valid competitor datasets were loaded."
        )

    combined_df = pd.concat(
        datasets,
        ignore_index=True
    )

    return combined_df


# ============================================================
# CLEAN DATA
# ============================================================

def clean_data(df):
    df["publishedDate"] = pd.to_datetime(
        df["publishedDate"],
        errors="coerce",
        utc=True
    )

    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "reviewId",
            "rating",
            "publishedDate",
            "text"
        ]
    )

    df = df.drop_duplicates(
        subset=["reviewId"]
    )

    return df


# ============================================================
# FILTER 2024 - 2026
# ============================================================

def filter_analysis_period(df):
    filtered_df = df[
        (df["publishedDate"] >= START_DATE)
        &
        (df["publishedDate"] <= END_DATE)
    ].copy()

    filtered_df["year"] = (
        filtered_df["publishedDate"].dt.year
    )

    return filtered_df


# ============================================================
# REVIEW COVERAGE
# ============================================================

def analyse_coverage(df):
    print("\n" + "=" * 60)
    print("2024 - 2026 REVIEW COVERAGE")
    print("=" * 60)

    total_counts = (
        df.groupby("competitor")
        .size()
        .sort_values(ascending=False)
    )

    print("\nTOTAL REVIEWS:")
    print(total_counts)

    yearly_counts = (
        df.groupby(
            ["competitor", "year"]
        )
        .size()
        .unstack(fill_value=0)
    )

    print("\nREVIEWS BY YEAR:")
    print(yearly_counts)

    rating_counts = (
        df.groupby(
            ["competitor", "rating"]
        )
        .size()
        .unstack(fill_value=0)
    )

    print("\nREVIEWS BY RATING:")
    print(rating_counts)

    return yearly_counts, rating_counts


# ============================================================
# NEGATIVE REVIEW ANALYSIS
# ============================================================

def analyse_negative_reviews(df):
    negative_df = df[
        df["rating"].isin([1, 2, 3])
    ].copy()

    print("\n" + "=" * 60)
    print("NEGATIVE REVIEWS: 1★–3★")
    print("=" * 60)

    negative_counts = (
        negative_df
        .groupby(
            ["competitor", "year", "rating"]
        )
        .size()
        .unstack(fill_value=0)
    )

    print("\nNEGATIVE REVIEWS BY YEAR AND RATING:")
    print(negative_counts)

    totals = (
        negative_df
        .groupby("competitor")
        .size()
        .sort_values(ascending=False)
    )

    print("\nTOTAL 1★–3★ REVIEWS:")
    print(totals)

    output_file = (
        OUTPUT_DIR
        / "negative_review_counts_2024_2026.csv"
    )

    negative_counts.to_csv(
        output_file
    )

    negative_reviews_file = (
        OUTPUT_DIR
        / "negative_reviews_2024_2026.csv"
    )

    negative_df.to_csv(
        negative_reviews_file,
        index=False
    )

    print(
        f"\nNegative review counts saved to:\n{output_file}"
    )

    print(
        f"\nNegative reviews saved to:\n{negative_reviews_file}"
    )

    return negative_df, negative_counts


# ============================================================
# CREATE COVERAGE GRAPH
# ============================================================

def create_coverage_graph(yearly_counts):
    graph_file = (
        OUTPUT_DIR
        / "trustpilot_review_coverage_2024_2026.png"
    )

    yearly_counts = yearly_counts.copy()

    yearly_counts.columns = [
        "2024" if year == 2024
        else "2025" if year == 2025
        else "2026 YTD"
        for year in yearly_counts.columns
    ]

    ax = yearly_counts.plot(
        kind="bar",
        figsize=(10, 6)
    )

    ax.set_title(
        "Trustpilot Review Sample Coverage: 2024–2026"
    )

    ax.set_xlabel("Competitor")

    ax.set_ylabel(
        "Number of collected reviews"
    )

    ax.legend(
        title="Year"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        graph_file,
        dpi=300
    )

    plt.close()

    print(
        f"\nCoverage graph saved to:\n{graph_file}"
    )


# ============================================================
# CREATE NEGATIVE REVIEW GRAPH
# ============================================================

def create_negative_review_graph(negative_df):
    negative_by_year = (
        negative_df
        .groupby(["competitor", "year"])
        .size()
        .unstack(fill_value=0)
    )

    negative_by_year.columns = [
        "2024" if year == 2024
        else "2025" if year == 2025
        else "2026 YTD"
        for year in negative_by_year.columns
    ]

    graph_file = (
        OUTPUT_DIR
        / "negative_review_coverage_2024_2026.png"
    )

    ax = negative_by_year.plot(
        kind="bar",
        figsize=(10, 6)
    )

    ax.set_title(
        "Collected 1★–3★ Trustpilot Reviews: 2024–2026"
    )

    ax.set_xlabel("Competitor")

    ax.set_ylabel(
        "Number of collected negative reviews"
    )

    ax.legend(
        title="Year"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        graph_file,
        dpi=300
    )

    plt.close()

    print(
        f"\nNegative review graph saved to:\n{graph_file}"
    )


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(
    df,
    yearly_counts,
    rating_counts
):
    filtered_file = (
        OUTPUT_DIR
        / "trustpilot_reviews_2024_2026.csv"
    )

    yearly_file = (
        OUTPUT_DIR
        / "review_counts_by_year.csv"
    )

    rating_file = (
        OUTPUT_DIR
        / "review_counts_by_rating.csv"
    )

    df.to_csv(
        filtered_file,
        index=False
    )

    yearly_counts.to_csv(
        yearly_file
    )

    rating_counts.to_csv(
        rating_file
    )

    print("\nFILES SAVED:")
    print(filtered_file)
    print(yearly_file)
    print(rating_file)


# ============================================================
# MAIN
# ============================================================

def main():
    print("\nLoading datasets...")

    df = load_data()

    print(
        f"\nTotal raw reviews loaded: "
        f"{len(df):,}"
    )

    df = clean_data(df)

    print(
        f"Reviews after cleaning: "
        f"{len(df):,}"
    )

    filtered_df = filter_analysis_period(df)

    print(
        f"Reviews within 2024-2026: "
        f"{len(filtered_df):,}"
    )

    yearly_counts, rating_counts = (
        analyse_coverage(filtered_df)
    )

    negative_df, negative_counts = (
        analyse_negative_reviews(filtered_df)
    )

    save_outputs(
        filtered_df,
        yearly_counts,
        rating_counts
    )

    create_coverage_graph(
        yearly_counts
    )

    create_negative_review_graph(
        negative_df
    )

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
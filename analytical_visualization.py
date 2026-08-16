import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "analysis/sentiment_reviews_3y.csv"

OUTPUT_FOLDER = "analysis/visualizations"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Analysis file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "review",
        "sentiment",
        "category_id",
        "category_name"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df


# ============================================================
# DISPLAY GENERAL STATISTICS
# ============================================================

def display_summary(df):

    print()
    print("=" * 70)
    print("SERVERSE - REVIEW ANALYSIS")
    print("=" * 70)

    print(
        f"\nTotal reviews analysed: {len(df)}"
    )

    print(
        f"Total categories discovered: "
        f"{df['category_id'].nunique()}"
    )

    if "competitor" in df.columns:

        print(
            f"Total competitors: "
            f"{df['competitor'].nunique()}"
        )


# ============================================================
# SENTIMENT STATISTICS
# ============================================================

def sentiment_statistics(df):

    print()
    print("SENTIMENT DISTRIBUTION")
    print("-" * 40)

    stats = (
        df["sentiment"]
        .value_counts()
    )

    print(stats.to_string())

    percentages = (
        df["sentiment"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print()
    print("PERCENTAGE")

    for sentiment, percentage in percentages.items():

        print(
            f"{sentiment}: {percentage}%"
        )


# ============================================================
# CATEGORY STATISTICS
# ============================================================

def category_statistics(df):

    print()
    print("CATEGORY DISTRIBUTION")
    print("-" * 40)

    stats = (
        df["category_name"]
        .value_counts()
    )

    print(
        stats.to_string()
    )


# ============================================================
# COMPETITOR STATISTICS
# ============================================================

def competitor_statistics(df):

    if "competitor" not in df.columns:
        return

    print()
    print("REVIEWS PER COMPETITOR")
    print("-" * 40)

    stats = (
        df["competitor"]
        .value_counts()
    )

    print(
        stats.to_string()
    )


# ============================================================
# SENTIMENT CHART
# ============================================================

def plot_sentiment(df):

    stats = (
        df["sentiment"]
        .value_counts()
    )

    plt.figure(
        figsize=(7, 5)
    )

    stats.plot(
        kind="bar"
    )

    plt.title(
        "Review Sentiment Distribution"
    )

    plt.xlabel(
        "Sentiment"
    )

    plt.ylabel(
        "Number of Reviews"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_FOLDER,
        "sentiment_distribution.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.show()

    plt.close()


# ============================================================
# CATEGORY CHART
# ============================================================

def plot_categories(df):

    stats = (
        df["category_name"]
        .value_counts()
        .sort_values()
    )

    plt.figure(
        figsize=(10, max(6, len(stats) * 0.4))
    )

    stats.plot(
        kind="barh"
    )

    plt.title(
        "Reviews by Category"
    )

    plt.xlabel(
        "Number of Reviews"
    )

    plt.ylabel(
        "Category"
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_FOLDER,
        "category_distribution.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.show()

    plt.close()


# ============================================================
# CATEGORY × SENTIMENT
# ============================================================

def plot_category_sentiment(df):

    table = pd.crosstab(
        df["category_name"],
        df["sentiment"]
    )

    table = table.loc[
        table.sum(axis=1)
        .sort_values()
        .index
    ]

    table.plot(
        kind="barh",
        stacked=True,
        figsize=(
            11,
            max(6, len(table) * 0.4)
        )
    )

    plt.title(
        "Sentiment Distribution by Category"
    )

    plt.xlabel(
        "Number of Reviews"
    )

    plt.ylabel(
        "Category"
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_FOLDER,
        "category_sentiment.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.show()

    plt.close()


# ============================================================
# COMPETITOR × CATEGORY
# ============================================================

def plot_competitor_categories(df):

    if "competitor" not in df.columns:
        return

    table = pd.crosstab(
        df["competitor"],
        df["category_name"]
    )

    table.plot(
        kind="bar",
        stacked=True,
        figsize=(12, 7)
    )

    plt.title(
        "Category Distribution by Competitor"
    )

    plt.xlabel(
        "Competitor"
    )

    plt.ylabel(
        "Number of Reviews"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.legend(
        title="Category",
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_FOLDER,
        "competitor_categories.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


# ============================================================
# EXPORT SUMMARY TABLES
# ============================================================

def export_statistics(df):

    sentiment_stats = (
        df["sentiment"]
        .value_counts()
        .rename_axis("sentiment")
        .reset_index(name="review_count")
    )

    sentiment_stats.to_csv(
        os.path.join(
            OUTPUT_FOLDER,
            "sentiment_statistics.csv"
        ),
        index=False
    )

    category_stats = (
        df["category_name"]
        .value_counts()
        .rename_axis("category_name")
        .reset_index(name="review_count")
    )

    category_stats.to_csv(
        os.path.join(
            OUTPUT_FOLDER,
            "category_statistics.csv"
        ),
        index=False
    )

    if "competitor" in df.columns:

        competitor_category = pd.crosstab(
            df["competitor"],
            df["category_name"]
        )

        competitor_category.to_csv(
            os.path.join(
                OUTPUT_FOLDER,
                "competitor_category_matrix.csv"
            )
        )


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    df = load_data()

    display_summary(df)

    sentiment_statistics(df)

    category_statistics(df)

    competitor_statistics(df)

    export_statistics(df)

    plot_sentiment(df)

    plot_categories(df)

    plot_category_sentiment(df)

    plot_competitor_categories(df)

    print()
    print("=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)

    print(
        f"Files saved to: {OUTPUT_FOLDER}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
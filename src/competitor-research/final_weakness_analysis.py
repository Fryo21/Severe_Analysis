from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "processed-data"
    / "competitor-research"
    / "complaint-analysis-final"
    / "training_sample_ai_labeled_v3.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed-data"
    / "competitor-research"
    / "final-weakness-analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

COMPETITORS = [
    "Bark",
    "Checkatrade",
    "Yell",
    "Yelp",
]

VALID_SIDES = [
    "Customer",
    "Service Provider / Business",
]

SEVERITY_WEIGHT = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
}

# Side-level percentages become unreliable with very tiny samples.
# We will still save them, but flag them.
MIN_SIDE_SAMPLE = 10


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"\nCould not find V3 labelled dataset:\n"
            f"{INPUT_FILE}\n\n"
            f"Save training_sample_ai_labeled_v3.csv "
            f"to that location first."
        )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    required = {
        "reviewId",
        "competitor",
        "text",
        "manual_relevance",
        "manual_user_type",
        "primary_complaint_theme",
        "secondary_complaint_theme",
        "manual_severity",
    }

    missing = required - set(df.columns)

    if missing:

        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(sorted(missing))
        )

    df["reviewId"] = (
        df["reviewId"]
        .astype(str)
    )

    return df


# ============================================================
# PREPARE RELEVANT REVIEWS
# ============================================================

def prepare_data(df):

    relevant = (
        df[
            df["manual_relevance"]
            .astype(str)
            .str.strip()
            .eq("Relevant")
        ]
        .copy()
    )

    relevant["manual_severity"] = (
        relevant["manual_severity"]
        .astype(str)
        .str.strip()
    )

    relevant["severity_weight"] = (
        relevant["manual_severity"]
        .map(SEVERITY_WEIGHT)
    )

    return relevant


# ============================================================
# EXPLODE PRIMARY + SECONDARY THEMES
# ============================================================

def build_theme_mentions(relevant):

    rows = []

    for _, row in relevant.iterrows():

        primary = row.get(
            "primary_complaint_theme"
        )

        secondary = row.get(
            "secondary_complaint_theme"
        )

        themes = []

        if (
            pd.notna(primary)
            and str(primary).strip()
            and str(primary).strip()
            != "Not applicable"
        ):
            themes.append(
                (
                    str(primary).strip(),
                    "Primary"
                )
            )

        if (
            pd.notna(secondary)
            and str(secondary).strip()
            and str(secondary).strip()
            != "Not applicable"
        ):

            secondary = (
                str(secondary).strip()
            )

            # Do not count the same theme twice
            # within one review.
            if (
                not themes
                or secondary
                != themes[0][0]
            ):
                themes.append(
                    (
                        secondary,
                        "Secondary"
                    )
                )

        for theme, theme_level in themes:

            rows.append(
                {
                    "reviewId":
                        row["reviewId"],

                    "competitor":
                        row["competitor"],

                    "user_type":
                        row["manual_user_type"],

                    "theme":
                        theme,

                    "theme_level":
                        theme_level,

                    "severity":
                        row["manual_severity"],

                    "severity_weight":
                        row["severity_weight"],

                    "rating":
                        row.get("rating"),

                    "text":
                        row.get("text"),
                }
            )

    mentions = pd.DataFrame(
        rows
    )

    return mentions


# ============================================================
# SAMPLE COVERAGE
# ============================================================

def calculate_sample_coverage(
    original,
    relevant,
):

    original_counts = (
        original
        .groupby("competitor")[
            "reviewId"
        ]
        .nunique()
        .rename(
            "sampled_reviews"
        )
    )

    relevant_counts = (
        relevant
        .groupby("competitor")[
            "reviewId"
        ]
        .nunique()
        .rename(
            "relevant_reviews"
        )
    )

    excluded_counts = (
        original[
            original[
                "manual_relevance"
            ]
            .astype(str)
            .str.strip()
            .eq("Exclude")
        ]
        .groupby("competitor")[
            "reviewId"
        ]
        .nunique()
        .rename(
            "excluded_reviews"
        )
    )

    coverage = (
        pd.concat(
            [
                original_counts,
                relevant_counts,
                excluded_counts,
            ],
            axis=1,
        )
        .fillna(0)
        .reset_index()
    )

    for column in [
        "sampled_reviews",
        "relevant_reviews",
        "excluded_reviews",
    ]:

        coverage[column] = (
            coverage[column]
            .astype(int)
        )

    return coverage


# ============================================================
# USER-SIDE SAMPLE COUNTS
# ============================================================

def calculate_side_counts(
    relevant,
):

    counts = (
        relevant[
            relevant[
                "manual_user_type"
            ]
            .isin(VALID_SIDES)
        ]
        .groupby(
            [
                "competitor",
                "manual_user_type",
            ]
        )[
            "reviewId"
        ]
        .nunique()
        .reset_index(
            name="side_sample_size"
        )
    )

    counts[
        "sample_warning"
    ] = np.where(
        counts[
            "side_sample_size"
        ]
        < MIN_SIDE_SAMPLE,
        "LOW SAMPLE - interpret cautiously",
        "",
    )

    return counts


# ============================================================
# OVERALL COMPETITOR WEAKNESS FREQUENCY
# ============================================================

def calculate_overall_frequency(
    relevant,
    mentions,
):

    denominators = (
        relevant
        .groupby("competitor")[
            "reviewId"
        ]
        .nunique()
        .rename(
            "relevant_reviews"
        )
    )

    frequency = (
        mentions
        .groupby(
            [
                "competitor",
                "theme",
            ]
        )[
            "reviewId"
        ]
        .nunique()
        .reset_index(
            name="reviews_mentioning_theme"
        )
        .merge(
            denominators,
            on="competitor",
            how="left",
        )
    )

    frequency[
        "percent_of_relevant_sample"
    ] = (
        frequency[
            "reviews_mentioning_theme"
        ]
        /
        frequency[
            "relevant_reviews"
        ]
        * 100
    ).round(1)

    return (
        frequency
        .sort_values(
            [
                "competitor",
                "percent_of_relevant_sample",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )


# ============================================================
# CUSTOMER / PROVIDER WEAKNESS FREQUENCY
# ============================================================

def calculate_side_frequency(
    relevant,
    mentions,
):

    relevant_sides = (
        relevant[
            relevant[
                "manual_user_type"
            ]
            .isin(VALID_SIDES)
        ]
        .copy()
    )

    denominators = (
        relevant_sides
        .groupby(
            [
                "competitor",
                "manual_user_type",
            ]
        )[
            "reviewId"
        ]
        .nunique()
        .reset_index(
            name="side_sample_size"
        )
    )

    useful_mentions = (
        mentions[
            mentions[
                "user_type"
            ]
            .isin(VALID_SIDES)
        ]
        .copy()
    )

    frequency = (
        useful_mentions
        .groupby(
            [
                "competitor",
                "user_type",
                "theme",
            ]
        )[
            "reviewId"
        ]
        .nunique()
        .reset_index(
            name="reviews_mentioning_theme"
        )
    )

    frequency = (
        frequency.merge(
            denominators,
            left_on=[
                "competitor",
                "user_type",
            ],
            right_on=[
                "competitor",
                "manual_user_type",
            ],
            how="left",
        )
        .drop(
            columns=[
                "manual_user_type"
            ]
        )
    )

    frequency[
        "percent_of_side_sample"
    ] = (
        frequency[
            "reviews_mentioning_theme"
        ]
        /
        frequency[
            "side_sample_size"
        ]
        * 100
    ).round(1)

    frequency[
        "sample_warning"
    ] = np.where(
        frequency[
            "side_sample_size"
        ]
        < MIN_SIDE_SAMPLE,
        "LOW SAMPLE - interpret cautiously",
        "",
    )

    return (
        frequency
        .sort_values(
            [
                "competitor",
                "user_type",
                "percent_of_side_sample",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )
    )


# ============================================================
# SEVERITY PROFILE
# ============================================================

def calculate_severity_profile(
    mentions,
):

    counts = (
        mentions
        .groupby(
            [
                "competitor",
                "user_type",
                "theme",
                "severity",
            ]
        )[
            "reviewId"
        ]
        .nunique()
        .reset_index(
            name="reviews"
        )
    )

    severity_matrix = (
        counts
        .pivot_table(
            index=[
                "competitor",
                "user_type",
                "theme",
            ],
            columns="severity",
            values="reviews",
            fill_value=0,
        )
        .reset_index()
    )

    for severity in [
        "Low",
        "Medium",
        "High",
    ]:

        if severity not in (
            severity_matrix.columns
        ):
            severity_matrix[
                severity
            ] = 0

    average_severity = (
        mentions
        .groupby(
            [
                "competitor",
                "user_type",
                "theme",
            ]
        )[
            "severity_weight"
        ]
        .mean()
        .reset_index(
            name="average_severity_score"
        )
    )

    result = (
        severity_matrix
        .merge(
            average_severity,
            on=[
                "competitor",
                "user_type",
                "theme",
            ],
            how="left",
        )
    )

    result[
        "average_severity_score"
    ] = (
        result[
            "average_severity_score"
        ]
        .round(2)
    )

    return result


# ============================================================
# PRIORITY WEAKNESSES
# ============================================================

def calculate_priority_weaknesses(
    side_frequency,
    severity_profile,
):

    priority = (
        side_frequency
        .merge(
            severity_profile,
            on=[
                "competitor",
                "user_type",
                "theme",
            ],
            how="left",
        )
    )

    # Evidence priority combines:
    # frequency within that side's sample
    # × actual observed impact.
    #
    # This is NOT yet a full product "opportunity score",
    # because we have not scored competitors' existing solutions.
    priority[
        "evidence_priority_score"
    ] = (
        priority[
            "percent_of_side_sample"
        ]
        *
        priority[
            "average_severity_score"
        ]
    ).round(1)

    priority[
        "high_severity_share"
    ] = (
        priority["High"]
        /
        priority[
            "reviews_mentioning_theme"
        ]
        * 100
    ).fillna(0).round(1)

    return (
        priority
        .sort_values(
            [
                "competitor",
                "user_type",
                "evidence_priority_score",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )
    )


# ============================================================
# CROSS-COMPETITOR OPPORTUNITIES
# ============================================================

def calculate_cross_competitor(
    priority,
):

    # Ignore tiny competitor-side samples when
    # building cross-market conclusions.
    reliable = (
        priority[
            priority[
                "side_sample_size"
            ]
            >= MIN_SIDE_SAMPLE
        ]
        .copy()
    )

    results = []

    for (
        user_type,
        theme
    ), group in reliable.groupby(
        [
            "user_type",
            "theme",
        ]
    ):

        competitors_with_theme = (
            group[
                "competitor"
            ]
            .nunique()
        )

        macro_prevalence = (
            group[
                "percent_of_side_sample"
            ]
            .mean()
        )

        average_severity = (
            group[
                "average_severity_score"
            ]
            .mean()
        )

        cross_score = (
            macro_prevalence
            * average_severity
        )

        results.append(
            {
                "user_type":
                    user_type,

                "theme":
                    theme,

                "competitors_with_evidence":
                    competitors_with_theme,

                "macro_average_prevalence_percent":
                    round(
                        macro_prevalence,
                        1,
                    ),

                "average_severity_score":
                    round(
                        average_severity,
                        2,
                    ),

                "cross_competitor_evidence_score":
                    round(
                        cross_score,
                        1,
                    ),
            }
        )

    result = pd.DataFrame(
        results
    )

    if result.empty:
        return result

    return (
        result.sort_values(
            [
                "user_type",
                "cross_competitor_evidence_score",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )


# ============================================================
# MATRICES
# ============================================================

def build_overall_matrix(
    overall_frequency,
):

    matrix = (
        overall_frequency
        .pivot_table(
            index="theme",
            columns="competitor",
            values="percent_of_relevant_sample",
            fill_value=0,
        )
    )

    for competitor in COMPETITORS:

        if competitor not in matrix.columns:
            matrix[
                competitor
            ] = 0

    matrix = (
        matrix[
            COMPETITORS
        ]
        .reset_index()
    )

    return matrix


def build_side_matrix(
    side_frequency,
    side,
):

    data = (
        side_frequency[
            side_frequency[
                "user_type"
            ]
            .eq(side)
        ]
        .copy()
    )

    matrix = (
        data
        .pivot_table(
            index="theme",
            columns="competitor",
            values="percent_of_side_sample",
            fill_value=0,
        )
    )

    for competitor in COMPETITORS:

        if competitor not in matrix.columns:
            matrix[
                competitor
            ] = 0

    matrix = (
        matrix[
            COMPETITORS
        ]
        .reset_index()
    )

    return matrix


# ============================================================
# HEATMAP
# ============================================================

def create_heatmap(
    matrix_df,
    title,
    filename,
):

    if matrix_df.empty:
        return

    data = (
        matrix_df
        .set_index("theme")
    )

    values = (
        data.values
        .astype(float)
    )

    height = max(
        7,
        len(data) * 0.5,
    )

    fig, ax = plt.subplots(
        figsize=(
            11,
            height,
        )
    )

    image = ax.imshow(
        values,
        aspect="auto",
    )

    ax.set_xticks(
        range(
            len(data.columns)
        )
    )

    ax.set_xticklabels(
        data.columns
    )

    ax.set_yticks(
        range(
            len(data.index)
        )
    )

    ax.set_yticklabels(
        data.index
    )

    for row_index in range(
        values.shape[0]
    ):

        for col_index in range(
            values.shape[1]
        ):

            value = values[
                row_index,
                col_index,
            ]

            ax.text(
                col_index,
                row_index,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
            )

    ax.set_title(
        title
    )

    ax.set_xlabel(
        "Competitor"
    )

    ax.set_ylabel(
        "Weakness"
    )

    fig.colorbar(
        image,
        ax=ax,
        label=(
            "% of relevant side-specific "
            "review sample"
        ),
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# TOP PRIORITY CHARTS
# ============================================================

def create_priority_chart(
    cross_competitor,
    side,
    filename,
):

    subset = (
        cross_competitor[
            cross_competitor[
                "user_type"
            ]
            .eq(side)
        ]
        .head(10)
        .sort_values(
            "cross_competitor_evidence_score"
        )
    )

    if subset.empty:
        return

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.barh(
        subset["theme"],
        subset[
            "cross_competitor_evidence_score"
        ],
    )

    ax.set_title(
        f"Highest-Priority "
        f"{side} Weaknesses"
    )

    ax.set_xlabel(
        "Evidence priority score "
        "(frequency × severity)"
    )

    ax.set_ylabel(
        "Weakness"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    original,
    relevant,
    side_counts,
    priority,
    cross_competitor,
):

    print(
        "\n"
        + "=" * 76
    )

    print(
        "FINAL COMPETITOR WEAKNESS ANALYSIS"
    )

    print(
        "=" * 76
    )

    print(
        f"\nTotal sampled reviews: "
        f"{len(original):,}"
    )

    print(
        f"Relevant marketplace reviews: "
        f"{len(relevant):,}"
    )

    print(
        f"Excluded/off-topic reviews: "
        f"{len(original) - len(relevant):,}"
    )

    print(
        "\nRELEVANT REVIEWS BY COMPETITOR / SIDE"
    )

    print(
        "-" * 76
    )

    print(
        side_counts.to_string(
            index=False
        )
    )

    print(
        "\nTOP WEAKNESSES BY COMPETITOR / SIDE"
    )

    print(
        "-" * 76
    )

    for competitor in COMPETITORS:

        for side in VALID_SIDES:

            subset = (
                priority[
                    priority[
                        "competitor"
                    ].eq(
                        competitor
                    )
                    &
                    priority[
                        "user_type"
                    ].eq(
                        side
                    )
                ]
                .head(5)
            )

            if subset.empty:
                continue

            sample_size = (
                subset[
                    "side_sample_size"
                ]
                .iloc[0]
            )

            print(
                f"\n{competitor} "
                f"| {side} "
                f"| n={sample_size}"
            )

            print(
                subset[
                    [
                        "theme",
                        "reviews_mentioning_theme",
                        "percent_of_side_sample",
                        "average_severity_score",
                        "evidence_priority_score",
                    ]
                ]
                .to_string(
                    index=False
                )
            )

    print(
        "\nCROSS-COMPETITOR PRIORITIES"
    )

    print(
        "-" * 76
    )

    for side in VALID_SIDES:

        subset = (
            cross_competitor[
                cross_competitor[
                    "user_type"
                ].eq(side)
            ]
            .head(8)
        )

        if subset.empty:
            continue

        print(
            f"\n{side}"
        )

        print(
            subset[
                [
                    "theme",
                    "competitors_with_evidence",
                    "macro_average_prevalence_percent",
                    "average_severity_score",
                    "cross_competitor_evidence_score",
                ]
            ]
            .to_string(
                index=False
            )
        )


# ============================================================
# SAVE METHODOLOGY NOTE
# ============================================================

def save_methodology_note():

    note = """
FINAL WEAKNESS ANALYSIS - METHODOLOGY NOTE

Source:
400-review balanced competitor sample:
100 Bark, 100 Checkatrade, 100 Yell, 100 Yelp.

Annotations:
Uses the V3 AI-assisted reviewed labels:
- manual_relevance
- manual_user_type
- primary_complaint_theme
- secondary_complaint_theme
- manual_severity

Important interpretation rule:
Percentages are the share of RELEVANT REVIEWS IN THE ANALYSED SAMPLE
mentioning a weakness.

They are NOT estimates of the percentage of all users of a platform
who experience that problem.

For customer/provider comparisons, the denominator is the number of
relevant reviews from that marketplace side for that competitor.

Very small side-specific samples are flagged and should not be used
for strong comparative conclusions.

Evidence priority score:
percent of side-specific sample mentioning a weakness
MULTIPLIED BY
average severity score (Low=1, Medium=2, High=3).

This is an evidence-priority score, NOT yet a complete product
opportunity score. A full product opportunity score would additionally
consider the quality of competitors' existing solutions and our
ability to solve the problem.
""".strip()

    with open(
        OUTPUT_DIR
        / "METHODOLOGY.txt",
        "w",
        encoding="utf-8",
    ) as file:

        file.write(note)


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(
    coverage,
    side_counts,
    mentions,
    overall_frequency,
    side_frequency,
    severity_profile,
    priority,
    cross_competitor,
    overall_matrix,
    customer_matrix,
    provider_matrix,
):

    coverage.to_csv(
        OUTPUT_DIR
        / "sample_coverage.csv",
        index=False,
    )

    side_counts.to_csv(
        OUTPUT_DIR
        / "side_sample_sizes.csv",
        index=False,
    )

    mentions.to_csv(
        OUTPUT_DIR
        / "theme_mentions.csv",
        index=False,
    )

    overall_frequency.to_csv(
        OUTPUT_DIR
        / "competitor_weakness_frequency.csv",
        index=False,
    )

    side_frequency.to_csv(
        OUTPUT_DIR
        / "competitor_side_weakness_frequency.csv",
        index=False,
    )

    severity_profile.to_csv(
        OUTPUT_DIR
        / "weakness_severity_profile.csv",
        index=False,
    )

    priority.to_csv(
        OUTPUT_DIR
        / "priority_weaknesses.csv",
        index=False,
    )

    cross_competitor.to_csv(
        OUTPUT_DIR
        / "cross_competitor_opportunities.csv",
        index=False,
    )

    overall_matrix.to_csv(
        OUTPUT_DIR
        / "competitor_weakness_matrix.csv",
        index=False,
    )

    customer_matrix.to_csv(
        OUTPUT_DIR
        / "customer_weakness_matrix.csv",
        index=False,
    )

    provider_matrix.to_csv(
        OUTPUT_DIR
        / "provider_weakness_matrix.csv",
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\nStarting final weakness analysis..."
    )

    original = load_data()

    relevant = prepare_data(
        original
    )

    mentions = build_theme_mentions(
        relevant
    )

    coverage = (
        calculate_sample_coverage(
            original,
            relevant,
        )
    )

    side_counts = (
        calculate_side_counts(
            relevant
        )
    )

    overall_frequency = (
        calculate_overall_frequency(
            relevant,
            mentions,
        )
    )

    side_frequency = (
        calculate_side_frequency(
            relevant,
            mentions,
        )
    )

    severity_profile = (
        calculate_severity_profile(
            mentions
        )
    )

    priority = (
        calculate_priority_weaknesses(
            side_frequency,
            severity_profile,
        )
    )

    cross_competitor = (
        calculate_cross_competitor(
            priority
        )
    )

    overall_matrix = (
        build_overall_matrix(
            overall_frequency
        )
    )

    customer_matrix = (
        build_side_matrix(
            side_frequency,
            "Customer",
        )
    )

    provider_matrix = (
        build_side_matrix(
            side_frequency,
            "Service Provider / Business",
        )
    )

    save_outputs(
        coverage,
        side_counts,
        mentions,
        overall_frequency,
        side_frequency,
        severity_profile,
        priority,
        cross_competitor,
        overall_matrix,
        customer_matrix,
        provider_matrix,
    )

    save_methodology_note()

    create_heatmap(
        customer_matrix,
        (
            "Customer-Side Weaknesses "
            "by Competitor"
        ),
        "customer_weakness_heatmap.png",
    )

    create_heatmap(
        provider_matrix,
        (
            "Service Provider / Business "
            "Weaknesses by Competitor"
        ),
        "provider_weakness_heatmap.png",
    )

    create_priority_chart(
        cross_competitor,
        "Customer",
        "top_customer_opportunities.png",
    )

    create_priority_chart(
        cross_competitor,
        "Service Provider / Business",
        "top_provider_opportunities.png",
    )

    print_summary(
        original,
        relevant,
        side_counts,
        priority,
        cross_competitor,
    )

    print(
        "\n"
        + "=" * 76
    )

    print(
        "FILES SAVED"
    )

    print(
        "=" * 76
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nIMPORTANT:"
        "\nThese percentages describe the "
        "analysed review sample."
        "\nThey are NOT population-wide "
        "competitor complaint rates."
    )


if __name__ == "__main__":
    main()
    
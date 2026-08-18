from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLASSIFIED_FILE = (
    PROJECT_ROOT
    / "processed-data"
    / "competitor-research"
    / "complaint-analysis-final"
    / "classified_negative_reviews_final.csv"
)

GOLD_VALIDATION_FILE = (
    PROJECT_ROOT
    / "processed-data"
    / "competitor-research"
    / "complaint-analysis-final"
    / "gold_validation_set.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "processed-data"
    / "competitor-research"
    / "complaint-analysis-final"
    / "training_sample.csv"
)

COMPETITORS = [
    "Bark",
    "Checkatrade",
    "Yell",
    "Yelp",
]

SAMPLE_PER_COMPETITOR = 100
RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not CLASSIFIED_FILE.exists():
        raise FileNotFoundError(
            f"Could not find:\n{CLASSIFIED_FILE}"
        )

    if not GOLD_VALIDATION_FILE.exists():
        raise FileNotFoundError(
            f"Could not find:\n{GOLD_VALIDATION_FILE}"
        )

    reviews = pd.read_csv(
        CLASSIFIED_FILE,
        low_memory=False
    )

    gold = pd.read_csv(
        GOLD_VALIDATION_FILE,
        low_memory=False
    )

    print(f"Classified reviews loaded: {len(reviews):,}")
    print(f"Gold validation reviews: {len(gold):,}")

    return reviews, gold


# ============================================================
# REMOVE GOLD VALIDATION REVIEWS
# ============================================================

def remove_validation_reviews(reviews, gold):

    gold_ids = set(
        gold["reviewId"]
        .astype(str)
    )

    reviews["reviewId"] = (
        reviews["reviewId"]
        .astype(str)
    )

    remaining = reviews[
        ~reviews["reviewId"].isin(gold_ids)
    ].copy()

    removed = (
        len(reviews)
        - len(remaining)
    )

    print(
        f"Gold validation reviews removed: {removed:,}"
    )

    print(
        f"Reviews available for training sample: "
        f"{len(remaining):,}"
    )

    return remaining


# ============================================================
# SAMPLE ONE COMPETITOR
# ============================================================

def sample_competitor(df, competitor):

    subset = df[
        df["competitor"] == competitor
    ].copy()

    print(
        f"\n{competitor}: "
        f"{len(subset):,} available reviews"
    )

    if len(subset) <= SAMPLE_PER_COMPETITOR:

        return subset

    # --------------------------------------------------------
    # Group 1:
    # Reviews the rule-based classifier struggled with.
    #
    # These are especially useful because they contain
    # weaknesses our current rules missed.
    # --------------------------------------------------------

    uncertain = subset[
        (
            subset["user_type"] == "Unknown"
        )
        |
        (
            subset["needs_manual_review"] == True
        )
        |
        (
            subset["complaint_themes"]
            .fillna("")
            .str.contains(
                "Other / Manual Review",
                regex=False
            )
        )
    ].copy()

    # --------------------------------------------------------
    # Group 2:
    # Reviews where V3 found at least one complaint.
    #
    # Since V3 had relatively good precision when it DID
    # detect a complaint, these help ensure that known
    # complaint categories appear in the training data.
    # --------------------------------------------------------

    classified = subset[
        ~subset["reviewId"]
        .isin(
            uncertain["reviewId"]
        )
    ].copy()

    # Aim for 60 difficult + 40 already-classified reviews.

    uncertain_target = min(
        60,
        len(uncertain)
    )

    selected_uncertain = (
        uncertain.sample(
            n=uncertain_target,
            random_state=RANDOM_STATE
        )
        if uncertain_target > 0
        else uncertain
    )

    remaining_needed = (
        SAMPLE_PER_COMPETITOR
        - len(selected_uncertain)
    )

    classified_target = min(
        remaining_needed,
        len(classified)
    )

    selected_classified = (
        classified.sample(
            n=classified_target,
            random_state=RANDOM_STATE
        )
        if classified_target > 0
        else classified
    )

    sample = pd.concat(
        [
            selected_uncertain,
            selected_classified
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # If we still don't have 100 because one group was small,
    # fill from any unused reviews.
    # --------------------------------------------------------

    if len(sample) < SAMPLE_PER_COMPETITOR:

        used_ids = set(
            sample["reviewId"]
        )

        unused = subset[
            ~subset["reviewId"]
            .isin(used_ids)
        ]

        needed = (
            SAMPLE_PER_COMPETITOR
            - len(sample)
        )

        if len(unused) > 0:

            extra = unused.sample(
                n=min(
                    needed,
                    len(unused)
                ),
                random_state=RANDOM_STATE
            )

            sample = pd.concat(
                [
                    sample,
                    extra
                ],
                ignore_index=True
            )

    return sample


# ============================================================
# BUILD TRAINING SAMPLE
# ============================================================

def create_training_sample(df):

    samples = []

    for competitor in COMPETITORS:

        sample = sample_competitor(
            df,
            competitor
        )

        samples.append(sample)

    training = pd.concat(
        samples,
        ignore_index=True
    )

    # Shuffle the full dataset so reviews are not grouped
    # together when manually labelling.

    training = (
        training.sample(
            frac=1,
            random_state=RANDOM_STATE
        )
        .reset_index(drop=True)
    )

    return training


# ============================================================
# ADD MANUAL LABEL COLUMNS
# ============================================================

def prepare_for_manual_labelling(training):

    columns_to_keep = [
        "reviewId",
        "competitor",
        "rating",
        "publishedDate",
        "text",

        # Existing predictions are kept for comparison,
        # but they are NOT treated as ground truth.
        "relevance",
        "user_type",
        "user_type_confidence",
        "complaint_themes",
        "severity",
    ]

    existing = [
        col
        for col in columns_to_keep
        if col in training.columns
    ]

    training = training[
        existing
    ].copy()

    # --------------------------------------------------------
    # Human labels
    # --------------------------------------------------------

    training["manual_relevance"] = ""

    training["manual_user_type"] = ""

    training["manual_complaint_themes"] = ""

    training["manual_severity"] = ""

    training["manual_notes"] = ""

    return training


# ============================================================
# QUALITY CHECK
# ============================================================

def print_summary(training):

    print("\n" + "=" * 70)
    print("TRAINING SAMPLE CREATED")
    print("=" * 70)

    print(
        f"Total selected reviews: "
        f"{len(training):,}"
    )

    print("\nReviews per competitor:")

    print(
        training[
            "competitor"
        ]
        .value_counts()
        .to_string()
    )

    duplicated = (
        training[
            "reviewId"
        ]
        .duplicated()
        .sum()
    )

    print(
        f"\nDuplicate review IDs: "
        f"{duplicated}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\nCreating manually labelled "
        "training sample..."
    )

    reviews, gold = load_data()

    remaining = (
        remove_validation_reviews(
            reviews,
            gold
        )
    )

    training = (
        create_training_sample(
            remaining
        )
    )

    training = (
        prepare_for_manual_labelling(
            training
        )
    )

    training.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print_summary(
        training
    )

    print(
        f"\nSaved training sample to:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        "\nIMPORTANT:"
        "\nThis file is for MANUAL LABELLING."
        "\nDo not train a model until the manual_* "
        "columns have been completed."
    )


if __name__ == "__main__":
    main()
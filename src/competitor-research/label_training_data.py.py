from pathlib import Path
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
    / "training_sample.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "processed-data"
    / "competitor-research"
    / "complaint-analysis-final"
    / "training_sample_labeled.csv"
)


# ============================================================
# LABEL OPTIONS
# ============================================================

RELEVANCE_OPTIONS = {
    "1": "Relevant",
    "2": "Exclude",
    "3": "Unsure",
}

USER_TYPE_OPTIONS = {
    "1": "Customer",
    "2": "Service Provider / Business",
    "3": "Unknown",
}

THEME_OPTIONS = {
    "1": "Finding a provider / low response",
    "2": "Poor matching / location",
    "3": "Provider quality",
    "4": "Provider verification / safety",
    "5": "Too many provider contacts / spam",
    "6": "Pricing / quote transparency",
    "7": "Review trust",
    "8": "Dispute protection",

    "9": "Poor lead quality",
    "10": "Fake / invalid leads",
    "11": "Unresponsive leads",
    "12": "Poor ROI / expensive leads",
    "13": "Too much provider competition",
    "14": "Billing / refunds",

    "15": "Customer support",
    "16": "Account / cancellation",
    "17": "Privacy / unwanted contact",
    "18": "Platform usability",
    "19": "Misleading claims / transparency",

    "20": "Other",
}

SEVERITY_OPTIONS = {
    "1": "Low",
    "2": "Medium",
    "3": "High",
}


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if OUTPUT_FILE.exists():

        print("\nExisting labelled file found.")
        print("Resuming previous progress...")

        df = pd.read_csv(
            OUTPUT_FILE,
            low_memory=False
        )

    else:

        if not INPUT_FILE.exists():
            raise FileNotFoundError(
                f"Training sample not found:\n{INPUT_FILE}"
            )

        df = pd.read_csv(
            INPUT_FILE,
            low_memory=False
        )

        manual_columns = [
            "manual_relevance",
            "manual_user_type",
            "manual_complaint_themes",
            "manual_severity",
            "manual_notes",
        ]

        for column in manual_columns:

            if column not in df.columns:
                df[column] = ""

    # Prevent NaN issues
    manual_columns = [
        "manual_relevance",
        "manual_user_type",
        "manual_complaint_themes",
        "manual_severity",
        "manual_notes",
    ]

    for column in manual_columns:
        df[column] = df[column].fillna("").astype(str)

    return df


# ============================================================
# SAVE
# ============================================================

def save_progress(df):

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )


# ============================================================
# DISPLAY HELPERS
# ============================================================

def print_separator():
    print("\n" + "=" * 90)


def print_options(options):

    for number, label in options.items():
        print(f"{number}. {label}")


def ask_single_choice(
    prompt,
    options,
    allow_skip=False
):

    while True:

        value = input(prompt).strip().lower()

        if value == "q":
            return "QUIT"

        if allow_skip and value == "s":
            return "SKIP"

        if value in options:
            return options[value]

        print("Invalid choice. Please try again.")


def ask_themes():

    print("\nCOMPLAINT THEME(S)")
    print("-" * 40)

    print_options(
        THEME_OPTIONS
    )

    print(
        "\nChoose one or more numbers separated by commas."
    )

    print(
        "Example: 9,11,12"
    )

    while True:

        value = input(
            "\nTheme(s): "
        ).strip().lower()

        if value == "q":
            return "QUIT"

        if value == "s":
            return "SKIP"

        choices = [
            item.strip()
            for item in value.split(",")
            if item.strip()
        ]

        if not choices:
            print(
                "Choose at least one theme."
            )
            continue

        invalid = [
            choice
            for choice in choices
            if choice not in THEME_OPTIONS
        ]

        if invalid:
            print(
                f"Invalid option(s): "
                f"{', '.join(invalid)}"
            )
            continue

        themes = []

        for choice in choices:

            theme = THEME_OPTIONS[
                choice
            ]

            if theme not in themes:
                themes.append(theme)

        return " | ".join(
            themes
        )


# ============================================================
# FIND NEXT UNLABELLED REVIEW
# ============================================================

def find_next_review(df):

    required_fields = [
        "manual_relevance",
        "manual_user_type",
        "manual_complaint_themes",
        "manual_severity",
    ]

    for index, row in df.iterrows():

        complete = all(
            str(
                row[field]
            ).strip()
            for field in required_fields
        )

        if not complete:
            return index

    return None


# ============================================================
# DISPLAY REVIEW
# ============================================================

def display_review(
    row,
    position,
    total
):

    print_separator()

    print(
        f"REVIEW {position:,} / {total:,}"
    )

    print_separator()

    print(
        f"Competitor: {row.get('competitor', '')}"
    )

    print(
        f"Rating: {row.get('rating', '')}"
    )

    print(
        f"Date: {row.get('publishedDate', '')}"
    )

    print(
        f"Review ID: {row.get('reviewId', '')}"
    )

    print("\nREVIEW TEXT")
    print("-" * 90)

    print(
        row.get(
            "text",
            ""
        )
    )

    print("\n" + "-" * 90)

    print(
        "V3 prediction (reference only — "
        "do NOT copy it automatically):"
    )

    print(
        f"Relevance: "
        f"{row.get('relevance', '')}"
    )

    print(
        f"User type: "
        f"{row.get('user_type', '')}"
    )

    print(
        f"Themes: "
        f"{row.get('complaint_themes', '')}"
    )

    print(
        f"Severity: "
        f"{row.get('severity', '')}"
    )


# ============================================================
# LABEL ONE REVIEW
# ============================================================

def label_review(
    df,
    index
):

    row = df.loc[index]

    display_review(
        row,
        index + 1,
        len(df)
    )

    print(
        "\nCommands:"
        "\nq = quit and save"
        "\ns = skip this review"
    )

    # --------------------------------------------------------
    # RELEVANCE
    # --------------------------------------------------------

    print("\nRELEVANCE")
    print("-" * 40)

    print_options(
        RELEVANCE_OPTIONS
    )

    relevance = ask_single_choice(
        "\nChoice: ",
        RELEVANCE_OPTIONS,
        allow_skip=True
    )

    if relevance == "QUIT":
        return "QUIT"

    if relevance == "SKIP":
        return "SKIP"

    # --------------------------------------------------------
    # EXCLUDED REVIEW
    # --------------------------------------------------------

    if relevance == "Exclude":

        df.at[
            index,
            "manual_relevance"
        ] = "Exclude"

        df.at[
            index,
            "manual_user_type"
        ] = "Not applicable"

        df.at[
            index,
            "manual_complaint_themes"
        ] = "Not applicable"

        df.at[
            index,
            "manual_severity"
        ] = "Not applicable"

        notes = input(
            "\nWhy is this excluded? "
            "(optional): "
        ).strip()

        df.at[
            index,
            "manual_notes"
        ] = notes

        return "DONE"

    # --------------------------------------------------------
    # USER TYPE
    # --------------------------------------------------------

    print("\nUSER TYPE")
    print("-" * 40)

    print_options(
        USER_TYPE_OPTIONS
    )

    user_type = ask_single_choice(
        "\nChoice: ",
        USER_TYPE_OPTIONS
    )

    if user_type == "QUIT":
        return "QUIT"

    # --------------------------------------------------------
    # THEMES
    # --------------------------------------------------------

    themes = ask_themes()

    if themes == "QUIT":
        return "QUIT"

    if themes == "SKIP":
        return "SKIP"

    # --------------------------------------------------------
    # SEVERITY
    # --------------------------------------------------------

    print("\nSEVERITY")
    print("-" * 40)

    print_options(
        SEVERITY_OPTIONS
    )

    severity = ask_single_choice(
        "\nChoice: ",
        SEVERITY_OPTIONS
    )

    if severity == "QUIT":
        return "QUIT"

    # --------------------------------------------------------
    # NOTES
    # --------------------------------------------------------

    notes = input(
        "\nNotes (optional): "
    ).strip()

    # --------------------------------------------------------
    # SAVE LABELS
    # --------------------------------------------------------

    df.at[
        index,
        "manual_relevance"
    ] = relevance

    df.at[
        index,
        "manual_user_type"
    ] = user_type

    df.at[
        index,
        "manual_complaint_themes"
    ] = themes

    df.at[
        index,
        "manual_severity"
    ] = severity

    df.at[
        index,
        "manual_notes"
    ] = notes

    return "DONE"


# ============================================================
# PROGRESS
# ============================================================

def progress_summary(df):

    completed = (
        df[
            "manual_relevance"
        ]
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )

    total = len(df)

    percent = (
        completed
        / total
        * 100
    )

    print(
        f"\nProgress: "
        f"{completed}/{total} "
        f"({percent:.1f}%)"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\nCompetitor Review "
        "Manual Labelling Tool"
    )

    print(
        "\nThe script saves your work "
        "after every labelled review."
    )

    print(
        "\nPress q whenever you want "
        "to stop."
    )

    df = load_data()

    progress_summary(
        df
    )

    while True:

        index = find_next_review(
            df
        )

        if index is None:

            print_separator()

            print(
                "ALL REVIEWS HAVE BEEN LABELLED."
            )

            save_progress(
                df
            )

            progress_summary(
                df
            )

            print(
                f"\nFinal labelled dataset:\n"
                f"{OUTPUT_FILE}"
            )

            break

        result = label_review(
            df,
            index
        )

        if result == "QUIT":

            save_progress(
                df
            )

            print(
                "\nProgress saved."
            )

            progress_summary(
                df
            )

            print(
                f"\nSaved to:\n"
                f"{OUTPUT_FILE}"
            )

            break

        if result == "SKIP":

            # Put a marker in notes so that the
            # skipped review can be identified later.

            df.at[
                index,
                "manual_notes"
            ] = "SKIPPED"

            save_progress(
                df
            )

            print(
                "\nReview skipped."
            )

            # Move skipped review temporarily to the end
            # so the script does not immediately show it again.

            skipped_row = (
                df.loc[
                    [index]
                ].copy()
            )

            df = (
                pd.concat(
                    [
                        df.drop(
                            index=index
                        ),
                        skipped_row
                    ],
                    ignore_index=True
                )
            )

            continue

        # Save after EVERY review
        save_progress(
            df
        )

        progress_summary(
            df
        )


if __name__ == "__main__":
    main()
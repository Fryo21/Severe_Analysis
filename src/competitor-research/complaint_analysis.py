from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "processed-data"
    / "competitor-research"
    / "negative_reviews_2024_2026.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed-data"
    / "competitor-research"
    / "complaint-analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# USER TYPE KEYWORDS
# ============================================================

PROVIDER_KEYWORDS = [
    "lead",
    "leads",
    "customer lead",
    "potential customer",
    "client",
    "clients",
    "credits",
    "credit",
    "advertising",
    "advertise",
    "business owner",
    "my business",
    "our business",
    "my company",
    "our company",
    "subscription",
    "sales team",
    "sales rep",
    "conversion",
    "return on investment",
    "roi",
    "prospect",
    "prospects",
    "jobs",
    "enquiries",
    "enquiry",
    "inquiries",
    "inquiry",
    "tradesman",
    "tradesperson",
    "contractor",
    "professional",
]

CUSTOMER_KEYWORDS = [
    "hired",
    "hire",
    "looking for",
    "needed someone",
    "needed a",
    "looking for a",
    "contractor I hired",
    "tradesman I hired",
    "company I hired",
    "service I received",
    "work done",
    "job done",
    "quote",
    "quotes",
    "provider",
    "tradesperson",
    "electrician",
    "plumber",
    "builder",
    "cleaner",
    "roofer",
    "gardener",
]


# ============================================================
# COMPLAINT THEMES
# ============================================================

COMPLAINT_THEMES = {

    "Poor lead quality": [
        "poor lead",
        "bad lead",
        "bad leads",
        "poor leads",
        "low quality lead",
        "low quality leads",
        "useless lead",
        "useless leads",
        "waste of leads",
        "lead quality",
        "not genuine",
        "not serious",
    ],

    "Fake or invalid leads": [
        "fake lead",
        "fake leads",
        "fake number",
        "fake phone",
        "invalid number",
        "invalid phone",
        "wrong number",
        "incorrect number",
        "fake email",
        "invalid email",
        "incorrect contact",
        "fake customer",
        "fake enquiry",
    ],

    "Unresponsive leads or customers": [
        "no response",
        "never responded",
        "didn't respond",
        "did not respond",
        "no reply",
        "never replied",
        "didn't reply",
        "did not reply",
        "unresponsive",
        "ghosted",
        "never answer",
        "never answered",
    ],

    "Pricing or cost": [
        "expensive",
        "too expensive",
        "overpriced",
        "cost too much",
        "high cost",
        "high price",
        "pricing",
        "price",
        "fees",
        "fee",
        "charged",
        "charges",
        "credits cost",
        "subscription cost",
    ],

    "Billing or unexpected charges": [
        "unexpected charge",
        "unexpected charges",
        "charged without",
        "charged me",
        "charged again",
        "automatic renewal",
        "auto renewal",
        "renewed",
        "billing",
        "invoice",
        "direct debit",
        "money taken",
        "took money",
    ],

    "Cancellation difficulty": [
        "cancel",
        "cancelled",
        "cancellation",
        "cannot cancel",
        "can't cancel",
        "could not cancel",
        "difficult to cancel",
        "subscription cancellation",
        "terminate",
    ],

    "Refund problems": [
        "refund",
        "refunded",
        "money back",
        "wouldn't refund",
        "would not refund",
        "refused refund",
        "credit refund",
    ],

    "Poor customer support": [
        "customer service",
        "customer support",
        "support team",
        "poor support",
        "bad support",
        "terrible support",
        "no support",
        "ignored",
        "no help",
        "unhelpful",
        "rude",
        "complaint ignored",
    ],

    "Poor provider quality": [
        "poor workmanship",
        "bad workmanship",
        "terrible workmanship",
        "poor quality work",
        "bad work",
        "unfinished work",
        "damaged",
        "damage",
        "cowboy",
        "incompetent",
        "unprofessional",
        "failed to complete",
        "didn't complete",
        "did not complete",
    ],

    "Provider did not show up": [
        "didn't show",
        "did not show",
        "never showed",
        "no show",
        "failed to turn up",
        "didn't turn up",
        "did not turn up",
        "never arrived",
    ],

    "Poor matching": [
        "poor match",
        "bad match",
        "not relevant",
        "irrelevant",
        "wrong service",
        "wrong provider",
        "not suitable",
        "unsuitable",
        "not what i asked",
        "not what I needed",
    ],

    "Too many providers contacting customer": [
        "too many calls",
        "too many messages",
        "too many emails",
        "bombarded",
        "harassed",
        "spam calls",
        "spam emails",
        "constant calls",
        "multiple calls",
        "everyone calling",
    ],

    "Review trust": [
        "fake review",
        "fake reviews",
        "reviews are fake",
        "untrustworthy reviews",
        "review removed",
        "removed my review",
        "deleted my review",
        "review manipulation",
        "misleading reviews",
        "cannot trust reviews",
        "can't trust reviews",
    ],

    "Verification or trust": [
        "not vetted",
        "not verified",
        "verification",
        "vetting",
        "background check",
        "not checked",
        "should have checked",
        "verified",
        "vetted",
    ],

    "Dispute or complaint resolution": [
        "dispute",
        "complaint",
        "complaints",
        "resolution",
        "resolve",
        "resolved",
        "no protection",
        "wouldn't help",
        "would not help",
        "refused to help",
        "nothing they could do",
    ],

    "Account problems": [
        "account blocked",
        "account suspended",
        "account closed",
        "locked out",
        "cannot login",
        "can't login",
        "login problem",
        "account issue",
        "profile removed",
    ],

    "Sales pressure": [
        "sales call",
        "sales calls",
        "sales person",
        "salespeople",
        "sales team",
        "pushy",
        "pressure",
        "pressured",
        "hard sell",
        "cold call",
    ],
}


# ============================================================
# HIGH-SEVERITY TERMS
# ============================================================

HIGH_SEVERITY_KEYWORDS = [
    "fraud",
    "fraudulent",
    "scam",
    "scammed",
    "stolen",
    "theft",
    "police",
    "unsafe",
    "dangerous",
    "injury",
    "legal action",
    "court",
    "threat",
    "threatened",
    "thousands",
    "damage",
    "damaged property",
]


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# USER TYPE CLASSIFICATION
# ============================================================

def classify_user_type(text):

    provider_score = sum(
        keyword in text
        for keyword in PROVIDER_KEYWORDS
    )

    customer_score = sum(
        keyword in text
        for keyword in CUSTOMER_KEYWORDS
    )

    if provider_score > customer_score:
        return "Service Provider / Business"

    if customer_score > provider_score:
        return "Customer"

    return "Unknown"


# ============================================================
# COMPLAINT CLASSIFICATION
# ============================================================

def classify_complaints(text):

    matched_themes = []

    for theme, keywords in COMPLAINT_THEMES.items():

        if any(
            keyword.lower() in text
            for keyword in keywords
        ):
            matched_themes.append(theme)

    if not matched_themes:
        return ["Other / Manual Review"]

    return matched_themes


# ============================================================
# SEVERITY
# ============================================================

def classify_severity(row):

    text = row["clean_text"]

    rating = row["rating"]

    high_severity_match = any(
        keyword in text
        for keyword in HIGH_SEVERITY_KEYWORDS
    )

    if high_severity_match:
        return "High"

    if rating == 1:
        return "High"

    if rating == 2:
        return "Medium"

    return "Low"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    print(
        f"\nLoaded {len(df):,} negative reviews."
    )

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    df["publishedDate"] = pd.to_datetime(
        df["publishedDate"],
        errors="coerce",
        utc=True
    )

    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce"
    )

    df["clean_text"] = (
        df["text"]
        .apply(clean_text)
    )

    df["user_type"] = (
        df["clean_text"]
        .apply(classify_user_type)
    )

    df["complaint_themes"] = (
        df["clean_text"]
        .apply(classify_complaints)
    )

    df["severity"] = (
        df.apply(
            classify_severity,
            axis=1
        )
    )

    return df


# ============================================================
# EXPLODE COMPLAINT THEMES
# ============================================================

def explode_complaints(df):

    complaint_df = df.explode(
        "complaint_themes"
    ).copy()

    complaint_df = complaint_df.rename(
        columns={
            "complaint_themes": "complaint"
        }
    )

    return complaint_df


# ============================================================
# COMPLAINT FREQUENCY
# ============================================================

def analyse_complaint_frequency(complaint_df):

    complaint_counts = (
        complaint_df
        .groupby(
            ["competitor", "complaint"]
        )
        .size()
        .reset_index(
            name="frequency"
        )
    )

    complaint_counts = (
        complaint_counts
        .sort_values(
            ["competitor", "frequency"],
            ascending=[True, False]
        )
    )

    print("\n" + "=" * 70)

    print(
        "TOP COMPLAINT THEMES BY COMPETITOR"
    )

    print("=" * 70)

    for competitor in sorted(
        complaint_df["competitor"].unique()
    ):

        print(
            f"\n{competitor}"
        )

        competitor_results = (
            complaint_counts[
                complaint_counts[
                    "competitor"
                ] == competitor
            ]
            .head(10)
        )

        print(
            competitor_results[
                [
                    "complaint",
                    "frequency"
                ]
            ].to_string(
                index=False
            )
        )

    return complaint_counts


# ============================================================
# USER TYPE ANALYSIS
# ============================================================

def analyse_user_types(df):

    user_type_counts = (
        df.groupby(
            [
                "competitor",
                "user_type"
            ]
        )
        .size()
        .reset_index(
            name="frequency"
        )
    )

    print("\n" + "=" * 70)
    print("USER TYPE DISTRIBUTION")
    print("=" * 70)

    print(
        user_type_counts.to_string(
            index=False
        )
    )

    return user_type_counts


# ============================================================
# COMPLAINT + USER TYPE MATRIX
# ============================================================

def build_weakness_matrix(
    complaint_df
):

    matrix = (
        complaint_df
        .groupby(
            [
                "complaint",
                "competitor",
                "user_type",
                "severity"
            ]
        )
        .size()
        .reset_index(
            name="frequency"
        )
    )

    matrix = matrix.sort_values(
        "frequency",
        ascending=False
    )

    return matrix


# ============================================================
# GRAPH: TOP COMPLAINTS
# ============================================================

def create_complaint_graph(
    complaint_df
):

    graph_data = (
        complaint_df[
            complaint_df["complaint"]
            != "Other / Manual Review"
        ]
        .groupby("complaint")
        .size()
        .sort_values(
            ascending=False
        )
        .head(12)
        .sort_values()
    )

    graph_file = (
        OUTPUT_DIR
        / "top_complaint_themes.png"
    )

    plt.figure(
        figsize=(11, 7)
    )

    graph_data.plot(
        kind="barh"
    )

    plt.title(
        "Most Common Complaint Themes Across Competitors"
    )

    plt.xlabel(
        "Number of complaint mentions"
    )

    plt.ylabel(
        "Complaint theme"
    )

    plt.tight_layout()

    plt.savefig(
        graph_file,
        dpi=300
    )

    plt.close()

    print(
        f"\nComplaint graph saved to:\n{graph_file}"
    )


# ============================================================
# GRAPH: USER TYPE
# ============================================================

def create_user_type_graph(df):

    graph_data = (
        df.groupby(
            [
                "competitor",
                "user_type"
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    graph_file = (
        OUTPUT_DIR
        / "complaints_by_user_type.png"
    )

    ax = graph_data.plot(
        kind="bar",
        figsize=(10, 6)
    )

    ax.set_title(
        "Negative Reviews by Marketplace User Type"
    )

    ax.set_xlabel(
        "Competitor"
    )

    ax.set_ylabel(
        "Number of reviews"
    )

    ax.legend(
        title="User type"
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
        f"\nUser-type graph saved to:\n{graph_file}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_outputs(
    df,
    complaint_df,
    complaint_counts,
    user_type_counts,
    weakness_matrix
):

    classified_file = (
        OUTPUT_DIR
        / "classified_negative_reviews.csv"
    )

    complaints_file = (
        OUTPUT_DIR
        / "complaint_mentions.csv"
    )

    frequency_file = (
        OUTPUT_DIR
        / "complaint_frequency_by_competitor.csv"
    )

    user_type_file = (
        OUTPUT_DIR
        / "user_type_distribution.csv"
    )

    matrix_file = (
        OUTPUT_DIR
        / "competitor_weakness_matrix.csv"
    )

    # Convert lists to readable text
    save_df = df.copy()

    save_df["complaint_themes"] = (
        save_df["complaint_themes"]
        .apply(
            lambda themes:
            " | ".join(themes)
        )
    )

    save_df.to_csv(
        classified_file,
        index=False
    )

    complaint_df.to_csv(
        complaints_file,
        index=False
    )

    complaint_counts.to_csv(
        frequency_file,
        index=False
    )

    user_type_counts.to_csv(
        user_type_file,
        index=False
    )

    weakness_matrix.to_csv(
        matrix_file,
        index=False
    )

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(classified_file)
    print(complaints_file)
    print(frequency_file)
    print(user_type_file)
    print(matrix_file)


# ============================================================
# QUALITY CHECK
# ============================================================

def print_quality_check(df):

    total = len(df)

    unknown_users = (
        df["user_type"]
        .eq("Unknown")
        .sum()
    )

    manual_reviews = (
        df["complaint_themes"]
        .apply(
            lambda themes:
            "Other / Manual Review"
            in themes
        )
        .sum()
    )

    print("\n" + "=" * 70)
    print("CLASSIFICATION QUALITY CHECK")
    print("=" * 70)

    print(
        f"Total reviews: {total:,}"
    )

    print(
        f"Unknown user type: "
        f"{unknown_users:,} "
        f"({unknown_users / total:.1%})"
    )

    print(
        f"Reviews requiring manual complaint review: "
        f"{manual_reviews:,} "
        f"({manual_reviews / total:.1%})"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\nStarting complaint analysis..."
    )

    df = load_data()

    df = prepare_data(df)

    complaint_df = (
        explode_complaints(df)
    )

    user_type_counts = (
        analyse_user_types(df)
    )

    complaint_counts = (
        analyse_complaint_frequency(
            complaint_df
        )
    )

    weakness_matrix = (
        build_weakness_matrix(
            complaint_df
        )
    )

    print_quality_check(df)

    create_complaint_graph(
        complaint_df
    )

    create_user_type_graph(
        df
    )

    save_outputs(
        df,
        complaint_df,
        complaint_counts,
        user_type_counts,
        weakness_matrix
    )

    print(
        "\nComplaint analysis complete."
    )


if __name__ == "__main__":
    main()
from pathlib import Path
import re

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FULL_REVIEW_FILE = (
    PROJECT_ROOT
    / "processed-data"
    / "competitor-research"
    / "negative_reviews_2024_2026.csv"
)

LABELLED_FILE = (
    PROJECT_ROOT
    / "processed-data"
    / "competitor-research"
    / "complaint-analysis-final"
    / "full_reviews_ai_labeled_v3.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed-data"
    / "competitor-research"
    / "complaint-analysis-final"
    / "repair-queue"
)

REPAIR_QUEUE_FILE = (
    OUTPUT_DIR
    / "repair_queue.csv"
)


# ============================================================
# CONSERVATIVE ROLE SIGNALS
# ============================================================

PROVIDER_SIGNAL_PATTERNS = [
    r"\bi(?:'m| am) a supplier\b",
    r"\bi(?:'m| am) a provider\b",
    r"\bi(?:'m| am) a tradesperson\b",
    r"\bi(?:'m| am) a tradesman\b",
    r"\bi(?:'m| am) a contractor\b",
    r"\bi(?:'m| am) an advertiser\b",
    r"\bas a (?:small )?business owner\b",
    r"\bas a (?:small )?business\b",
    r"\badvertis(?:e|ed|ing) (?:my|our) business\b",
    r"\bbuy(?:ing)? leads\b",
    r"\bbought leads\b",
    r"\bpaid for leads\b",
    r"\bpurchase(?:d|ing)? credits\b",
    r"\bbuy(?:ing)? credits\b",
    r"\bbought credits\b",
    r"\bmy leads\b",
    r"\bour leads\b",
    r"\blead credits\b",
    r"\bwin jobs\b",
    r"\bwinning jobs\b",
    r"\bget customers\b",
    r"\bgetting customers\b",
    r"\bget clients\b",
    r"\bgetting clients\b",
    r"\blead conversion\b",
    r"\breturn on investment\b",
    r"\bmy roi\b",
    r"\badvertising package\b",
    r"\bmarketing package\b",
]

CUSTOMER_SIGNAL_PATTERNS = [
    r"\bi hired\b",
    r"\bwe hired\b",
    r"\bi was looking for\b",
    r"\bwe were looking for\b",
    r"\bi am looking for\b",
    r"\bwe are looking for\b",
    r"\bi needed a\b",
    r"\bwe needed a\b",
    r"\bi need a\b",
    r"\bwe need a\b",
    r"\bi requested a quote\b",
    r"\bwe requested a quote\b",
    r"\bi asked for a quote\b",
    r"\bwe asked for a quote\b",
    r"\blooking for a local company\b",
    r"\blooking for a company\b",
    r"\blooking for someone\b",
    r"\bcompanies they suggested\b",
    r"\bdevelopers they suggested\b",
]


# ============================================================
# AUDIT-DERIVED SIGNALS
# ============================================================

YELL_BUSINESS_PATTERNS = [
    r"\bmy business\b",
    r"\bour business\b",
    r"\bmy company\b",
    r"\bour company\b",
    r"\bfor my business\b",
    r"\bfor our business\b",
    r"\badvertis(?:e|ed|ing|ement)\b",
    r"\bmarketing\b",
    r"\bwebsite package\b",
    r"\bmy website\b",
    r"\bour website\b",
    r"\bdomain name\b",
    r"\benom\b",
    r"\bmonthly (?:payment|fee|cost)\b",
    r"\byear(?:ly)? contract\b",
    r"\b12[- ]month contract\b",
    r"\bsubscription\b",
    r"\bgenerate(?:d|s|ing)? leads\b",
    r"\bbusiness leads\b",
]

YELP_PLATFORM_TERMS = [
    "yelp",
    "review",
    "reviews",
    "listing",
    "listed",
    "profile",
    "business page",
    "advertising",
    "advertise",
    "advertised",
    " ads ",
    "lead",
    "leads",
    "rating",
    "ratings",
    "stars",
    "account",
    "signup",
    "sign up",
    "algorithm",
    "filtered",
    "removed",
    "sales call",
    "sales calls",
]


# ============================================================
# HELPERS
# ============================================================

def normalise_text(value):
    if pd.isna(value):
        return ""
    return str(value)


def has_pattern(text, patterns):
    text = normalise_text(text).lower()

    return any(
        re.search(pattern, text)
        for pattern in patterns
    )


def extract_currency_amounts(text):
    text = normalise_text(text)

    raw_amounts = re.findall(
        r"[£$€]\s*([\d,]+(?:\.\d+)?)",
        text,
    )

    amounts = []

    for raw in raw_amounts:
        try:
            amounts.append(
                float(
                    raw.replace(",", "")
                )
            )
        except ValueError:
            continue

    return amounts


# ============================================================
# REPAIR RULES
# ============================================================

def find_repair_reasons(row):

    text = normalise_text(
        row.get("text", "")
    )

    text_lower = text.lower()

    competitor = normalise_text(
        row.get("competitor", "")
    ).strip()

    relevance = normalise_text(
        row.get(
            "manual_relevance",
            "",
        )
    ).strip()

    user_type = normalise_text(
        row.get(
            "manual_user_type",
            "",
        )
    ).strip()

    primary = normalise_text(
        row.get(
            "primary_complaint_theme",
            "",
        )
    ).strip()

    severity = normalise_text(
        row.get(
            "manual_severity",
            "",
        )
    ).strip()

    reasons = []

    provider_signal = has_pattern(
        text,
        PROVIDER_SIGNAL_PATTERNS,
    )

    customer_signal = has_pattern(
        text,
        CUSTOMER_SIGNAL_PATTERNS,
    )


    # --------------------------------------------------------
    # 1. CUSTOMER / PROVIDER SIDE CONFLICTS
    # --------------------------------------------------------

    if (
        user_type == "Customer"
        and provider_signal
        and not customer_signal
    ):
        reasons.append(
            "side_conflict_customer_vs_provider_evidence"
        )

    if (
        user_type
        == "Service Provider / Business"
        and customer_signal
        and not provider_signal
    ):
        reasons.append(
            "side_conflict_provider_vs_customer_evidence"
        )


    # A customer-labelled review talking about leads + ROI/money
    # is worth rechecking as a possible provider/business review.

    if (
        user_type == "Customer"
        and re.search(
            r"\bleads?\b",
            text_lower,
        )
    ):

        money_or_roi_terms = [
            "loss",
            "lost",
            "money",
            "argent",
            "perte",
            "roi",
            "paid",
            "paying",
            "cost",
            "spent",
            "waste",
            "wasted",
            "£",
            "$",
            "€",
        ]

        has_money_or_roi = any(
            term in text_lower
            for term in money_or_roi_terms
        )

        if (
            has_money_or_roi
            and not customer_signal
        ):
            reasons.append(
                "customer_label_with_lead_roi_language"
            )


    # --------------------------------------------------------
    # 2. YELL BUSINESS / ADVERTISING MISCLASSIFICATION
    # --------------------------------------------------------

    if (
        competitor == "Yell"
        and user_type == "Customer"
        and has_pattern(
            text,
            YELL_BUSINESS_PATTERNS,
        )
    ):
        reasons.append(
            "yell_customer_with_business_advertising_signals"
        )


    # --------------------------------------------------------
    # 3. YELP THIRD-PARTY RELEVANCE CHECK
    # --------------------------------------------------------

    if (
        competitor == "Yelp"
        and relevance == "Relevant"
    ):

        padded_text = (
            f" {text_lower} "
        )

        has_platform_reference = any(
            term in padded_text
            for term in YELP_PLATFORM_TERMS
        )

        if not has_platform_reference:
            reasons.append(
                "yelp_relevance_without_platform_reference"
            )


    # --------------------------------------------------------
    # 4. AI EXCLUSIONS / UNCERTAIN LABELS
    # --------------------------------------------------------

    if relevance == "Exclude":
        reasons.append(
            "ai_exclude_recheck"
        )

    if user_type == "Unknown":
        reasons.append(
            "unknown_user_type"
        )

    if primary == "Other":
        reasons.append(
            "other_theme_recheck"
        )


    # --------------------------------------------------------
    # 5. SEVERITY: LARGE EXPLICIT MONEY IMPACT
    # --------------------------------------------------------

    amounts = extract_currency_amounts(
        text
    )

    loss_terms = [
        "lost",
        "loss",
        "wasted",
        "waste",
        "spent",
        "charged",
        "paid",
        "cost me",
        "out of pocket",
        "stole",
        "stolen",
        "perte",
    ]

    has_loss_language = any(
        term in text_lower
        for term in loss_terms
    )

    if (
        severity in {"Low", "Medium"}
        and any(
            amount >= 500
            for amount in amounts
        )
        and has_loss_language
    ):
        reasons.append(
            "large_explicit_money_impact_not_high"
        )

    if (
        severity in {"Low", "Medium"}
        and any(
            term in text_lower
            for term in [
                "thousand",
                "thousands",
                "hundreds",
            ]
        )
        and has_loss_language
    ):
        reasons.append(
            "large_money_words_not_high"
        )


    # --------------------------------------------------------
    # 6. THEME-SPECIFIC SANITY CHECKS
    # --------------------------------------------------------

    if primary == "Unresponsive leads":

        lead_terms = [
            "lead",
            "leads",
            "prospect",
            "customer",
            "customers",
            "enquiry",
            "enquiries",
            "client",
            "clients",
        ]

        support_terms = [
            "call",
            "called",
            "callback",
            "call back",
            "support",
            "customer service",
            "salesperson",
            "sales person",
            "account manager",
        ]

        has_lead_language = any(
            term in text_lower
            for term in lead_terms
        )

        has_support_language = any(
            term in text_lower
            for term in support_terms
        )

        if (
            not has_lead_language
            and has_support_language
        ):
            reasons.append(
                "unresponsive_leads_may_be_support"
            )


    if primary == "Fake / invalid leads":

        lead_terms = [
            "lead",
            "leads",
            "enquiry",
            "enquiries",
            "phone",
            "number",
            "email",
            "contact",
            "client",
            "customer",
        ]

        has_lead_evidence = any(
            term in text_lower
            for term in lead_terms
        )

        if not has_lead_evidence:
            reasons.append(
                "fake_invalid_leads_without_lead_evidence"
            )


    if (
        primary
        == "Pricing / quote transparency"
    ):

        pricing_terms = [
            "price",
            "pricing",
            "quote",
            "quoted",
            "cost",
            "expensive",
            "£",
            "$",
            "€",
            "fee",
            "charge",
        ]

        technical_terms = [
            "attach",
            "upload",
            "photo",
            "photos",
            "site",
            "website",
            "app",
            "error",
            "technical",
        ]

        has_pricing_language = any(
            term in text_lower
            for term in pricing_terms
        )

        has_technical_language = any(
            term in text_lower
            for term in technical_terms
        )

        if (
            not has_pricing_language
            and has_technical_language
        ):
            reasons.append(
                "pricing_theme_may_be_technical_issue"
            )


    if (
        primary
        == "Too many provider contacts / spam"
    ):

        contact_terms = [
            "call",
            "calls",
            "called",
            "message",
            "messages",
            "email",
            "emails",
            "contact",
            "spam",
            "phone",
        ]

        low_response_terms = [
            "reply",
            "response",
            "respond",
            "nobody",
            "no one",
            "not come back",
            "never came back",
        ]

        has_contact_language = any(
            term in text_lower
            for term in contact_terms
        )

        has_low_response_language = any(
            term in text_lower
            for term in low_response_terms
        )

        if (
            not has_contact_language
            and has_low_response_language
        ):
            reasons.append(
                "spam_theme_may_be_low_response"
            )


    return reasons


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not FULL_REVIEW_FILE.exists():
        raise FileNotFoundError(
            "Full negative review file not found:\n"
            f"{FULL_REVIEW_FILE}"
        )

    if not LABELLED_FILE.exists():
        raise FileNotFoundError(
            "Labelled review file not found:\n"
            f"{LABELLED_FILE}"
        )

    full = pd.read_csv(
        FULL_REVIEW_FILE,
        low_memory=False,
    )

    labelled = pd.read_csv(
        LABELLED_FILE,
        low_memory=False,
    )

    full["reviewId"] = (
        full["reviewId"]
        .astype(str)
    )

    labelled["reviewId"] = (
        labelled["reviewId"]
        .astype(str)
    )

    return full, labelled


# ============================================================
# BUILD SUSPICIOUS AI LABEL QUEUE
# ============================================================

def build_suspicious_ai_queue(
    labelled,
):

    ai_rows = (
        labelled[
            labelled[
                "annotation_status"
            ]
            .astype(str)
            .eq(
                "ai_label_needs_audit"
            )
        ]
        .copy()
    )

    ai_rows[
        "repair_reason_list"
    ] = ai_rows.apply(
        find_repair_reasons,
        axis=1,
    )

    suspicious = (
        ai_rows[
            ai_rows[
                "repair_reason_list"
            ]
            .map(bool)
        ]
        .copy()
    )

    suspicious[
        "repair_queue_type"
    ] = "suspect_ai_label"

    suspicious[
        "repair_reasons"
    ] = (
        suspicious[
            "repair_reason_list"
        ]
        .apply(
            lambda reasons:
            " | ".join(reasons)
        )
    )

    suspicious.drop(
        columns=[
            "repair_reason_list",
        ],
        inplace=True,
    )

    return suspicious


# ============================================================
# BUILD PREVIOUSLY SKIPPED / UNLABELLED QUEUE
# ============================================================

def build_unlabelled_queue(
    full,
    labelled,
):

    labelled_ids = set(
        labelled[
            "reviewId"
        ]
        .astype(str)
    )

    unlabelled = (
        full[
            ~full[
                "reviewId"
            ]
            .astype(str)
            .isin(
                labelled_ids
            )
        ]
        .copy()
    )

    if unlabelled.empty:
        return unlabelled

    label_columns = [
        "manual_relevance",
        "manual_user_type",
        "primary_complaint_theme",
        "secondary_complaint_theme",
        "manual_complaint_themes",
        "manual_severity",
        "annotation_method",
        "annotation_status",
    ]

    for column in label_columns:
        if column not in unlabelled.columns:
            unlabelled[column] = ""

    unlabelled[
        "annotation_status"
    ] = "unlabelled_retry"

    unlabelled[
        "repair_queue_type"
    ] = "unlabelled_retry"

    unlabelled[
        "repair_reasons"
    ] = (
        "previous_full_run_skipped_or_unlabelled"
    )

    return unlabelled


# ============================================================
# COMBINE AND SAVE REPAIR QUEUE
# ============================================================

def build_repair_queue(
    full,
    labelled,
):

    suspicious = (
        build_suspicious_ai_queue(
            labelled
        )
    )

    unlabelled = (
        build_unlabelled_queue(
            full,
            labelled,
        )
    )

    parts = []

    if not suspicious.empty:
        parts.append(
            suspicious
        )

    if not unlabelled.empty:
        parts.append(
            unlabelled
        )

    if not parts:
        return pd.DataFrame()

    queue = pd.concat(
        parts,
        ignore_index=True,
        sort=False,
    )

    queue = (
        queue
        .drop_duplicates(
            subset=[
                "reviewId",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    preferred_columns = [
        "repair_queue_type",
        "repair_reasons",
        "reviewId",
        "rating",
        "publishedDate",
        "text",
        "competitor",
        "year",
        "manual_relevance",
        "manual_user_type",
        "primary_complaint_theme",
        "secondary_complaint_theme",
        "manual_complaint_themes",
        "manual_severity",
        "annotation_method",
        "annotation_status",
    ]

    existing_preferred = [
        column
        for column in preferred_columns
        if column in queue.columns
    ]

    remaining_columns = [
        column
        for column in queue.columns
        if column not in existing_preferred
    ]

    queue = queue[
        existing_preferred
        + remaining_columns
    ]

    return queue


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    full,
    labelled,
    queue,
):

    frozen_count = (
        labelled[
            "annotation_status"
        ]
        .astype(str)
        .eq(
            "frozen_v3_label"
        )
        .sum()
    )

    ai_count = (
        labelled[
            "annotation_status"
        ]
        .astype(str)
        .eq(
            "ai_label_needs_audit"
        )
        .sum()
    )

    if queue.empty:
        suspicious_count = 0
        unlabelled_count = 0
    else:
        suspicious_count = (
            queue[
                "repair_queue_type"
            ]
            .eq(
                "suspect_ai_label"
            )
            .sum()
        )

        unlabelled_count = (
            queue[
                "repair_queue_type"
            ]
            .eq(
                "unlabelled_retry"
            )
            .sum()
        )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "V3 REPAIR QUEUE SUMMARY"
    )

    print(
        "=" * 72
    )

    print(
        f"Full negative reviews: "
        f"{len(full):,}"
    )

    print(
        f"Currently labelled: "
        f"{len(labelled):,}"
    )

    print(
        f"Frozen V3 labels: "
        f"{frozen_count:,}"
    )

    print(
        f"AI labels: "
        f"{ai_count:,}"
    )

    print(
        f"\nSuspicious AI labels flagged: "
        f"{suspicious_count:,}"
    )

    print(
        f"Previously skipped/unlabelled: "
        f"{unlabelled_count:,}"
    )

    print(
        f"Total repair queue: "
        f"{len(queue):,}"
    )


    if not queue.empty:

        print(
            "\nQueue by competitor:"
        )

        print(
            queue[
                "competitor"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )


        suspicious = (
            queue[
                queue[
                    "repair_queue_type"
                ]
                .eq(
                    "suspect_ai_label"
                )
            ]
        )

        if not suspicious.empty:

            reason_counts = (
                suspicious[
                    "repair_reasons"
                ]
                .str.split(
                    " | ",
                    regex=False,
                )
                .explode()
                .value_counts()
            )

            print(
                "\nSuspicious-label reasons:"
            )

            print(
                reason_counts.to_string()
            )


    print(
        "\nRepair queue written to:"
    )

    print(
        REPAIR_QUEUE_FILE
    )

    print(
        "\nThe original labelled file "
        "was NOT modified."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    full, labelled = (
        load_data()
    )

    queue = (
        build_repair_queue(
            full,
            labelled,
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    queue.to_csv(
        REPAIR_QUEUE_FILE,
        index=False,
    )

    print_summary(
        full,
        labelled,
        queue,
    )


if __name__ == "__main__":
    main()

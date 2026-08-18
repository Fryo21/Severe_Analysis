from pathlib import Path
import argparse
import json
import time
from unittest import skip
import urllib.error
import urllib.request

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GOLD_FILE = (
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
    / "llm_gold_predictions.csv"
)

MODEL = "qwen3:4b-instruct"

OLLAMA_URL = "http://localhost:11434/api/chat"

MAX_RETRIES = 3


# ============================================================
# FIXED TAXONOMY
# ============================================================

THEMES = [
    "Finding a provider / low response",
    "Poor matching / location",
    "Provider quality",
    "Provider verification / safety",
    "Too many provider contacts / spam",
    "Pricing / quote transparency",
    "Review trust",
    "Dispute protection",

    "Poor lead quality",
    "Fake / invalid leads",
    "Unresponsive leads",
    "Poor ROI / expensive leads",
    "Too much provider competition",
    "Billing / refunds",

    "Customer support",
    "Account / cancellation",
    "Privacy / unwanted contact",
    "Platform usability",
    "Misleading claims / transparency",

    "Other",
]


# ============================================================
# JSON SCHEMA
# ============================================================

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "relevance": {
            "type": "string",
            "enum": [
                "Relevant",
                "Exclude",
                "Unsure"
            ]
        },

        "user_type": {
            "type": "string",
            "enum": [
                "Customer",
                "Service Provider / Business",
                "Unknown",
                "Not applicable"
            ]
        },

        "themes": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": THEMES
            }
        },

        "severity": {
            "type": "string",
            "enum": [
                "Low",
                "Medium",
                "High",
                "Not applicable"
            ]
        },

        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1
        },

        "reason": {
            "type": "string"
        }
    },

    "required": [
        "relevance",
        "user_type",
        "themes",
        "severity",
        "confidence",
        "reason"
    ]
}


# ============================================================
# PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are classifying negative Trustpilot reviews for competitor research
on local-service marketplace platforms:

- Bark
- Checkatrade
- Yell
- Yelp

Read the ENTIRE review and classify the reviewer's actual experience.
Do not classify from isolated keywords.

Follow this decision order:

1. Is the review relevant to the marketplace/platform?
2. Is the reviewer a Customer or Service Provider / Business?
3. What complaint theme or themes are actually stated?
4. How severe is the complaint?

============================================================
1. RELEVANCE
============================================================

Relevant:
The review describes an experience with the marketplace/platform,
including finding a provider, hiring a provider through the platform,
buying leads, advertising a business, managing a listing, billing,
support, reviews, disputes, or platform functionality.

A review can still be Relevant even if:
- it contains some positive comments;
- the reviewer eventually found someone;
- the complaint concerns a provider found through the marketplace.

Exclude:
The review is solely about an unrelated company/product or solely about
a restaurant, hotel, shop, mechanic, etc., with no meaningful complaint
about the marketplace/platform experience.

Unsure:
There is genuinely insufficient information.

IMPORTANT:
Do NOT exclude a review merely because it contains positive statements.
If a marketplace weakness is described, it is Relevant.

============================================================
2. USER TYPE
============================================================

Customer:
Someone using the platform to FIND, CONTACT, COMPARE, HIRE or REVIEW
a provider/business.

Strong Customer evidence includes:
- "I hired..."
- "I needed..."
- "I was looking for..."
- "I requested a quote..."
- "the contractor/provider I found..."
- a tradesperson performed work for the reviewer
- the reviewer paid a provider for work

Service Provider / Business:
A tradesperson, freelancer, professional, advertiser or business using
the platform to GET CUSTOMERS, BUY LEADS, ADVERTISE or manage a
business listing.

STRONG Provider/Business evidence includes:
- buying or paying for LEADS
- buying or using CREDITS
- paying for advertising
- receiving customer enquiries
- trying to win jobs or customers
- discussing lead conversion or ROI
- managing a business listing/profile
- being charged for a business subscription
- prospective customers failing to respond

IMPORTANT PRECEDENCE RULE:

If the reviewer says they PAID FOR LEADS, BOUGHT CREDITS,
RECEIVED LEADS, ADVERTISED THEIR BUSINESS, or was trying to
GET JOBS/CUSTOMERS, classify them as:

Service Provider / Business

Do not classify them as Customer simply because the review also uses
words such as customer, contractor, company, quote or service.

Unknown:
Use only when the side genuinely cannot be inferred.

============================================================
3. COMPLAINT THEMES
============================================================

Choose ONLY themes directly supported by the review.

Do NOT invent secondary themes merely because they seem plausible.

A Relevant review must have AT LEAST ONE theme.


CUSTOMER EXPERIENCES
--------------------

Finding a provider / low response:
Customer cannot get providers, quotes or responses.

Poor matching / location:
Provider is wrong category, unsuitable, irrelevant, or geographically
too far away.

IMPORTANT:
Contacting several providers or receiving several responses does NOT,
by itself, mean Poor matching / location.

Provider quality:
Bad workmanship, poor service, no-show, ghosting, incomplete work,
unprofessional provider.

Provider verification / safety:
Fraud, scams, lack of vetting, identity/trust concerns, unsafe provider.

Too many provider contacts / spam:
Customer receives excessive calls/messages from providers after
submitting a request.

Pricing / quote transparency:
Provider quote or price is unclear, unexpectedly increased, misleading
or excessive.

Review trust:
Fake, manipulated, removed, filtered or otherwise untrustworthy reviews.

Dispute protection:
Platform fails to protect the customer or meaningfully resolve a
problem with a provider.


PROVIDER / BUSINESS EXPERIENCES
-------------------------------

Poor lead quality:
Leads have low intent, unrealistic needs/budgets, are unsuitable,
or unlikely to purchase.

Fake / invalid leads:
False enquiries, fake people, wrong phone/email details or leads that
were never genuine.

Unresponsive leads:
The prospective customer does not answer calls/messages or stops
responding.

Poor ROI / expensive leads:
Lead credits, advertising or customer-acquisition costs produce poor
value, wasted money, few jobs, or inadequate return.

IMPORTANT:
If a provider paid for leads/credits and complains that they produced
little or no business, this is Poor ROI / expensive leads.

Too much provider competition:
Too many providers compete for the same lead or opportunity.

Billing / refunds:
Platform charges, subscription, renewal, credits, refund or payment
problems.


BOTH SIDES
----------

Customer support:
Poor, inaccessible, unhelpful or ineffective PLATFORM support.

Account / cancellation:
Login, account access, suspension, account deletion, subscription
cancellation or account-management problem.

Privacy / unwanted contact:
Unsolicited communication, spam, use of personal information,
data/privacy concern.

Platform usability:
Website, application, technical or automated-system problem.

Misleading claims / transparency:
Misleading sales claims, unclear terms, deceptive promises or
misrepresentation by the PLATFORM.

Review trust can also apply to a business owner when the complaint
concerns Yelp/Yell/Bark/Checkatrade's review system.

============================================================
STRICT THEME SELECTION RULES
============================================================

Use the SMALLEST set of themes that fully describes the complaint.

Default to ONE theme.

Use TWO or more themes only when the review contains clearly separate,
explicit complaints.

Do NOT add a theme merely because it commonly occurs together with
another theme.

Each selected theme must have direct textual evidence.

------------------------------------------------------------
MARKETPLACE SIDE RESTRICTIONS
------------------------------------------------------------

If user_type = Customer:

Allowed customer-specific themes:
- Finding a provider / low response
- Poor matching / location
- Provider quality
- Provider verification / safety
- Too many provider contacts / spam
- Pricing / quote transparency
- Review trust
- Dispute protection

Also allowed shared themes:
- Customer support
- Account / cancellation
- Privacy / unwanted contact
- Platform usability
- Misleading claims / transparency

DO NOT use these provider/business themes for a Customer:
- Poor lead quality
- Fake / invalid leads
- Unresponsive leads
- Poor ROI / expensive leads
- Too much provider competition
- Billing / refunds


If user_type = Service Provider / Business:

Allowed provider-specific themes:
- Poor lead quality
- Fake / invalid leads
- Unresponsive leads
- Poor ROI / expensive leads
- Too much provider competition
- Billing / refunds

Also allowed shared themes:
- Customer support
- Account / cancellation
- Privacy / unwanted contact
- Platform usability
- Misleading claims / transparency
- Review trust

Do NOT use customer-side provider-service themes simply because
the review mentions customers or providers.

------------------------------------------------------------
IMPORTANT DISTINCTIONS
------------------------------------------------------------

Provider verification / safety:
Use ONLY when the review explicitly raises fraud, scam, vetting,
identity, legitimacy, safety or trustworthiness concerns.

Bad workmanship, a no-show, poor communication, no invoice or an
unprofessional provider alone does NOT mean verification/safety.

Billing / refunds:
For this taxonomy, use mainly when a Service Provider / Business
complains about PLATFORM charges, credits, subscription, renewals,
payments or refunds.

A Customer complaining about money paid directly to a contractor or
tradesperson is NOT automatically Billing / refunds.

Customer support:
Use ONLY when the reviewer explicitly describes dealing with platform
support/customer service or being unable to obtain platform support.

A frustrating form, onboarding flow or rejected request alone is not
Customer support.

Poor ROI / expensive leads:
Use when the business explicitly complains about money/value/credits/
advertising return.

Unresponsive leads alone does NOT automatically imply poor ROI.

Too much provider competition:
Use when a Service Provider / Business complains that too many
providers compete for the same leads.

Too many provider contacts / spam:
Use when a Customer complains that too many providers contact them.

Poor matching / location:
Use only for wrong category, unsuitable provider or wrong geographic
location.

Provider quality:
Use for workmanship, service quality, no-show, ghosting,
professionalism or incomplete work.


============================================================
MULTIPLE THEMES
============================================================

Use multiple themes only when the review clearly contains multiple
distinct complaints.

Example:

"I bought expensive Bark credits. Most customers never replied and
Bark refused to refund them."

Correct:
- Unresponsive leads
- Poor ROI / expensive leads
- Billing / refunds

Do NOT add themes that are only implied.

============================================================
NON-ENGLISH REVIEWS
============================================================

If the review is not in English:

Understand or translate its meaning internally first, then perform the
same classification.

Do not output a translation.

============================================================
4. SEVERITY
============================================================

Low:
Minor inconvenience, small usability/friction issue.

Medium:
Meaningful financial/time loss, bad workmanship, unusable leads,
billing/refund problem, persistent support failure, substantial
matching issue.

High:
Fraud/scam, substantial financial loss, serious safety issue,
significant property damage, or severe failure of customer protection.

Do NOT determine severity from star rating alone.

============================================================
5. CONFIDENCE
============================================================

Return a value between 0 and 1 for confidence in the COMPLETE
classification.

Do not use high confidence when the review is genuinely ambiguous.

============================================================
OUTPUT
============================================================

For relevance = Exclude:
- user_type = "Not applicable"
- themes = []
- severity = "Not applicable"

For relevance = Relevant:
- identify the user type where reasonably possible
- return at least one complaint theme
- return Low, Medium or High severity

Keep "reason" extremely short:
ONE sentence, maximum about 20 words.

Return only data matching the supplied JSON schema.


"""

def build_review_prompt(row):

    competitor = str(
        row.get("competitor", "")
    )

    rating = str(
        row.get("rating", "")
    )

    text = str(
        row.get("text", "")
    )

    return f"""
Classify this review.

Competitor: {competitor}
Rating: {rating}

Review:
{text}
""".strip()


# ============================================================
# OLLAMA CALL
# ============================================================

def call_ollama(row):

    payload = {
        "model": MODEL,
        "keep_alive": "30m",
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": build_review_prompt(row)
            }
        ],

        "stream": False,

        "format": OUTPUT_SCHEMA,

        "options": {
            "temperature": 0,
            "seed": 42,
            "num_ctx": 4096
        }
    }

    encoded = json.dumps(
        payload
    ).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=encoded,
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            with urllib.request.urlopen(
                request,
                timeout=300
            ) as response:

                body = json.loads(
                    response.read()
                    .decode("utf-8")
                )

            content = (
                body["message"]["content"]
            )

            result = json.loads(
                content
            )

            validate_prediction(
                result
            )

            return result

        except Exception as error:

            print(
                f"    Attempt "
                f"{attempt}/{MAX_RETRIES} "
                f"failed: {error}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(2)

    return None


# ============================================================
# VALIDATE MODEL RESPONSE
# ============================================================

def validate_prediction(result):

    required = [
        "relevance",
        "user_type",
        "themes",
        "severity",
        "confidence",
        "reason",
    ]

    for field in required:

        if field not in result:

            raise ValueError(
                f"Missing field: {field}"
            )

    if result["relevance"] not in {
        "Relevant",
        "Exclude",
        "Unsure"
    }:

        raise ValueError(
            "Invalid relevance"
        )

    if result["user_type"] not in {
        "Customer",
        "Service Provider / Business",
        "Unknown",
        "Not applicable"
    }:

        raise ValueError(
            "Invalid user type"
        )

    if result["severity"] not in {
        "Low",
        "Medium",
        "High",
        "Not applicable"
    }:

        raise ValueError(
            "Invalid severity"
        )

    if not isinstance(
        result["themes"],
        list
    ):

        raise ValueError(
            "themes must be a list"
        )

    invalid_themes = [
        theme
        for theme in result["themes"]
        if theme not in THEMES
    ]

    if invalid_themes:

        raise ValueError(
            f"Invalid themes: "
            f"{invalid_themes}"
        )


# ============================================================
# LOAD GOLD DATA
# ============================================================

def load_gold():

    if not GOLD_FILE.exists():

        raise FileNotFoundError(
            f"Gold validation file "
            f"not found:\n{GOLD_FILE}"
        )

    df = pd.read_csv(
        GOLD_FILE,
        low_memory=False
    )

    required = [
        "reviewId",
        "competitor",
        "rating",
        "text",
        "manual_relevance",
        "manual_user_type",
        "manual_complaint_themes",
        "manual_severity",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Gold validation set is "
            "missing columns:\n"
            + "\n".join(missing)
        )

    df["reviewId"] = (
        df["reviewId"]
        .astype(str)
    )

    return df


# ============================================================
# EXISTING PREDICTIONS / RESUME
# ============================================================

def load_existing_predictions():

    if not OUTPUT_FILE.exists():

        return pd.DataFrame()

    df = pd.read_csv(
        OUTPUT_FILE,
        low_memory=False
    )

    if "reviewId" in df.columns:

        df["reviewId"] = (
            df["reviewId"]
            .astype(str)
        )

    return df


# ============================================================
# RUN CLASSIFICATION
# ============================================================

def classify_reviews(
    gold,
    limit=None,
    skip=0
    ):

    existing = (
        load_existing_predictions()
    )

    completed_ids = set()

    if not existing.empty:

        completed_ids = set(
            existing["reviewId"]
        )

        print(
            f"\nExisting predictions found: "
            f"{len(existing):,}"
        )

    # Apply skip to the ORIGINAL gold dataset first.
# This makes --skip an absolute position in the 100-review gold set.
    candidate_reviews = gold.iloc[skip:].copy()

    # Then remove anything already classified.
    remaining = candidate_reviews[
    ~candidate_reviews["reviewId"].isin(completed_ids)
    ].copy()
    if limit is not None:

        remaining = (
            remaining.head(limit)
        )

    print(
        f"Reviews to classify now: "
        f"{len(remaining):,}"
    )

    new_rows = []

    for count, (_, row) in enumerate(
        remaining.iterrows(),
        start=1
    ):

        print(
            f"\n[{count}/{len(remaining)}] "
            f"{row['competitor']} "
            f"| Rating {row['rating']}"
        )

        prediction = (
            call_ollama(row)
        )

        if prediction is None:

            print(
                "    Classification failed. "
                "Skipping review."
            )

            continue

        result_row = {
            "reviewId":
                str(row["reviewId"]),

            "competitor":
                row["competitor"],

            "rating":
                row["rating"],

            "text":
                row["text"],

            # Gold labels
            "gold_relevance":
                row["manual_relevance"],

            "gold_user_type":
                row["manual_user_type"],

            "gold_themes":
                row[
                    "manual_complaint_themes"
                ],

            "gold_severity":
                row["manual_severity"],

            # LLM predictions
            "llm_relevance":
                prediction["relevance"],

            "llm_user_type":
                prediction["user_type"],

            "llm_themes":
                " | ".join(
                    prediction["themes"]
                ),

            "llm_severity":
                prediction["severity"],

            "llm_confidence":
                prediction["confidence"],

            "llm_reason":
                prediction["reason"],
        }

        new_rows.append(
            result_row
        )

        # Save after EVERY review
        current_new = pd.DataFrame(
            new_rows
        )

        if existing.empty:

            combined = current_new

        else:

            combined = pd.concat(
                [
                    existing,
                    current_new
                ],
                ignore_index=True
            )

        combined.to_csv(
            OUTPUT_FILE,
            index=False
        )

        print(
            "    "
            f"{prediction['user_type']} | "
            f"{', '.join(prediction['themes'])} | "
            f"{prediction['severity']} | "
            f"confidence "
            f"{prediction['confidence']:.2f}"
        )

    return load_existing_predictions()


# ============================================================
# THEME PARSING
# ============================================================

def parse_themes(value):

    if pd.isna(value):
        return set()

    text = str(value).strip()

    if not text:
        return set()

    if text.lower() in {
        "not applicable",
        "nan"
    }:
        return set()

    return {
        item.strip()
        for item in text.split("|")
        if item.strip()
    }


# ============================================================
# METRICS
# ============================================================

def exact_accuracy(
    gold_series,
    predicted_series
):

    valid = (
        gold_series.notna()
        & predicted_series.notna()
    )

    if valid.sum() == 0:
        return 0.0

    return (
        gold_series[valid]
        .astype(str)
        .str.strip()
        .eq(
            predicted_series[valid]
            .astype(str)
            .str.strip()
        )
        .mean()
    )


def calculate_theme_metrics(df):

    tp = 0
    fp = 0
    fn = 0
    exact_matches = 0
    reviewed = 0

    for _, row in df.iterrows():

        if (
            str(
                row["gold_relevance"]
            ).strip()
            != "Relevant"
        ):
            continue

        gold = parse_themes(
            row["gold_themes"]
        )

        predicted = parse_themes(
            row["llm_themes"]
        )

        if not gold:
            continue

        reviewed += 1

        if gold == predicted:
            exact_matches += 1

        tp += len(
            gold & predicted
        )

        fp += len(
            predicted - gold
        )

        fn += len(
            gold - predicted
        )

    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn)
        else 0
    )

    f1 = (
        2
        * precision
        * recall
        / (precision + recall)
        if (precision + recall)
        else 0
    )

    exact_match = (
        exact_matches / reviewed
        if reviewed
        else 0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact_match,
    }


def evaluate_predictions(df):

    if df.empty:

        print(
            "\nNo predictions available "
            "for evaluation."
        )

        return

    print(
        "\n"
        + "=" * 72
    )

    print(
        "QWEN GOLD VALIDATION RESULTS"
    )

    print(
        "=" * 72
    )

    print(
        f"Reviews classified: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # RELEVANCE
    # --------------------------------------------------------

    relevance_accuracy = (
        exact_accuracy(
            df["gold_relevance"],
            df["llm_relevance"]
        )
    )

    print(
        f"\nRelevance accuracy: "
        f"{relevance_accuracy:.1%}"
    )

    # --------------------------------------------------------
    # USER TYPE
    # Only relevant gold reviews
    # --------------------------------------------------------

    relevant = df[
        df[
            "gold_relevance"
        ]
        .astype(str)
        .str.strip()
        .eq("Relevant")
    ].copy()

    if not relevant.empty:

        user_accuracy = (
            exact_accuracy(
                relevant[
                    "gold_user_type"
                ],
                relevant[
                    "llm_user_type"
                ]
            )
        )

        severity_accuracy = (
            exact_accuracy(
                relevant[
                    "gold_severity"
                ],
                relevant[
                    "llm_severity"
                ]
            )
        )

        print(
            f"User-type accuracy: "
            f"{user_accuracy:.1%}"
        )

        print(
            f"Severity accuracy: "
            f"{severity_accuracy:.1%}"
        )

    # --------------------------------------------------------
    # THEMES
    # --------------------------------------------------------

    theme_metrics = (
        calculate_theme_metrics(
            df
        )
    )

    print(
        "\nComplaint-theme metrics:"
    )

    print(
        f"Precision: "
        f"{theme_metrics['precision']:.1%}"
    )

    print(
        f"Recall: "
        f"{theme_metrics['recall']:.1%}"
    )

    print(
        f"F1 score: "
        f"{theme_metrics['f1']:.1%}"
    )

    print(
        f"Exact theme-set match: "
        f"{theme_metrics['exact_match']:.1%}"
    )

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    confidence = pd.to_numeric(
        df[
            "llm_confidence"
        ],
        errors="coerce"
    )

    print(
        f"\nAverage LLM confidence: "
        f"{confidence.mean():.2f}"
    )

    low_confidence = (
        confidence < 0.75
    ).sum()

    print(
        f"Predictions below 0.75 "
        f"confidence: "
        f"{low_confidence:,}"
    )

    print(
        "\nPredictions saved to:"
    )

    print(
        OUTPUT_FILE
    )


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

def get_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Validate local Qwen complaint "
            "classification against the "
            "gold validation set."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Classify only this many new "
            "reviews. Useful for testing."
        )
    )

    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Skip this many gold-validation reviews before classification."
    )
    

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = get_arguments()

    print(
        "\nLocal LLM validation"
    )

    print(
        f"Model: {MODEL}"
    )

    print(
        f"Gold file: {GOLD_FILE}"
    )

    gold = load_gold()

    print(
        f"\nGold validation reviews: "
        f"{len(gold):,}"
    )

    predictions = (
        classify_reviews(
            gold,
            limit=args.limit,
            skip=args.skip
        )
    )

    evaluate_predictions(
        predictions
    )


if __name__ == "__main__":
    main()
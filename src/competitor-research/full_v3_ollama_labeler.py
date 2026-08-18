from pathlib import Path
import argparse
import json
import re
import time
import urllib.request

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

V3_LABEL_FILE = (
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
    / "complaint-analysis-final"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "full_reviews_ai_labeled_v3.csv"
)

AUDIT_FILE = (
    OUTPUT_DIR
    / "full_reviews_ai_labeled_v3_audit_sample.csv"
)


# ============================================================
# OLLAMA CONFIG
# ============================================================

MODEL = "qwen3:4b-instruct"

OLLAMA_URL = (
    "http://localhost:11434/api/chat"
)

MAX_RETRIES = 2

REQUEST_TIMEOUT_SECONDS = 300


# ============================================================
# FROZEN V3 CODEBOOK
# ============================================================

CUSTOMER_THEMES = [
    "Finding a provider / low response",
    "Poor matching / location",
    "Provider quality / no-show",
    "Provider verification / safety",
    "Too many provider contacts / spam",
    "Pricing / quote transparency",
    "Review trust / moderation",
    "Dispute resolution / protection",
]

PROVIDER_THEMES = [
    "Poor lead quality",
    "Fake / invalid leads",
    "Unresponsive leads",
    "Poor ROI / expensive leads",
    "Too much provider competition",
    "Billing / refunds",
]

SHARED_THEMES = [
    "Customer support",
    "Account access / cancellation",
    "Privacy / unwanted contact",
    "Platform usability / technical issues",
    "Misleading claims / transparency",
    "Sales pressure / contract terms",
    "Listing / profile accuracy",
    "Other",
]

THEMES = (
    CUSTOMER_THEMES
    + PROVIDER_THEMES
    + SHARED_THEMES
)

VALID_RELEVANCE = {
    "Relevant",
    "Exclude",
}

VALID_USER_TYPES = {
    "Customer",
    "Service Provider / Business",
    "Unknown",
    "Not applicable",
}

VALID_SEVERITIES = {
    "Low",
    "Medium",
    "High",
    "Not applicable",
}


# ============================================================
# STRONG USER-TYPE SIGNALS
# ============================================================

PROVIDER_SIGNAL_PATTERNS = [

    r"\bi(?:'m| am) a supplier\b",
    r"\bi(?:'m| am) a provider\b",
    r"\bbuy(?:ing)? leads\b",
    r"\bbought leads\b",
    r"\bpaid for leads\b",
    r"\bpurchase(?:d|ing)? credits\b",
    r"\bbuy(?:ing)? credits\b",
    r"\bmy leads\b",
    r"\blead credits\b",
    r"\bwin jobs\b",
    r"\bget customers\b",
    r"\blead conversion\b",
    r"\breturn on investment\b",
    r"\bmy roi\b",
    r"\badvertising package\b",
    r"\bmarketing package\b",

    r"\bi(?:'m| am) a supplier\b",
    r"\bi(?:'m| am) a provider\b",
    r"\bi(?:'m| am) a tradesperson\b",
    r"\bi(?:'m| am) a tradesman\b",
    r"\bi(?:'m| am) a contractor\b",
    r"\bi(?:'m| am) an advertiser\b",

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

    r"\bmy customers\b",
    r"\bour customers\b",

    r"\bmy clients\b",
    r"\bour clients\b",

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


def infer_strong_user_type(text):

    text = str(text).lower()

    provider_hits = sum(
        bool(
            re.search(
                pattern,
                text,
            )
        )
        for pattern
        in PROVIDER_SIGNAL_PATTERNS
    )

    customer_hits = sum(
        bool(
            re.search(
                pattern,
                text,
            )
        )
        for pattern
        in CUSTOMER_SIGNAL_PATTERNS
    )

    if (
        provider_hits > 0
        and customer_hits == 0
    ):
        return (
            "Service Provider / Business"
        )

    if (
        customer_hits > 0
        and provider_hits == 0
    ):
        return "Customer"

    return None


# ============================================================
# V3 SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You classify negative reviews from Bark, Checkatrade, Yell and Yelp
for marketplace competitor research.

Use ONLY the frozen V3 labels below.

Return JSON only:

{
  "relevance": "...",
  "user_type": "...",
  "primary_theme": "...",
  "secondary_theme": "...",
  "severity": "..."
}

============================================================
RELEVANCE
============================================================

Relevant:

The complaint concerns the platform, marketplace, providers,
leads, advertising, listings, reviews, billing, support,
accounts, contracts, quotes, matching or disputes.

Exclude:

The review is solely about an unrelated third-party company,
restaurant, product, hotel, shop or service and contains no
meaningful platform complaint.

If Exclude:

user_type = Not applicable
primary_theme = Not applicable
secondary_theme = Not applicable
severity = Not applicable


============================================================
USER TYPE
============================================================

CUSTOMER

The reviewer uses the marketplace to:

- find a provider
- request a quote
- compare providers
- hire a provider
- receive work/services


SERVICE PROVIDER / BUSINESS

The reviewer uses the marketplace to:

- advertise their business
- purchase leads
- purchase credits
- contact prospects
- win jobs
- obtain customers
- manage a business profile
- pay platform advertising fees
- discuss lead ROI


UNKNOWN

Use only if the marketplace side genuinely cannot be determined.

IMPORTANT:

Determine user type BEFORE complaint theme.

Do NOT determine the user type from the theme.


============================================================
CUSTOMER THEMES
============================================================

Finding a provider / low response

Customer cannot obtain enough providers,
responses, quotes or availability.


Poor matching / location

Platform suggests unsuitable, irrelevant
or geographically inappropriate providers.


Provider quality / no-show

Poor work, bad service, incomplete work,
provider ghosting or provider no-show.


Provider verification / safety

Fraud, scam, identity, vetting or safety concern
about a provider.


Too many provider contacts / spam

Customer receives excessive calls,
messages or contacts from providers.


Pricing / quote transparency

Provider quotes/prices are unclear,
unexpected, excessive or misleading.


Review trust / moderation

Reviews appear fake, manipulated,
filtered, removed or unreliable.


Dispute resolution / protection

Platform fails to protect customer
or resolve a provider dispute.


============================================================
PROVIDER / BUSINESS THEMES
============================================================

Poor lead quality

Low-intent, unsuitable or unrealistic prospects.


Fake / invalid leads

Fake enquiries, invalid phone numbers,
invalid emails or non-genuine requests.


Unresponsive leads

Prospective customers do not respond
or stop responding.


Poor ROI / expensive leads

Leads, credits or advertising cost too much
relative to business generated.


Too much provider competition

Too many businesses compete
for the same lead/opportunity.


Billing / refunds

Platform billing, renewal, subscription,
credits, payment or refund problem.


============================================================
SHARED THEMES
============================================================

Customer support

Poor, unavailable or ineffective
PLATFORM customer support.


Account access / cancellation

Login, suspension, deletion,
account access or cancellation problem.


Privacy / unwanted contact

Spam, unwanted contact or
privacy/data concern.


Platform usability / technical issues

Website, app, filtering, profile,
technical or automation failure.


Misleading claims / transparency

Misleading platform claims,
unclear promises or deceptive representation.


Sales pressure / contract terms

Pushy sales or restrictive
contract/renewal terms.


Listing / profile accuracy

Incorrect, outdated, duplicate
or unauthorized listing/profile.


Other

Relevant complaint that genuinely
does not fit the frozen taxonomy.


============================================================
SIDE/THEME RULE
============================================================

CUSTOMERS may use:

Customer themes
+
Shared themes


SERVICE PROVIDERS / BUSINESSES may use:

Provider themes
+
Shared themes

Business owners may also use:

Review trust / moderation

when complaining about Yelp or another platform
filtering/removing/manipulating reviews.


============================================================
PRIMARY / SECONDARY
============================================================

Choose exactly ONE primary theme.

That is the CENTRAL complaint.

secondary_theme should normally be:

"None"

Only use a secondary theme if there is a clearly
separate second complaint.

Do NOT add themes merely because they are related.


============================================================
SEVERITY
============================================================

LOW

Minor inconvenience/friction.

Examples:

small usability problem
minor matching issue
limited provider choice
mild unwanted contact


MEDIUM

Meaningful failure or meaningful waste
of money/time.

Examples:

unresponsive leads
poor lead
fake lead
no-show
poor workmanship
refund difficulty
cancellation difficulty
support failure
misleading sales
platform malfunction


HIGH

Serious harm.

Examples:

fraud WITH actual financial loss
serious safety concern
substantial property damage
hundreds/thousands lost
continued unauthorized charges
debt/legal threats
major business loss/outage


IMPORTANT:

Do NOT classify High merely because:

- rating is 1 star
- reviewer is angry
- reviewer uses words like "scam"
- service was disappointing


============================================================
CORRECT EXAMPLES
============================================================

EXAMPLE 1

"I was looking for a window cleaner but Bark
suggested companies 80 miles away."

Customer
Poor matching / location
None
Low


EXAMPLE 2

"I buy Bark credits. Too many professionals
chase the same leads and the credits cost more
than the jobs I win."

Service Provider / Business
Poor ROI / expensive leads
Too much provider competition
Medium


EXAMPLE 3

"I am a supplier on Bark. My account filtering
has stopped working and leads aren't reaching me.
Support has not fixed it."

Service Provider / Business
Platform usability / technical issues
Customer support
Medium


EXAMPLE 4

"I hired a builder and lost £2,500 after the
platform said he had been vetted."

Customer
Provider verification / safety
None
High


EXAMPLE 5

"I paid for several leads but their phone
numbers were invalid and people said they never
requested the work."

Service Provider / Business
Fake / invalid leads
None
Medium


EXAMPLE 6

"I purchased credits and contacted 21 leads.
Nobody answered and Bark refused the credit refund."

Service Provider / Business
Unresponsive leads
Billing / refunds
Medium


Use the label spellings exactly.
""".strip()


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not FULL_REVIEW_FILE.exists():

        raise FileNotFoundError(
            f"Full review file not found:\n"
            f"{FULL_REVIEW_FILE}"
        )

    if not V3_LABEL_FILE.exists():

        raise FileNotFoundError(
            f"V3 labelled file not found:\n"
            f"{V3_LABEL_FILE}"
        )

    full = pd.read_csv(
        FULL_REVIEW_FILE,
        low_memory=False,
    )

    v3 = pd.read_csv(
        V3_LABEL_FILE,
        low_memory=False,
    )

    full["reviewId"] = (
        full["reviewId"]
        .astype(str)
    )

    v3["reviewId"] = (
        v3["reviewId"]
        .astype(str)
    )

    overlap = len(
        set(full["reviewId"])
        & set(v3["reviewId"])
    )

    print(
        f"\nFull negative reviews: "
        f"{len(full):,}"
    )

    print(
        f"Existing reviewed V3 labels: "
        f"{len(v3):,}"
    )

    print(
        f"V3 reviews found in full dataset: "
        f"{overlap:,}"
    )

    if overlap != len(v3):

        raise ValueError(
            "Not every reviewed V3 review "
            "is present in the full dataset."
        )

    return full, v3


# ============================================================
# COPY EXISTING 400 V3 LABELS
# ============================================================

def build_existing_v3_rows(
    full,
    v3,
):

    full_lookup = (
        full.set_index(
            "reviewId"
        )
    )

    rows = []

    for _, label in v3.iterrows():

        review_id = str(
            label["reviewId"]
        )

        original = (
            full_lookup.loc[
                review_id
            ]
        )

        secondary = (
            label.get(
                "secondary_complaint_theme",
                "",
            )
        )

        if pd.isna(secondary):
            secondary = ""

        secondary = (
            str(secondary).strip()
        )

        rows.append(
            {
                "reviewId":
                    review_id,

                "rating":
                    original.get(
                        "rating",
                        "",
                    ),

                "publishedDate":
                    original.get(
                        "publishedDate",
                        "",
                    ),

                "text":
                    original.get(
                        "text",
                        "",
                    ),

                "competitor":
                    original.get(
                        "competitor",
                        "",
                    ),

                "year":
                    original.get(
                        "year",
                        "",
                    ),

                "manual_relevance":
                    label.get(
                        "manual_relevance",
                        "",
                    ),

                "manual_user_type":
                    label.get(
                        "manual_user_type",
                        "",
                    ),

                "primary_complaint_theme":
                    label.get(
                        "primary_complaint_theme",
                        "",
                    ),

                "secondary_complaint_theme":
                    secondary,

                "manual_complaint_themes":
                    label.get(
                        "manual_complaint_themes",
                        "",
                    ),

                "manual_severity":
                    label.get(
                        "manual_severity",
                        "",
                    ),

                "annotation_method":
                    "V3 reviewed anchor label",

                "annotation_status":
                    "frozen_v3_label",
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# RESET AI TEST LABELS
# ============================================================

def reset_ai_labels(
    full,
    v3,
):

    frozen = (
        build_existing_v3_rows(
            full,
            v3,
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    frozen.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    if AUDIT_FILE.exists():
        AUDIT_FILE.unlink()

    print(
        f"\nReset output to "
        f"{len(frozen):,} frozen V3 labels."
    )


# ============================================================
# BUILD USER PROMPT
# ============================================================

def build_user_prompt(
    row,
    forced_user_type=None,
    retry_instruction=None,
):

    text = str(
        row.get(
            "text",
            "",
        )
    ).strip()

    side_instruction = ""

    if (
        forced_user_type
        == "Service Provider / Business"
    ):

        side_instruction = """
IMPORTANT:

Strong first-person evidence proves this reviewer is:

Service Provider / Business

You MUST return:

"user_type": "Service Provider / Business"

ALLOWED THEMES:

Poor lead quality
Fake / invalid leads
Unresponsive leads
Poor ROI / expensive leads
Too much provider competition
Billing / refunds
Customer support
Account access / cancellation
Privacy / unwanted contact
Platform usability / technical issues
Misleading claims / transparency
Sales pressure / contract terms
Listing / profile accuracy
Review trust / moderation
Other

DO NOT use customer-only themes.
"""

    elif (
        forced_user_type
        == "Customer"
    ):

        side_instruction = """
IMPORTANT:

Strong first-person evidence proves this reviewer is:

Customer

You MUST return:

"user_type": "Customer"

ALLOWED THEMES:

Finding a provider / low response
Poor matching / location
Provider quality / no-show
Provider verification / safety
Too many provider contacts / spam
Pricing / quote transparency
Review trust / moderation
Dispute resolution / protection
Customer support
Account access / cancellation
Privacy / unwanted contact
Platform usability / technical issues
Misleading claims / transparency
Sales pressure / contract terms
Listing / profile accuracy
Other

DO NOT use provider/business-only themes.
"""

    retry_text = ""

    if retry_instruction:

        retry_text = f"""
CORRECTION REQUIRED:

Your previous classification broke
the V3 rules:

{retry_instruction}

Correct the classification.
"""

    return f"""
Competitor:
{row.get("competitor", "")}

Rating:
{row.get("rating", "")}

REVIEW:

{text}


{side_instruction}


{retry_text}


Classify this review using the frozen V3 codebook.

Return JSON only.
""".strip()


# ============================================================
# VALIDATE MODEL OUTPUT
# ============================================================

def validate_prediction(
    prediction,
    forced_user_type=None,
):

    required = {
        "relevance",
        "user_type",
        "primary_theme",
        "secondary_theme",
        "severity",
    }

    if not isinstance(prediction, dict):
        raise ValueError(
            "Model output is not JSON."
        )

    missing = required - set(
        prediction.keys()
    )

    if missing:
        raise ValueError(
            f"Missing keys: {sorted(missing)}"
        )

    # --------------------------------------------------------
    # READ MODEL VALUES FIRST
    # --------------------------------------------------------

    relevance = str(
        prediction["relevance"]
    ).strip()

    user_type = str(
        prediction["user_type"]
    ).strip()

    primary = str(
        prediction["primary_theme"]
    ).strip()

    secondary = str(
        prediction["secondary_theme"]
    ).strip()

    severity = str(
        prediction["severity"]
    ).strip()


    # --------------------------------------------------------
    # NORMALIZE THEME CAPITALIZATION
    # --------------------------------------------------------

    theme_map = {
        theme.lower(): theme
        for theme in THEMES
    }

    theme_map["not applicable"] = "Not applicable"
    theme_map["none"] = "None"

    primary = theme_map.get(
        primary.lower(),
        primary,
    )

    secondary = theme_map.get(
        secondary.lower(),
        secondary,
    )


    # --------------------------------------------------------
    # NORMALIZE CAPITALIZATION
    # --------------------------------------------------------

    relevance_map = {
        "relevant": "Relevant",
        "exclude": "Exclude",
    }

    user_type_map = {
        "customer": "Customer",
        "service provider / business":
            "Service Provider / Business",
        "unknown": "Unknown",
        "not applicable": "Not applicable",
    }

    severity_map = {
        "low": "Low",
        "medium": "Medium",
        "high": "High",
        "not applicable": "Not applicable",
    }

    relevance = relevance_map.get(
        relevance.lower(),
        relevance,
    )

    user_type = user_type_map.get(
        user_type.lower(),
        user_type,
    )

    severity = severity_map.get(
        severity.lower(),
        severity,
    )


    # --------------------------------------------------------
    # BASIC VALIDATION
    # --------------------------------------------------------

    if relevance not in VALID_RELEVANCE:
        raise ValueError(
            f"Invalid relevance: {relevance}"
        )

    if user_type not in VALID_USER_TYPES:
        raise ValueError(
            f"Invalid user type: {user_type}"
        )

    if severity not in VALID_SEVERITIES:
        raise ValueError(
            f"Invalid severity: {severity}"
        )

    valid_primary = (
        set(THEMES)
        | {"Not applicable"}
    )

    valid_secondary = (
        set(THEMES)
        | {
            "None",
            "Not applicable",
            "",
        }
    )

    if primary not in valid_primary:
        raise ValueError(
            f"Invalid primary theme: {primary}"
        )

    if secondary not in valid_secondary:
        raise ValueError(
            f"Invalid secondary theme: {secondary}"
        )


    # --------------------------------------------------------
    # EXCLUDED REVIEWS
    # --------------------------------------------------------

    if relevance == "Exclude":

        return {
            "relevance": "Exclude",
            "user_type": "Not applicable",
            "primary_theme": "Not applicable",
            "secondary_theme": "Not applicable",
            "severity": "Not applicable",
        }


    # --------------------------------------------------------
    # RELEVANT REVIEW VALIDATION
    # --------------------------------------------------------

    if user_type == "Not applicable":
        raise ValueError(
            "Relevant review cannot have "
            "Not applicable user type."
        )

    if primary == "Not applicable":
        raise ValueError(
            "Relevant review cannot have "
            "Not applicable primary theme."
        )

    if severity == "Not applicable":
        raise ValueError(
            "Relevant review cannot have "
            "Not applicable severity."
        )

    if secondary == "Not applicable":
        secondary = "None"

    if secondary == primary:
        secondary = "None"


    # --------------------------------------------------------
    # STRONG SIDE SIGNAL OVERRIDES MODEL SIDE
    # --------------------------------------------------------

    if forced_user_type:
        user_type = forced_user_type


    # --------------------------------------------------------
    # USER TYPE CONTROLS ALLOWED THEMES
    # --------------------------------------------------------

    if user_type == "Customer":

        allowed_customer = (
            set(CUSTOMER_THEMES)
            | set(SHARED_THEMES)
        )

        if primary not in allowed_customer:
            raise ValueError(
                f"Customer cannot use "
                f"primary theme: {primary}"
            )

        if (
            secondary != "None"
            and secondary not in allowed_customer
        ):
            raise ValueError(
                f"Customer cannot use "
                f"secondary theme: {secondary}"
            )


    elif (
        user_type
        == "Service Provider / Business"
    ):

        allowed_provider = (
            set(PROVIDER_THEMES)
            | set(SHARED_THEMES)
            | {"Review trust / moderation"}
        )

        if primary not in allowed_provider:
            raise ValueError(
                f"Provider cannot use "
                f"primary theme: {primary}"
            )

        if (
            secondary != "None"
            and secondary not in allowed_provider
        ):
            raise ValueError(
                f"Provider cannot use "
                f"secondary theme: {secondary}"
            )


    return {
        "relevance": relevance,
        "user_type": user_type,
        "primary_theme": primary,
        "secondary_theme": secondary,
        "severity": severity,
    }


# ============================================================
# OLLAMA REQUEST
# ============================================================

def call_ollama(
    row,
    model,
):

    forced_user_type = (
        infer_strong_user_type(
            row.get(
                "text",
                "",
            )
        )
    )

    retry_instruction = None

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        payload = {

            "model":
                model,

            "keep_alive":
                "30m",

            "messages": [

                {
                    "role":
                        "system",

                    "content":
                        SYSTEM_PROMPT,
                },

                {
                    "role":
                        "user",

                    "content":
                        build_user_prompt(
                            row,
                            forced_user_type=(
                                forced_user_type
                            ),
                            retry_instruction=(
                                retry_instruction
                            ),
                        ),
                },
            ],

            "stream":
                False,

            "format":
                "json",

            "options": {

                "temperature":
                    0,

                "seed":
                    42,

                "num_ctx":
                    4096,

                "num_predict":
                    140,
            },
        }


        encoded = (
            json.dumps(
                payload
            )
            .encode(
                "utf-8"
            )
        )


        request = (
            urllib.request.Request(

                OLLAMA_URL,

                data=encoded,

                headers={
                    "Content-Type":
                        "application/json"
                },

                method="POST",
            )
        )


        try:

            with (
                urllib.request.urlopen(
                    request,
                    timeout=(
                        REQUEST_TIMEOUT_SECONDS
                    ),
                )
                as response
            ):

                body = (
                    json.loads(
                        response
                        .read()
                        .decode(
                            "utf-8"
                        )
                    )
                )


            prediction = (
                json.loads(
                    body[
                        "message"
                    ][
                        "content"
                    ]
                )
            )


            return (
                validate_prediction(
                    prediction,
                    forced_user_type=(
                        forced_user_type
                    ),
                )
            )


        except Exception as error:

            last_error = error

            retry_instruction = (
                str(error)
            )

            print(
                f"    Attempt "
                f"{attempt}/"
                f"{MAX_RETRIES} "
                f"failed: "
                f"{error}"
            )


            if (
                attempt
                < MAX_RETRIES
            ):

                print(
                    "    Retrying with "
                    "explicit correction..."
                )

                time.sleep(2)


    raise RuntimeError(
        f"Ollama failed: "
        f"{last_error}"
    )


# ============================================================
# CREATE OUTPUT ROW
# ============================================================

def build_output_row(
    original_row,
    prediction,
    model,
):

    relevance = prediction["relevance"]
    user_type = prediction["user_type"]
    primary = prediction["primary_theme"]
    secondary = prediction["secondary_theme"]
    severity = prediction["severity"]

    # Read the review text BEFORE using it below
    text = str(
        original_row.get(
            "text",
            "",
        )
    )

    # ========================================================
    # HIGH-SEVERITY SAFEGUARD
    # ========================================================

    if severity == "High":

        text_lower = text.lower()

        high_impact_signals = [
            "thousand",
            "hundreds",
            "lost money",
            "financial loss",
            "stolen",
            "serious damage",
            "unsafe",
            "safety",
            "debt",
            "court",
            "legal",
            "unauthorised charge",
            "unauthorized charge",
            "charged repeatedly",
        ]

        has_high_impact = any(
            signal in text_lower
            for signal in high_impact_signals
        )

        if not has_high_impact:
            severity = "Medium"


    # ========================================================
    # SECONDARY THEME
    # ========================================================

    secondary_for_file = ""

    if secondary not in {
        "None",
        "Not applicable",
        "",
    }:
        secondary_for_file = secondary


    # ========================================================
    # COMBINED THEMES
    # ========================================================

    themes = ""

    if relevance == "Relevant":

        themes = primary

        if secondary_for_file:
            themes += (
                " | "
                + secondary_for_file
            )


    # ========================================================
    # OUTPUT ROW
    # ========================================================

    return {

        "reviewId":
            str(
                original_row["reviewId"]
            ),

        "rating":
            original_row.get(
                "rating",
                "",
            ),

        "publishedDate":
            original_row.get(
                "publishedDate",
                "",
            ),

        "text":
            text,

        "competitor":
            original_row.get(
                "competitor",
                "",
            ),

        "year":
            original_row.get(
                "year",
                "",
            ),

        "manual_relevance":
            relevance,

        "manual_user_type":
            user_type,

        "primary_complaint_theme":
            primary,

        "secondary_complaint_theme":
            secondary_for_file,

        "manual_complaint_themes":
            themes,

        "manual_severity":
            severity,

        "annotation_method":
            (
                f"Ollama {model} "
                f"+ frozen V3 codebook "
                f"+ deterministic side check"
            ),

        "annotation_status":
            "ai_label_needs_audit",
    }


# ============================================================
# LOAD / SAVE OUTPUT
# ============================================================

def load_existing_output():

    if not OUTPUT_FILE.exists():

        return pd.DataFrame()


    existing = pd.read_csv(
        OUTPUT_FILE,
        low_memory=False,
    )


    existing[
        "reviewId"
    ] = (
        existing[
            "reviewId"
        ]
        .astype(str)
    )


    return existing


def save_output(
    dataframe,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
    )


def ensure_output_initialized(
    full,
    v3,
):

    existing = (
        load_existing_output()
    )


    if not existing.empty:

        frozen_count = (

            existing[
                "annotation_status"
            ]
            .astype(str)
            .eq(
                "frozen_v3_label"
            )
            .sum()
        )


        print(
            f"Existing output found: "
            f"{len(existing):,} rows "
            f"({frozen_count:,} "
            f"frozen V3 labels)"
        )


        return existing


    frozen = (
        build_existing_v3_rows(
            full,
            v3,
        )
    )


    save_output(
        frozen
    )


    print(
        f"\nInitialized with "
        f"{len(frozen):,} "
        f"reviewed V3 labels."
    )


    return frozen


# ============================================================
# AUDIT SAMPLE
# ============================================================

def create_audit_sample(
    final_df,
    sample_size=100,
):

    ai_rows = (

        final_df[

            final_df[
                "annotation_status"
            ]
            .astype(str)
            .eq(
                "ai_label_needs_audit"
            )
        ]
        .copy()
    )


    if ai_rows.empty:

        return


    samples = []


    competitors = [
        "Bark",
        "Checkatrade",
        "Yell",
        "Yelp",
    ]


    per_competitor = max(
        1,
        sample_size
        // len(
            competitors
        ),
    )


    for competitor in competitors:

        subset = (

            ai_rows[

                ai_rows[
                    "competitor"
                ]
                == competitor
            ]
        )


        if subset.empty:

            continue


        samples.append(

            subset.sample(

                n=min(
                    per_competitor,
                    len(subset),
                ),

                random_state=42,
            )
        )


    if not samples:

        return


    audit = (

        pd.concat(
            samples,
            ignore_index=True,
        )
        .head(
            sample_size
        )
    )


    audit[
        "audit_relevance"
    ] = ""


    audit[
        "audit_user_type"
    ] = ""


    audit[
        "audit_primary_theme"
    ] = ""


    audit[
        "audit_secondary_theme"
    ] = ""


    audit[
        "audit_severity"
    ] = ""


    audit[
        "audit_notes"
    ] = ""


    audit.to_csv(
        AUDIT_FILE,
        index=False,
    )


# ============================================================
# CLASSIFY REVIEWS
# ============================================================

def classify_reviews(
    full,
    v3,
    model,
    limit=None,
):

    existing = (
        ensure_output_initialized(
            full,
            v3,
        )
    )


    completed_ids = set(

        existing[
            "reviewId"
        ]
        .astype(str)
    )


    remaining = (

        full[

            ~full[
                "reviewId"
            ]
            .astype(str)
            .isin(
                completed_ids
            )
        ]
        .copy()
    )


    total_unlabelled = (
        len(remaining)
    )


    if limit is not None:

        remaining = (
            remaining.head(
                limit
            )
        )


    print(
        f"\nReviews still "
        f"unlabelled overall: "
        f"{total_unlabelled:,}"
    )


    print(
        f"Reviews to process "
        f"in this run: "
        f"{len(remaining):,}"
    )


    print(
        f"Model: {model}"
    )


    if remaining.empty:

        final_df = (
            load_existing_output()
        )

        create_audit_sample(
            final_df
        )

        return final_df


    run_start = (
        time.perf_counter()
    )


    completed_this_run = 0

    failed_this_run = 0


    for run_number, (_, row) in enumerate(

        remaining.iterrows(),
        start=1,
    ):

        review_start = (
            time.perf_counter()
        )


        side_hint = (
            infer_strong_user_type(
                row.get(
                    "text",
                    "",
                )
            )
        )


        print(
            f"\nReview "
            f"{run_number}/"
            f"{len(remaining)} "
            f"| {row['competitor']} "
            f"| rating "
            f"{row.get('rating', '')}"
        )


        if side_hint:

            print(
                f"    Strong side signal: "
                f"{side_hint}"
            )


        try:

            prediction = (
                call_ollama(
                    row,
                    model,
                )
            )


        except Exception as error:

            failed_this_run += 1

            print(
                f"    SKIPPED: "
                f"{error}"
            )

            print(
                "    Moving to next review. "
                "This review remains "
                "unlabelled for retry later."
            )

            continue


        output_row = (
            build_output_row(
                row,
                prediction,
                model,
            )
        )


        current = (
            load_existing_output()
        )


        combined = (
            pd.concat(
                [
                    current,
                    pd.DataFrame(
                        [
                            output_row
                        ]
                    ),
                ],
                ignore_index=True,
            )
        )


        combined = (

            combined
            .drop_duplicates(
                subset=[
                    "reviewId"
                ],
                keep="last",
            )
        )


        save_output(
            combined
        )


        completed_this_run += 1


        review_seconds = (

            time.perf_counter()
            - review_start
        )


        run_seconds = (

            time.perf_counter()
            - run_start
        )


        average_seconds = (

            run_seconds
            / max(
                completed_this_run,
                1,
            )
        )


        current_remaining = (

            total_unlabelled
            - completed_this_run
        )


        print(
            f"    "
            f"{output_row['manual_user_type']} "
            f"| "
            f"{output_row['primary_complaint_theme']} "
            f"| "
            f"{output_row['manual_severity']}"
        )


        if (
            output_row[
                "secondary_complaint_theme"
            ]
        ):

            print(
                f"    Secondary: "
                f"{output_row['secondary_complaint_theme']}"
            )


        print(
            f"    Time: "
            f"{review_seconds:.1f}s"
        )


        print(
            f"    Running average: "
            f"{average_seconds:.1f}"
            f"s/review"
        )


        print(
            f"    Saved progress: "
            f"{len(combined):,}/"
            f"{len(full):,}"
        )


        if current_remaining > 0:

            eta_hours = (

                average_seconds
                * current_remaining
                / 3600
            )


            print(
                f"    Rough ETA: "
                f"{eta_hours:.1f} hours"
            )


    final_df = (
        load_existing_output()
    )


    create_audit_sample(
        final_df
    )


    print(
        f"\nCompleted this run: "
        f"{completed_this_run}"
    )


    print(
        f"Skipped/failed this run: "
        f"{failed_this_run}"
    )


    return final_df


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    final_df,
    total_reviews,
):

    print(
        "\n"
        + "=" * 72
    )


    print(
        "FULL V3 LABELLING STATUS"
    )


    print(
        "=" * 72
    )


    print(
        f"Reviews labelled: "
        f"{len(final_df):,}/"
        f"{total_reviews:,}"
    )


    print(
        "\nAnnotation status:"
    )


    print(
        final_df[
            "annotation_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )


    print(
        "\nOutput:"
    )


    print(
        OUTPUT_FILE
    )


    if AUDIT_FILE.exists():

        print(
            "\nAudit sample:"
        )

        print(
            AUDIT_FILE
        )


# ============================================================
# ARGUMENTS
# ============================================================

def get_arguments():

    parser = (
        argparse.ArgumentParser(

            description=(

                "Apply frozen V3 "
                "complaint codebook "
                "using constrained "
                "Qwen classification."
            )
        )
    )


    parser.add_argument(

        "--limit",

        type=int,

        default=None,

        help=(
            "Only classify this many "
            "NEW reviews."
        ),
    )


    parser.add_argument(

        "--model",

        type=str,

        default=MODEL,

        help=(
            f"Ollama model. "
            f"Default: {MODEL}"
        ),
    )


    parser.add_argument(

        "--reset-ai",

        action="store_true",

        help=(

            "Remove AI-generated labels "
            "but preserve the 400 "
            "reviewed V3 labels."
        ),
    )


    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = (
        get_arguments()
    )


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    full, v3 = (
        load_data()
    )


    if args.reset_ai:

        reset_ai_labels(
            full,
            v3,
        )


    final_df = (
        classify_reviews(

            full,
            v3,

            model=args.model,

            limit=args.limit,
        )
    )


    print_summary(
        final_df,
        len(full),
    )


if __name__ == "__main__":
    main()
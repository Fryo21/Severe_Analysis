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

REPAIR_QUEUE_FILE = (
    PROJECT_ROOT
    / "processed-data"
    / "competitor-research"
    / "complaint-analysis-final"
    / "repair-queue"
    / "repair_queue.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed-data"
    / "competitor-research"
    / "complaint-analysis-final"
    / "repair-queue"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "repair_predictions.csv"
)


# ============================================================
# OLLAMA CONFIG
# ============================================================

MODEL = "qwen3:4b-instruct"

OLLAMA_URL = "http://localhost:11434/api/chat"

MAX_RETRIES = 3

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
# CONSERVATIVE USER-TYPE SIGNALS
# ============================================================

PROVIDER_SIGNAL_PATTERNS = [
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

YELL_BUSINESS_CONTEXT_PATTERNS = [
    r"\bmy business\b",
    r"\bour business\b",
    r"\bmy company\b",
    r"\bour company\b",
    r"\bsmall independent business\b",
]

YELL_COMMERCIAL_PLATFORM_PATTERNS = [
    r"\badvertis(?:e|ed|ing|ement)\b",
    r"\bmarketing\b",
    r"\bwebsite\b",
    r"\bdomain\b",
    r"\bsubscription\b",
    r"\bcontract\b",
    r"\bleads?\b",
    r"\baccount manager\b",
]


def pattern_hits(text, patterns):

    text = str(text).lower()

    return sum(
        bool(
            re.search(
                pattern,
                text,
            )
        )
        for pattern in patterns
    )


def infer_strong_user_type(
    text,
    competitor,
):

    text = str(text)

    provider_hits = pattern_hits(
        text,
        PROVIDER_SIGNAL_PATTERNS,
    )

    customer_hits = pattern_hits(
        text,
        CUSTOMER_SIGNAL_PATTERNS,
    )

    # Audit-derived Yell rule:
    # Business ownership by itself is NOT enough.
    # It must occur with Yell advertising/website/contract/lead context.
    if str(competitor).strip() == "Yell":

        business_hits = pattern_hits(
            text,
            YELL_BUSINESS_CONTEXT_PATTERNS,
        )

        commercial_hits = pattern_hits(
            text,
            YELL_COMMERCIAL_PLATFORM_PATTERNS,
        )

        if (
            business_hits > 0
            and commercial_hits > 0
        ):
            provider_hits += 2

    if (
        provider_hits > 0
        and customer_hits == 0
    ):
        return "Service Provider / Business"

    if (
        customer_hits > 0
        and provider_hits == 0
    ):
        return "Customer"

    return None


# ============================================================
# REPAIR REASON GUIDANCE
# ============================================================

REPAIR_GUIDANCE = {

    "large_explicit_money_impact_not_high":
        (
            "Check severity carefully. An explicit large financial "
            "loss may justify High, but only when the money was "
            "actually lost, wasted, wrongly charged, or tied to "
            "serious harm. A price mentioned without serious loss "
            "does not automatically make the review High."
        ),

    "large_money_words_not_high":
        (
            "Check whether words such as hundreds/thousands describe "
            "actual financial loss. Do not treat unrelated uses such "
            "as distance or general scale as financial harm."
        ),

    "ai_exclude_recheck":
        (
            "Recheck relevance from scratch. Keep Exclude when the "
            "review is only about an unrelated third-party business "
            "or service and does not meaningfully complain about the "
            "marketplace/platform."
        ),

    "yelp_relevance_without_platform_reference":
        (
            "For Yelp, distinguish a complaint ABOUT YELP from a "
            "review merely hosted on Yelp about another company. "
            "If the complaint is only about the restaurant, cleaner, "
            "roofer, car company, salon, etc., return Exclude."
        ),

    "spam_theme_may_be_low_response":
        (
            "Distinguish excessive provider contact/spam from lack of "
            "responses. If the customer says nobody or too few "
            "providers replied, use Finding a provider / low response."
        ),

    "customer_label_with_lead_roi_language":
        (
            "Recheck marketplace side. Someone paying for leads, "
            "credits, advertising, or trying to win jobs/customers "
            "is a Service Provider / Business, even if they call "
            "themselves a customer of the platform."
        ),

    "fake_invalid_leads_without_lead_evidence":
        (
            "Use Fake / invalid leads only for fake enquiries, "
            "invalid contact details, or non-genuine requests. "
            "Do not use it for poor ROI, no jobs, misleading sales, "
            "listing errors, or general dissatisfaction."
        ),

    "yell_customer_with_business_advertising_signals":
        (
            "A business paying Yell for advertising, websites, "
            "subscriptions, contracts, or lead generation is a "
            "Service Provider / Business under this codebook."
        ),

    "pricing_theme_may_be_technical_issue":
        (
            "Recheck whether the complaint is actually about price/"
            "quotes. Problems attaching photos, uploading, forms, "
            "email acceptance, app/site behaviour, or missing "
            "features are Platform usability / technical issues."
        ),

    "unresponsive_leads_may_be_support":
        (
            "Use Unresponsive leads when prospects/leads fail to "
            "reply. Use Customer support when the PLATFORM support "
            "team, salesperson, or account manager fails to respond."
        ),

    "side_conflict_customer_vs_provider_evidence":
        (
            "Re-evaluate user type from the reviewer's marketplace "
            "role, not their occupation or the complaint theme."
        ),

    "side_conflict_provider_vs_customer_evidence":
        (
            "Re-evaluate user type from the reviewer's marketplace "
            "role, not their occupation or the complaint theme."
        ),

    "unknown_user_type":
        (
            "Try to resolve the marketplace side from context. Keep "
            "Unknown only if it genuinely cannot be determined."
        ),

    "other_theme_recheck":
        (
            "Try every frozen V3 theme before using Other. Other "
            "should be rare."
        ),

    "previous_full_run_skipped_or_unlabelled":
        (
            "This review was not successfully labelled previously. "
            "Classify it from scratch using the frozen V3 codebook."
        ),
}


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are repairing labels for negative reviews from Bark,
Checkatrade, Yell and Yelp.

Use ONLY the frozen V3 taxonomy below.

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
The complaint concerns the marketplace/platform, providers,
leads, advertising, listings, reviews, billing, support,
accounts, contracts, quotes, matching or disputes.

Exclude:
The review is solely about an unrelated third-party company,
restaurant, product, hotel, shop, cleaner, roofer, salon,
car company or other service and contains no meaningful
complaint about the marketplace/platform itself.

IMPORTANT FOR YELP:
A negative review hosted on Yelp is NOT automatically a Yelp
platform complaint. If the reviewer only complains about the
third-party business being reviewed, return Exclude.

If Exclude:
user_type = Not applicable
primary_theme = Not applicable
secondary_theme = Not applicable
severity = Not applicable

============================================================
USER TYPE
============================================================

Customer:
Uses the marketplace to find, compare, request quotes from,
or hire a provider.

Service Provider / Business:
Uses the marketplace to advertise a business, buy leads or
credits, obtain customers/jobs, manage a business profile,
pay advertising fees, or discuss lead/advertising ROI.

IMPORTANT:
A business paying Yell for advertising, websites,
subscriptions, contracts, or lead generation is a
Service Provider / Business.

Business ownership by itself does NOT prove provider side.
Determine the reviewer's marketplace role.

Unknown:
Only when the side genuinely cannot be determined.

============================================================
CUSTOMER THEMES
============================================================

Finding a provider / low response
Poor matching / location
Provider quality / no-show
Provider verification / safety
Too many provider contacts / spam
Pricing / quote transparency
Review trust / moderation
Dispute resolution / protection

============================================================
PROVIDER / BUSINESS THEMES
============================================================

Poor lead quality
Fake / invalid leads
Unresponsive leads
Poor ROI / expensive leads
Too much provider competition
Billing / refunds

============================================================
SHARED THEMES
============================================================

Customer support
Account access / cancellation
Privacy / unwanted contact
Platform usability / technical issues
Misleading claims / transparency
Sales pressure / contract terms
Listing / profile accuracy
Other

Business owners may also use Review trust / moderation
when complaining about review filtering/removal/manipulation.

============================================================
THEME DISTINCTIONS
============================================================

Finding a provider / low response:
Customer receives no or too few provider replies.

Too many provider contacts / spam:
Customer receives excessive provider calls/messages/contact.
Do NOT use this for "no response".

Poor lead quality:
Provider receives low-intent, unsuitable, unrealistic prospects.

Fake / invalid leads:
Provider receives fake enquiries, invalid phone/email/contact
details, or demonstrably non-genuine requests.
Do NOT use this merely because advertising produced no jobs.

Unresponsive leads:
Provider contacts genuine-looking prospects who do not reply.

Poor ROI / expensive leads:
Provider pays for leads/credits/advertising but gets poor
business return.

Customer support:
PLATFORM support, salesperson, or account manager is poor,
unavailable, or unresponsive.

Platform usability / technical issues:
Website/app/forms/upload/filter/profile/automation failure.

Pricing / quote transparency:
Customer-facing provider quote/price is unclear, unexpected,
excessive or misleading.

============================================================
PRIMARY / SECONDARY
============================================================

Choose exactly ONE primary theme: the central complaint.

secondary_theme should normally be "None".

Only add a secondary theme for a clearly separate second
complaint. Do not add related themes automatically.

============================================================
SEVERITY
============================================================

Low:
Minor inconvenience/friction.

Medium:
Meaningful failure or meaningful waste of time/money,
including ordinary poor leads, no-show, support failure,
refund/cancellation trouble, misleading sales or platform
malfunction.

High:
Serious harm such as fraud WITH actual financial loss,
serious safety concern, substantial damage,
hundreds/thousands actually lost, continued unauthorized
charges, debt/legal threats, or major business loss/outage.

Do NOT use High merely because:
- rating is 1 star
- reviewer is angry
- the word "scam" appears
- a currency amount is mentioned

Severity depends on actual consequence.

Use the exact label spellings above.
""".strip()


# ============================================================
# LOAD / SAVE
# ============================================================

def load_queue():

    if not REPAIR_QUEUE_FILE.exists():

        raise FileNotFoundError(
            "Repair queue not found:\n"
            f"{REPAIR_QUEUE_FILE}"
        )

    queue = pd.read_csv(
        REPAIR_QUEUE_FILE,
        low_memory=False,
    )

    queue["reviewId"] = (
        queue["reviewId"]
        .astype(str)
    )

    return queue


def load_existing_output():

    if not OUTPUT_FILE.exists():
        return pd.DataFrame()

    existing = pd.read_csv(
        OUTPUT_FILE,
        low_memory=False,
    )

    existing["reviewId"] = (
        existing["reviewId"]
        .astype(str)
    )

    return existing


def save_output(dataframe):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
    )


# ============================================================
# NORMALIZATION / VALIDATION
# ============================================================

def canonicalize_theme(value):

    value = str(value).strip()

    lookup = {
        theme.casefold(): theme
        for theme in THEMES
    }

    lookup.update(
        {
            "none": "None",
            "not applicable": "Not applicable",

            # Common near-miss spellings seen during the first run.
            "provider no-show": "Provider quality / no-show",
            "provider no show": "Provider quality / no-show",
            "provider quality/no-show": "Provider quality / no-show",
            "poor matching/location": "Poor matching / location",
            "poor roi/expensive leads": "Poor ROI / expensive leads",
            "fake/invalid leads": "Fake / invalid leads",
        }
    )

    return lookup.get(
        value.casefold(),
        value,
    )


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

    if not isinstance(
        prediction,
        dict,
    ):
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

    relevance_raw = str(
        prediction["relevance"]
    ).strip()

    user_type_raw = str(
        prediction["user_type"]
    ).strip()

    primary_raw = str(
        prediction["primary_theme"]
    ).strip()

    secondary_raw = str(
        prediction["secondary_theme"]
    ).strip()

    severity_raw = str(
        prediction["severity"]
    ).strip()


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
        relevance_raw.casefold(),
        relevance_raw,
    )

    user_type = user_type_map.get(
        user_type_raw.casefold(),
        user_type_raw,
    )

    severity = severity_map.get(
        severity_raw.casefold(),
        severity_raw,
    )

    primary = canonicalize_theme(
        primary_raw
    )

    secondary = canonicalize_theme(
        secondary_raw
    )


    # If Qwen returns a fully Not-applicable classification,
    # interpret it as Exclude rather than wasting another retry.
    if (
        relevance.casefold()
        == "not applicable"
        and user_type
        == "Not applicable"
        and primary
        == "Not applicable"
        and severity
        == "Not applicable"
    ):
        relevance = "Exclude"
        secondary = "Not applicable"


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


    if relevance == "Exclude":

        return {
            "relevance": "Exclude",
            "user_type": "Not applicable",
            "primary_theme": "Not applicable",
            "secondary_theme": "Not applicable",
            "severity": "Not applicable",
        }


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


    if forced_user_type:
        user_type = forced_user_type


    if user_type == "Customer":

        allowed = (
            set(CUSTOMER_THEMES)
            | set(SHARED_THEMES)
        )

        if primary not in allowed:
            raise ValueError(
                "Customer cannot use "
                f"primary theme: {primary}"
            )

        if (
            secondary != "None"
            and secondary not in allowed
        ):
            raise ValueError(
                "Customer cannot use "
                f"secondary theme: {secondary}"
            )


    elif (
        user_type
        == "Service Provider / Business"
    ):

        allowed = (
            set(PROVIDER_THEMES)
            | set(SHARED_THEMES)
            | {
                "Review trust / moderation",
            }
        )

        if primary not in allowed:
            raise ValueError(
                "Provider cannot use "
                f"primary theme: {primary}"
            )

        if (
            secondary != "None"
            and secondary not in allowed
        ):
            raise ValueError(
                "Provider cannot use "
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
# PROMPT
# ============================================================

def build_reason_guidance(
    repair_reasons,
):

    reasons = [
        reason.strip()
        for reason in str(
            repair_reasons
        ).split("|")
        if reason.strip()
    ]

    guidance = []

    for reason in reasons:

        message = (
            REPAIR_GUIDANCE.get(
                reason
            )
        )

        if message:
            guidance.append(
                f"- {message}"
            )

    if not guidance:
        return (
            "- Re-evaluate the label carefully "
            "using the frozen V3 codebook."
        )

    return "\n".join(
        guidance
    )


def old_label_block(row):

    queue_type = str(
        row.get(
            "repair_queue_type",
            "",
        )
    ).strip()

    if queue_type != "suspect_ai_label":
        return (
            "There is no reliable previous AI label. "
            "Classify this review from scratch."
        )

    return f"""
PREVIOUS AI LABEL:

relevance:
{row.get("manual_relevance", "")}

user_type:
{row.get("manual_user_type", "")}

primary_theme:
{row.get("primary_complaint_theme", "")}

secondary_theme:
{row.get("secondary_complaint_theme", "")}

severity:
{row.get("manual_severity", "")}

The previous label is NOT ground truth.
Keep it if it is correct.
Change only what is actually wrong.
""".strip()


def build_user_prompt(
    row,
    forced_user_type=None,
    retry_instruction=None,
):

    competitor = str(
        row.get(
            "competitor",
            "",
        )
    ).strip()

    text = str(
        row.get(
            "text",
            "",
        )
    ).strip()

    reasons = str(
        row.get(
            "repair_reasons",
            "",
        )
    ).strip()

    guidance = (
        build_reason_guidance(
            reasons
        )
    )

    forced_block = ""

    if forced_user_type:

        forced_block = f"""
STRONG MARKETPLACE-SIDE EVIDENCE:

The review contains strong evidence that the reviewer is:

{forced_user_type}

Use that user type unless the review should be Exclude.
""".strip()


    retry_block = ""

    if retry_instruction:

        retry_block = f"""
CORRECTION REQUIRED:

Your previous attempt violated the V3 taxonomy:

{retry_instruction}

Return a corrected classification using exact V3 labels.
""".strip()


    return f"""
COMPETITOR:
{competitor}

RATING:
{row.get("rating", "")}

REPAIR QUEUE REASONS:
{reasons}

WHY THIS REVIEW WAS FLAGGED:
{guidance}

{old_label_block(row)}

{forced_block}

REVIEW:
{text}

{retry_block}

TASK:

Re-evaluate the review from scratch.

The repair-queue reason is only a warning that a mistake
MAY exist. It is not proof that the previous label is wrong.

Return JSON only using the frozen V3 labels.
""".strip()


# ============================================================
# OLLAMA
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
            ),
            row.get(
                "competitor",
                "",
            ),
        )
    )

    retry_instruction = None
    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        payload = {
            "model": model,
            "keep_alive": "30m",

            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": build_user_prompt(
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

            "stream": False,
            "format": "json",

            "options": {
                "temperature": 0,
                "seed": 42,
                "num_ctx": 4096,
                "num_predict": 140,
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
                        "application/json",
                },
                method="POST",
            )
        )


        try:

            with urllib.request.urlopen(
                request,
                timeout=(
                    REQUEST_TIMEOUT_SECONDS
                ),
            ) as response:

                body = json.loads(
                    response
                    .read()
                    .decode(
                        "utf-8"
                    )
                )


            prediction = json.loads(
                body[
                    "message"
                ][
                    "content"
                ]
            )


            return validate_prediction(
                prediction,
                forced_user_type=(
                    forced_user_type
                ),
            )


        except Exception as error:

            last_error = error

            retry_instruction = str(
                error
            )

            print(
                f"    Attempt "
                f"{attempt}/"
                f"{MAX_RETRIES} "
                f"failed: "
                f"{error}"
            )

            if attempt < MAX_RETRIES:

                print(
                    "    Retrying with "
                    "explicit correction..."
                )

                time.sleep(2)


    raise RuntimeError(
        "Ollama repair failed: "
        f"{last_error}"
    )


# ============================================================
# BUILD REPAIR RESULT
# ============================================================

def clean_optional(value):

    if pd.isna(value):
        return ""

    return str(
        value
    ).strip()


def build_result_row(
    row,
    prediction,
    model,
):

    old_relevance = clean_optional(
        row.get(
            "manual_relevance",
            "",
        )
    )

    old_user_type = clean_optional(
        row.get(
            "manual_user_type",
            "",
        )
    )

    old_primary = clean_optional(
        row.get(
            "primary_complaint_theme",
            "",
        )
    )

    old_secondary = clean_optional(
        row.get(
            "secondary_complaint_theme",
            "",
        )
    )

    old_severity = clean_optional(
        row.get(
            "manual_severity",
            "",
        )
    )


    new_secondary = (
        prediction[
            "secondary_theme"
        ]
    )

    if new_secondary in {
        "None",
        "Not applicable",
        "",
    }:
        new_secondary_for_file = (
            ""
            if prediction[
                "relevance"
            ] == "Relevant"
            else "Not applicable"
        )
    else:
        new_secondary_for_file = (
            new_secondary
        )


    new_values = {
        "relevance":
            prediction[
                "relevance"
            ],

        "user_type":
            prediction[
                "user_type"
            ],

        "primary":
            prediction[
                "primary_theme"
            ],

        "secondary":
            new_secondary_for_file,

        "severity":
            prediction[
                "severity"
            ],
    }


    old_values = {
        "relevance":
            old_relevance,

        "user_type":
            old_user_type,

        "primary":
            old_primary,

        "secondary":
            old_secondary,

        "severity":
            old_severity,
    }


    queue_type = clean_optional(
        row.get(
            "repair_queue_type",
            "",
        )
    )


    if queue_type == "unlabelled_retry":

        changed = "new_label"

    else:

        changed = (
            "changed"
            if old_values != new_values
            else "unchanged"
        )


    return {
        "reviewId":
            str(
                row[
                    "reviewId"
                ]
            ),

        "competitor":
            row.get(
                "competitor",
                "",
            ),

        "rating":
            row.get(
                "rating",
                "",
            ),

        "publishedDate":
            row.get(
                "publishedDate",
                "",
            ),

        "text":
            row.get(
                "text",
                "",
            ),

        "repair_queue_type":
            queue_type,

        "repair_reasons":
            row.get(
                "repair_reasons",
                "",
            ),

        "old_relevance":
            old_relevance,

        "old_user_type":
            old_user_type,

        "old_primary_theme":
            old_primary,

        "old_secondary_theme":
            old_secondary,

        "old_severity":
            old_severity,

        "repaired_relevance":
            new_values[
                "relevance"
            ],

        "repaired_user_type":
            new_values[
                "user_type"
            ],

        "repaired_primary_theme":
            new_values[
                "primary"
            ],

        "repaired_secondary_theme":
            new_values[
                "secondary"
            ],

        "repaired_severity":
            new_values[
                "severity"
            ],

        "repair_change_status":
            changed,

        "repair_model":
            model,

        "repair_status":
            "repair_needs_audit",
    }


# ============================================================
# PROCESS QUEUE
# ============================================================

def process_queue(
    queue,
    model,
    limit=None,
):

    existing = (
        load_existing_output()
    )

    completed_ids = set()

    if not existing.empty:

        completed_ids = set(
            existing[
                "reviewId"
            ]
            .astype(str)
        )

        print(
            f"\nExisting repair output: "
            f"{len(existing):,}"
        )


    remaining = (
        queue[
            ~queue[
                "reviewId"
            ]
            .astype(str)
            .isin(
                completed_ids
            )
        ]
        .copy()
    )


    total_remaining = len(
        remaining
    )


    if limit is not None:

        remaining = (
            remaining.head(
                limit
            )
        )


    print(
        f"\nRepair queue: "
        f"{len(queue):,}"
    )

    print(
        f"Still unrepaired overall: "
        f"{total_remaining:,}"
    )

    print(
        f"Processing this run: "
        f"{len(remaining):,}"
    )

    print(
        f"Model: {model}"
    )


    if remaining.empty:

        return existing


    run_start = (
        time.perf_counter()
    )

    completed_this_run = 0
    failed_this_run = 0


    for number, (_, row) in enumerate(
        remaining.iterrows(),
        start=1,
    ):

        review_start = (
            time.perf_counter()
        )


        print(
            f"\nRepair "
            f"{number}/"
            f"{len(remaining)} "
            f"| "
            f"{row.get('competitor', '')} "
            f"| rating "
            f"{row.get('rating', '')}"
        )

        print(
            f"    Reason: "
            f"{row.get('repair_reasons', '')}"
        )


        side_hint = (
            infer_strong_user_type(
                row.get(
                    "text",
                    "",
                ),
                row.get(
                    "competitor",
                    "",
                ),
            )
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

            continue


        result_row = (
            build_result_row(
                row,
                prediction,
                model,
            )
        )


        current = (
            load_existing_output()
        )

        combined = pd.concat(
            [
                current,
                pd.DataFrame(
                    [result_row]
                ),
            ],
            ignore_index=True,
        )

        combined = (
            combined
            .drop_duplicates(
                subset=[
                    "reviewId",
                ],
                keep="last",
            )
        )

        save_output(
            combined
        )

        completed_this_run += 1


        elapsed = (
            time.perf_counter()
            - review_start
        )

        run_elapsed = (
            time.perf_counter()
            - run_start
        )

        average = (
            run_elapsed
            / max(
                completed_this_run,
                1,
            )
        )

        remaining_after = (
            total_remaining
            - completed_this_run
        )


        print(
            "    "
            f"{result_row['repaired_relevance']} "
            "| "
            f"{result_row['repaired_user_type']} "
            "| "
            f"{result_row['repaired_primary_theme']} "
            "| "
            f"{result_row['repaired_severity']}"
        )

        if result_row[
            "repaired_secondary_theme"
        ] not in {
            "",
            "Not applicable",
        }:

            print(
                f"    Secondary: "
                f"{result_row['repaired_secondary_theme']}"
            )


        print(
            f"    Change: "
            f"{result_row['repair_change_status']}"
        )

        print(
            f"    Time: "
            f"{elapsed:.1f}s"
        )

        print(
            f"    Saved repair progress: "
            f"{len(combined):,}/"
            f"{len(queue):,}"
        )


        if remaining_after > 0:

            eta_hours = (
                average
                * remaining_after
                / 3600
            )

            print(
                f"    Rough repair ETA: "
                f"{eta_hours:.1f} hours"
            )


    final = (
        load_existing_output()
    )


    print(
        f"\nCompleted this run: "
        f"{completed_this_run}"
    )

    print(
        f"Skipped/failed this run: "
        f"{failed_this_run}"
    )

    return final


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    final,
    queue_size,
):

    print(
        "\n"
        + "=" * 72
    )

    print(
        "TARGETED V3 REPAIR STATUS"
    )

    print(
        "=" * 72
    )

    print(
        f"Repair predictions saved: "
        f"{len(final):,}/"
        f"{queue_size:,}"
    )


    if not final.empty:

        print(
            "\nChange status:"
        )

        print(
            final[
                "repair_change_status"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )


        print(
            "\nBy competitor:"
        )

        print(
            final[
                "competitor"
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

    print(
        "\nThe original "
        "full_reviews_ai_labeled_v3.csv "
        "was NOT modified."
    )


# ============================================================
# ARGUMENTS
# ============================================================

def get_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Targeted V3 reclassification of "
            "the repair queue."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Only process this many NEW "
            "repair-queue reviews."
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        default=MODEL,
        help=(
            f"Ollama model. Default: {MODEL}"
        ),
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Delete previous repair predictions "
            "and start the repair queue again."
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

    if (
        args.reset
        and OUTPUT_FILE.exists()
    ):

        OUTPUT_FILE.unlink()

        print(
            "Reset previous repair predictions."
        )


    queue = (
        load_queue()
    )

    final = (
        process_queue(
            queue,
            model=args.model,
            limit=args.limit,
        )
    )

    print_summary(
        final,
        len(queue),
    )


if __name__ == "__main__":
    main()
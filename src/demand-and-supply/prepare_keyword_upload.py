from pathlib import Path
import pandas as pd
import re


ROOT = Path(__file__).resolve().parent.parent.parent

TAXONOMY_FILE = (
    ROOT
    / "processed-data"
    / "market-entry"
    / "service-categories"
    / "taxonomy"
    / "service_taxonomy_final_master_final.csv"
)

OUTPUT_DIR = (
    ROOT
    / "processed-data"
    / "market-demand"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "keyword_planner_upload.csv"
)

REJECTED_FILE = (
    OUTPUT_DIR
    / "keyword_planner_rejected.csv"
)


def clean_keyword(text):

    text = str(text).strip()

    # Replace problematic separators while keeping meaning
    text = text.replace("&", " and ")
    text = text.replace("/", " or ")
    text = text.replace("+", " and ")
    text = text.replace(",", " ")

    # Normalize punctuation
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("’", "'")

    # Remove characters Google Keyword Planner commonly rejects
    text = re.sub(
        r"[!@%*#<>={}\[\]|\\^~`]",
        " ",
        text
    )

    # Remove repeated spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


df = pd.read_csv(TAXONOMY_FILE)


# ============================================================
# GENERAL SERVICE CATEGORIES
# ============================================================

general = (
    df[["service_family"]]
    .dropna()
    .drop_duplicates()
    .rename(
        columns={
            "service_family": "keyword"
        }
    )
)


# ============================================================
# SPECIFIC SERVICES / SUB-CATEGORIES
# ============================================================

specific = (
    df[["sub_service"]]
    .dropna()
    .drop_duplicates()
    .rename(
        columns={
            "sub_service": "keyword"
        }
    )
)


# ============================================================
# COMBINE
# ============================================================

keywords = pd.concat(
    [
        general,
        specific
    ],
    ignore_index=True
)


# Keep original value for checking
keywords["original_keyword"] = (
    keywords["keyword"]
)


# ============================================================
# CLEAN KEYWORDS
# ============================================================

keywords["keyword"] = (
    keywords["keyword"]
    .apply(clean_keyword)
)


# Remove empty keywords
keywords = keywords[
    keywords["keyword"] != ""
].copy()


# ============================================================
# GOOGLE 80 CHARACTER LIMIT
# ============================================================

too_long = (
    keywords["keyword"]
    .str.len()
    > 80
)

rejected = keywords[
    too_long
].copy()

rejected["reason"] = (
    "Keyword longer than 80 characters"
)

keywords = keywords[
    ~too_long
].copy()


# ============================================================
# REMOVE DUPLICATES
# ============================================================

keywords = (
    keywords
    .drop_duplicates(
        subset=["keyword"]
    )
    .sort_values("keyword")
    .reset_index(drop=True)
)


# ============================================================
# SAVE CLEAN UPLOAD FILE
# ============================================================

keywords[
    ["keyword"]
].to_csv(
    OUTPUT_FILE,
    index=False
)


# Save rejected keywords separately
rejected.to_csv(
    REJECTED_FILE,
    index=False
)


print(
    f"Clean keywords saved: {len(keywords):,}"
)

print(
    OUTPUT_FILE
)

print()

print(
    f"Rejected keywords: {len(rejected):,}"
)

print(
    REJECTED_FILE
)
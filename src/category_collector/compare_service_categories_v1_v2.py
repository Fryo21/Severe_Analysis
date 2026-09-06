from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = (
    PROJECT_ROOT
    / "raw-data"
    / "market-entry"
    / "service-categories"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed-data"
    / "market-entry"
    / "service-categories"
    / "v1-v2-comparison"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

V1_FILE = RAW_DIR / "service_categories_raw.csv"
V2_FILE = RAW_DIR / "service_categories_raw_v2.csv"

SHARED_OUTPUT = OUTPUT_DIR / "shared_v1_v2.csv"
V1_ONLY_OUTPUT = OUTPUT_DIR / "v1_only_for_review.csv"
V2_ONLY_OUTPUT = OUTPUT_DIR / "v2_only.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "v1_v2_comparison_summary.csv"

# This file is created only after you review v1_only_for_review.csv
FINAL_OUTPUT = OUTPUT_DIR / "service_categories_final_raw.csv"


# ============================================================
# NORMALISATION
# ============================================================

def normalise_text(value: str) -> str:
    value = str(value or "").strip()
    value = re.sub(r"\s+", " ", value)
    return value


def normalise_key(value: str) -> str:
    value = normalise_text(value).casefold()

    # Light normalisation only.
    # Do not aggressively standardise service names here.
    value = value.replace("&", "and")
    value = re.sub(r"[^\w\s-]", "", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value


def prepare(df: pd.DataFrame, version: str) -> pd.DataFrame:
    required = {"platform", "raw_service"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{version} is missing required columns: {sorted(missing)}"
        )

    df = df.copy()

    df["platform"] = df["platform"].map(normalise_text)
    df["raw_service"] = df["raw_service"].map(normalise_text)

    df["platform_key"] = df["platform"].str.casefold()
    df["service_key"] = df["raw_service"].map(normalise_key)

    df["comparison_key"] = (
        df["platform_key"]
        + "||"
        + df["service_key"]
    )

    df["source_version"] = version

    return df


# ============================================================
# COMPARISON
# ============================================================

def main():
    if not V1_FILE.exists():
        raise FileNotFoundError(f"V1 file not found: {V1_FILE}")

    if not V2_FILE.exists():
        raise FileNotFoundError(f"V2 file not found: {V2_FILE}")

    v1 = prepare(pd.read_csv(V1_FILE), "V1")
    v2 = prepare(pd.read_csv(V2_FILE), "V2")

    v1 = v1.drop_duplicates("comparison_key").copy()
    v2 = v2.drop_duplicates("comparison_key").copy()

    v1_keys = set(v1["comparison_key"])
    v2_keys = set(v2["comparison_key"])

    shared_keys = v1_keys & v2_keys
    v1_only_keys = v1_keys - v2_keys
    v2_only_keys = v2_keys - v1_keys

    shared = v2[
        v2["comparison_key"].isin(shared_keys)
    ].copy()

    v1_only = v1[
        v1["comparison_key"].isin(v1_only_keys)
    ].copy()

    v2_only = v2[
        v2["comparison_key"].isin(v2_only_keys)
    ].copy()

    # Add review columns to V1-only rows.
    # You manually set keep_from_v1 to YES for genuine services
    # that V2 missed.
    v1_only["keep_from_v1"] = ""
    v1_only["review_reason"] = ""

    shared.to_csv(SHARED_OUTPUT, index=False)
    v1_only.to_csv(V1_ONLY_OUTPUT, index=False)
    v2_only.to_csv(V2_ONLY_OUTPUT, index=False)

    platform_summary = []

    all_platforms = sorted(
        set(v1["platform"]) | set(v2["platform"]),
        key=str.casefold,
    )

    for platform in all_platforms:
        p1 = v1[v1["platform"].str.casefold() == platform.casefold()]
        p2 = v2[v2["platform"].str.casefold() == platform.casefold()]

        p1_keys = set(p1["comparison_key"])
        p2_keys = set(p2["comparison_key"])

        platform_summary.append(
            {
                "platform": platform,
                "v1_count": len(p1_keys),
                "v2_count": len(p2_keys),
                "shared_count": len(p1_keys & p2_keys),
                "v1_only_count": len(p1_keys - p2_keys),
                "v2_only_count": len(p2_keys - p1_keys),
            }
        )

    summary_df = pd.DataFrame(platform_summary)
    summary_df.to_csv(SUMMARY_OUTPUT, index=False)

    print("=" * 72)
    print("V1 vs V2 SERVICE CATEGORY COMPARISON")
    print("=" * 72)

    print(f"\nV1 unique platform/service rows: {len(v1):,}")
    print(f"V2 unique platform/service rows: {len(v2):,}")
    print(f"Shared: {len(shared):,}")
    print(f"V1 only: {len(v1_only):,}")
    print(f"V2 only: {len(v2_only):,}")

    print("\nBy platform:\n")
    print(summary_df.to_string(index=False))

    print("\nFiles written:")
    print(f"  {SHARED_OUTPUT}")
    print(f"  {V1_ONLY_OUTPUT}")
    print(f"  {V2_ONLY_OUTPUT}")
    print(f"  {SUMMARY_OUTPUT}")

    print(
        "\nNEXT STEP:\n"
        "Open v1_only_for_review.csv. "
        "For every genuine service missing from V2, set keep_from_v1 = YES. "
        "Do not add navigation/location/noise rows."
    )


if __name__ == "__main__":
    main()

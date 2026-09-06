from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "processed-data"
    / "market-entry"
    / "service-categories"
    / "taxonomy"
    / "service_taxonomy_final_master_final_coverage.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed-data"
    / "market-entry"
    / "service-categories"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "service_market_candidates.csv"


# ============================================================
# BUILD MARKET CANDIDATE TABLE
# ============================================================

def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "service_family",
        "sub_service",
        "platform_count",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # Keep only the fields needed for market analysis
    candidates = df[
        [
            "service_family",
            "sub_service",
            "platform_count",
        ]
    ].copy()

    # Remove taxonomy placeholders if present
    candidates = candidates[
        ~candidates["service_family"].isin(
            [
                "Excluded",
                "Other Services / Unclassified",
            ]
        )
    ]

    # Remove duplicates
    candidates = candidates.drop_duplicates(
        subset=["service_family", "sub_service"]
    )

    # Sort by family, then strongest platform coverage
    candidates = candidates.sort_values(
        by=[
            "service_family",
            "platform_count",
            "sub_service",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    )

    candidates.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("=" * 70)
    print("SEVERSE SERVICE MARKET CANDIDATES")
    print("=" * 70)

    print(f"\nService families: {candidates['service_family'].nunique():,}")
    print(f"Sub-services: {candidates['sub_service'].nunique():,}")
    print(f"Candidate rows: {len(candidates):,}")

    print("\nPlatform coverage:")
    print(
        candidates["platform_count"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(f"\nFile written:")
    print(f"  {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
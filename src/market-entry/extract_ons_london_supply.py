from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "raw-data"
    / "market-entry"
    / "london-supply"
    / "ukbusinessworkbook2025new.xlsx"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed-data"
    / "market-entry"
    / "london-supply"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "ons_london_local_units_by_industry.csv"


# ============================================================
# CONFIG
# ============================================================

SHEET_NAME = "Table 16"

# Table 16 uses row 4 in Excel / index 3 in pandas
# as the industry headers.
HEADER_ROW = 3


# ============================================================
# HELPERS
# ============================================================

def split_geography(value: str):
    """
    Example:
    'E09000008 : Croydon'
        ->
    ('E09000008', 'Croydon')
    """

    value = str(value).strip()

    if " : " in value:
        code, name = value.split(" : ", 1)
        return code.strip(), name.strip()

    return None, value


# ============================================================
# MAIN
# ============================================================

def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"ONS workbook not found:\n{INPUT_FILE}"
        )

    print("=" * 75)
    print("SEVERSE - ONS LONDON BUSINESS SUPPLY EXTRACTION")
    print("=" * 75)

    # --------------------------------------------------------
    # Read Table 16
    # --------------------------------------------------------

    df = pd.read_excel(
        INPUT_FILE,
        sheet_name=SHEET_NAME,
        header=HEADER_ROW,
    )

    # First column is geography
    geography_col = df.columns[0]

    df = df.rename(
        columns={
            geography_col: "geography"
        }
    )

    # --------------------------------------------------------
    # Keep London borough/local-authority rows only
    #
    # All 32 London boroughs + City of London use E090 codes.
    # This automatically excludes:
    # - London regional total
    # - Inner London total
    # - Outer London total
    # --------------------------------------------------------

    london = df[
        df["geography"]
        .astype(str)
        .str.startswith("E090")
    ].copy()

    # --------------------------------------------------------
    # Separate geography code and borough name
    # --------------------------------------------------------

    london[
        ["borough_code", "borough"]
    ] = london["geography"].apply(
        lambda x: pd.Series(split_geography(x))
    )

    london = london.drop(
        columns=["geography"]
    )

    # --------------------------------------------------------
    # Remove Total column
    #
    # We want individual industry groups only.
    # --------------------------------------------------------

    if "Total" in london.columns:
        london = london.drop(
            columns=["Total"]
        )

    # --------------------------------------------------------
    # Convert wide ONS table into long format
    #
    # Before:
    # borough | Construction | Retail | Finance ...
    #
    # After:
    # borough | industry_group | local_unit_count
    # --------------------------------------------------------

    industry_columns = [
        column
        for column in london.columns
        if column not in {
            "borough_code",
            "borough",
        }
    ]

    long_df = london.melt(
        id_vars=[
            "borough_code",
            "borough",
        ],
        value_vars=industry_columns,
        var_name="ons_industry_group",
        value_name="local_unit_count",
    )

    # --------------------------------------------------------
    # Clean counts
    # --------------------------------------------------------

    long_df["local_unit_count"] = pd.to_numeric(
        long_df["local_unit_count"],
        errors="coerce",
    )

    long_df = long_df.dropna(
        subset=["local_unit_count"]
    )

    long_df["local_unit_count"] = (
        long_df["local_unit_count"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    long_df = long_df.sort_values(
        by=[
            "borough",
            "ons_industry_group",
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    long_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(f"\nLondon local authorities: {long_df['borough'].nunique():,}")
    print(
        f"ONS industry groups: "
        f"{long_df['ons_industry_group'].nunique():,}"
    )
    print(f"Output rows: {len(long_df):,}")

    print("\nLondon boroughs:")
    for borough in sorted(
        long_df["borough"].unique(),
        key=str.casefold,
    ):
        print(f"  - {borough}")

    print("\nIndustry groups:")
    for industry in long_df[
        "ons_industry_group"
    ].unique():
        print(f"  - {industry}")

    print("\nFile written:")
    print(f"  {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
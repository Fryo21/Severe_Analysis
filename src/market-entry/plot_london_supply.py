from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "processed-data"
    / "market-entry"
    / "london-supply"
    / "london_service_supply_proxy.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "analysis"
    / "visualizations"
    / "market-entry"
    / "london-supply"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Supply file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = {
        "borough",
        "service_family",
        "vendor_supply_proxy",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    return df


# ============================================================
# 1. BOROUGH SUPPLY
# ============================================================

def plot_borough_supply(df):

    borough_supply = (
        df.groupby(
            "borough",
            as_index=False
        )["vendor_supply_proxy"]
        .sum()
        .sort_values(
            "vendor_supply_proxy",
            ascending=True
        )
    )

    plt.figure(figsize=(10, 10))

    plt.barh(
        borough_supply["borough"],
        borough_supply["vendor_supply_proxy"],
    )

    plt.title(
        "London Boroughs by Overall Vendor Supply Proxy"
    )

    plt.xlabel(
        "Vendor Supply Proxy"
    )

    plt.ylabel(
        "London Borough"
    )

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "borough_vendor_supply_proxy.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 2. SERVICE FAMILY SUPPLY
# ============================================================

def plot_service_family_supply(df):

    service_supply = (
        df.groupby(
            "service_family",
            as_index=False
        )["vendor_supply_proxy"]
        .mean()
        .sort_values(
            "vendor_supply_proxy",
            ascending=False
        )
        .head(25)
        .sort_values(
            "vendor_supply_proxy",
            ascending=True
        )
    )

    plt.figure(figsize=(11, 10))

    plt.barh(
        service_supply["service_family"],
        service_supply["vendor_supply_proxy"],
    )

    plt.title(
        "Top 25 Service Families by Average London Supply Proxy"
    )

    plt.xlabel(
        "Average Vendor Supply Proxy"
    )

    plt.ylabel(
        "Service Family"
    )

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "top_service_family_supply_proxy.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 3. SERVICE × BOROUGH HEATMAP
# ============================================================

def plot_supply_heatmap(df):

    # Using top 25 service families keeps the chart readable.
    top_services = (
        df.groupby("service_family")[
            "vendor_supply_proxy"
        ]
        .mean()
        .sort_values(
            ascending=False
        )
        .head(25)
        .index
    )

    subset = df[
        df["service_family"].isin(
            top_services
        )
    ]

    pivot = subset.pivot_table(
        index="service_family",
        columns="borough",
        values="vendor_supply_proxy",
        aggfunc="sum",
    )

    plt.figure(figsize=(20, 12))

    image = plt.imshow(
        pivot.values,
        aspect="auto",
    )

    plt.colorbar(
        image,
        label="Vendor Supply Proxy"
    )

    plt.xticks(
        range(len(pivot.columns)),
        pivot.columns,
        rotation=90,
        fontsize=8,
    )

    plt.yticks(
        range(len(pivot.index)),
        pivot.index,
        fontsize=8,
    )

    plt.title(
        "London Vendor Supply Proxy: Service Family × Borough"
    )

    plt.xlabel(
        "London Borough"
    )

    plt.ylabel(
        "Service Family"
    )

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "service_borough_supply_heatmap.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("SEVERSE - LONDON SUPPLY VISUALISATIONS")
    print("=" * 75)

    df = load_data()

    print(
        f"\nRows loaded: {len(df):,}"
    )

    print(
        f"Service families: "
        f"{df['service_family'].nunique():,}"
    )

    print(
        f"London boroughs: "
        f"{df['borough'].nunique():,}"
    )

    plot_borough_supply(df)

    print(
        "\nCreated borough supply chart."
    )

    plot_service_family_supply(df)

    print(
        "Created service-family supply chart."
    )

    plot_supply_heatmap(df)

    print(
        "Created service × borough heatmap."
    )

    print(
        f"\nVisualisations written to:\n"
        f"  {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ONS_FILE = (
    PROJECT_ROOT
    / "processed-data"
    / "market-entry"
    / "london-supply"
    / "ons_london_local_units_by_industry.csv"
)

SERVICE_FILE = (
    PROJECT_ROOT
    / "processed-data"
    / "market-entry"
    / "service-categories"
    / "service_market_candidates.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed-data"
    / "market-entry"
    / "london-supply"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "london_service_supply_proxy.csv"
MAPPING_FILE = OUTPUT_DIR / "service_family_ons_mapping.csv"


# ============================================================
# ONS GROUP NAMES
# ============================================================

AGRICULTURE = "01-03 : Agriculture, forestry & fishing"
PRODUCTION = "05-39 : Production"
CONSTRUCTION = "41-43 : Construction"
MOTOR = "45 : Motor trades"
WHOLESALE = "46 : Wholesale"
RETAIL = "47 : Retail"
TRANSPORT = "49-53 : Transport & Storage (inc. postal)"
FOOD = "55-56 : Accommodation & food services"
INFO = "58-63 : Information & communication"
FINANCE = "64-66 : Finance & insurance"
PROPERTY = "68 : Property"
PROFESSIONAL = "69-75 : Professional, scientific & technical"
ADMIN = "77-82 : Business administration & support services"
PUBLIC = "84 : Public administration & defence"
EDUCATION = "85 : Education"
HEALTH = "86-88 : Health"
OTHER = "90-99 : Arts, entertainment, recreation & other services"


# ============================================================
# SERVICE FAMILY → ONS GROUP RULES
#
# Multiple groups are allowed.
# The result is a supply proxy, not an exact vendor count.
# ============================================================

FAMILY_RULES = {

    # Construction / property trades
    "Building & Construction": [CONSTRUCTION],
    "Plumbing & Drainage Services": [CONSTRUCTION],
    "Heating, Boiler & Gas Services": [CONSTRUCTION],
    "Electrical Services": [CONSTRUCTION],
    "Roofing Services": [CONSTRUCTION],
    "Doors, Windows & Glazing": [CONSTRUCTION],
    "Bathroom Services": [CONSTRUCTION],
    "Kitchen Services": [CONSTRUCTION],
    "Flooring & Tiling Services": [CONSTRUCTION],
    "Painting, Decorating & Wall Finishes": [CONSTRUCTION],
    "Groundworks, Excavation & Plant Hire": [CONSTRUCTION],
    "Concrete & Aggregate Services": [CONSTRUCTION],
    "Scaffolding Services": [CONSTRUCTION],
    "Stone & Masonry Services": [CONSTRUCTION],
    "Cladding & Exterior Finishes": [CONSTRUCTION],
    "Decking Services": [CONSTRUCTION],
    "Fencing, Gates & Outdoor Structures": [CONSTRUCTION],
    "Driveways, Paving & Surfacing": [CONSTRUCTION],
    "Damp, Waterproofing & Restoration": [CONSTRUCTION],
    "Conservatory Services": [CONSTRUCTION],
    "Chimney & Fireplace Services": [CONSTRUCTION],
    "Air Conditioning & Ventilation": [CONSTRUCTION],
    "Renewable Energy & Insulation": [CONSTRUCTION],
    "Loft, Attic & Storage Services": [CONSTRUCTION],
    "Home Improvement & General Maintenance": [CONSTRUCTION],
    "Handyman, Assembly & Mounting": [CONSTRUCTION, ADMIN],

    # Wood / fabrication
    "Furniture, Carpentry & Woodwork": [CONSTRUCTION, PRODUCTION],
    "Metalwork & Fabrication": [PRODUCTION, CONSTRUCTION],
    "Surface Treatment & Specialist Coatings": [PRODUCTION, CONSTRUCTION],

    # Motor
    "Vehicle Services": [MOTOR],
    "Vehicle Graphics & Signage": [MOTOR, OTHER],

    # Transport
    "Transport & Transfer Services": [TRANSPORT],
    "Removals, Delivery & Courier Services": [TRANSPORT],
    "Marine & Boat Services": [TRANSPORT, PRODUCTION],

    # Cleaning / waste / property support
    "Cleaning Services": [ADMIN],
    "Waste, Clearance & Recycling": [ADMIN],
    "Laundry, Housekeeping & Organisation": [ADMIN],
    "Property, Landlord & Estate Services": [PROPERTY],
    "Commercial & Business Property Services": [PROPERTY, PROFESSIONAL],

    # Security
    "Security & Alarm Services": [ADMIN],
    "Fire & Safety Services": [PROFESSIONAL, ADMIN],
    "Locksmith, Keys & Safes": [ADMIN, CONSTRUCTION],

    # IT / digital
    "IT Support & Device Services": [INFO],
    "Software & Development Services": [INFO],
    "Web & Digital Services": [INFO],
    "Telecoms & Telephone Services": [INFO],
    "Smart Home, Network & Cabling": [INFO, CONSTRUCTION],
    "Electronics Repair Services": [INFO, OTHER],
    "TV, Aerial & Audio Visual Services": [INFO, OTHER],

    # Professional services
    "Accounting & Finance Services": [FINANCE, PROFESSIONAL],
    "Legal Services": [PROFESSIONAL],
    "Architecture, Surveying & Planning": [PROFESSIONAL],
    "Marketing & Creative Services": [PROFESSIONAL, INFO],
    "Writing, Translation & Language Services": [PROFESSIONAL],
    "Business & Administrative Services": [PROFESSIONAL, ADMIN],
    "Coaching & Personal Development": [PROFESSIONAL, EDUCATION],

    # Education
    "Tutoring & Lessons": [EDUCATION],
    "Training & Instruction": [EDUCATION],

    # Health / care
    "Health, Therapy & Wellness": [HEALTH],
    "Childcare & Care Services": [HEALTH, EDUCATION],
    "Accessibility & Mobility Services": [HEALTH, RETAIL],

    # Hair / beauty / fitness
    "Hair & Barber Services": [OTHER],
    "Beauty & Personal Care": [OTHER],
    "Fitness & Sports Coaching": [OTHER],
    "Tailoring & Clothing Alterations": [OTHER],

    # Events / media
    "Events & Wedding Services": [OTHER, ADMIN],
    "Photography & Video Services": [OTHER, PROFESSIONAL],
    "Printing, Signage & Art Services": [OTHER, PRODUCTION],

    # Pets
    "Pet & Animal Services": [OTHER],

    # Food / hospitality
    "Food, Baking & Personal Chef Services": [FOOD],

    # Outdoor / land
    "Gardening & Landscaping": [CONSTRUCTION, OTHER],
    "Tree & Grounds Services": [AGRICULTURE, OTHER],
    "Agricultural & Rural Services": [AGRICULTURE],

    # Equipment
    "Appliance & Equipment Services": [OTHER, RETAIL],
    "Hire & Access Equipment": [ADMIN],
    "Water Treatment, Tanks & Pumps": [CONSTRUCTION],
    "Septic, Sewage & Drainage Infrastructure": [CONSTRUCTION],

    # Specialist
    "Pest & Bird Control": [ADMIN],
    "Asbestos & Hazardous Material Services": [CONSTRUCTION, PROFESSIONAL],
    "Specialist Repair Services": [OTHER],
    "Fabric, Clothing & Specialist Repair": [OTHER],
}


# ============================================================
# FALLBACK KEYWORD MAPPING
#
# Used only when a family is not explicitly defined above.
# ============================================================

def infer_ons_groups(service_family: str):

    name = service_family.casefold()

    if any(word in name for word in [
        "building", "construction", "roof", "plumb",
        "heating", "electrical", "window", "door",
        "floor", "tiling", "paint", "decorat",
        "groundwork", "concrete", "masonry",
    ]):
        return [CONSTRUCTION]

    if any(word in name for word in [
        "vehicle", "motor", "car ",
    ]):
        return [MOTOR]

    if any(word in name for word in [
        "transport", "courier", "delivery", "removal",
    ]):
        return [TRANSPORT]

    if any(word in name for word in [
        "software", "digital", "web", "it ", "telecom",
    ]):
        return [INFO]

    if any(word in name for word in [
        "finance", "account",
    ]):
        return [FINANCE, PROFESSIONAL]

    if any(word in name for word in [
        "legal", "architect", "survey", "consult",
    ]):
        return [PROFESSIONAL]

    if any(word in name for word in [
        "health", "therapy", "care",
    ]):
        return [HEALTH]

    if any(word in name for word in [
        "education", "training", "lesson", "tutor",
    ]):
        return [EDUCATION]

    if any(word in name for word in [
        "hair", "beauty", "fitness", "event",
        "wedding", "pet", "repair",
    ]):
        return [OTHER]

    return [OTHER]


# ============================================================
# MAIN
# ============================================================

def main():

    if not ONS_FILE.exists():
        raise FileNotFoundError(
            f"ONS supply file not found:\n{ONS_FILE}"
        )

    if not SERVICE_FILE.exists():
        raise FileNotFoundError(
            f"SEVERSE service file not found:\n{SERVICE_FILE}"
        )

    ons = pd.read_csv(ONS_FILE)
    services = pd.read_csv(SERVICE_FILE)

    required_ons = {
        "borough_code",
        "borough",
        "ons_industry_group",
        "local_unit_count",
    }

    required_services = {
        "service_family",
        "sub_service",
        "platform_count",
    }

    missing_ons = required_ons - set(ons.columns)
    missing_services = required_services - set(services.columns)

    if missing_ons:
        raise ValueError(
            f"Missing ONS columns: {sorted(missing_ons)}"
        )

    if missing_services:
        raise ValueError(
            f"Missing service columns: {sorted(missing_services)}"
        )

    # --------------------------------------------------------
    # Unique service families
    # --------------------------------------------------------

    families = (
        services[["service_family"]]
        .drop_duplicates()
        .sort_values("service_family")
    )

    # --------------------------------------------------------
    # Build service family → ONS mapping
    # --------------------------------------------------------

    mapping_rows = []

    for family in families["service_family"]:

        ons_groups = FAMILY_RULES.get(
            family,
            infer_ons_groups(family),
        )

        for group in ons_groups:

            mapping_rows.append({
                "service_family": family,
                "ons_industry_group": group,
            })

    mapping = pd.DataFrame(mapping_rows)

    mapping.to_csv(
        MAPPING_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Join ONS borough supply to SEVERSE families
    # --------------------------------------------------------

    supply = mapping.merge(
        ons,
        on="ons_industry_group",
        how="left",
    )

    # --------------------------------------------------------
    # Some families map to multiple ONS groups.
    #
    # Sum those relevant local-unit counts into one
    # comparative supply proxy.
    # --------------------------------------------------------

    supply_proxy = (
        supply
        .groupby(
            [
                "borough_code",
                "borough",
                "service_family",
            ],
            as_index=False,
        )
        .agg(
            vendor_supply_proxy=(
                "local_unit_count",
                "sum",
            ),
            ons_group_count=(
                "ons_industry_group",
                "nunique",
            ),
        )
    )

    # --------------------------------------------------------
    # Add within-family borough ranking
    #
    # 1 = highest supply borough for that service family
    # --------------------------------------------------------

    supply_proxy[
        "borough_supply_rank"
    ] = (
        supply_proxy
        .groupby("service_family")[
            "vendor_supply_proxy"
        ]
        .rank(
            method="dense",
            ascending=False,
        )
        .astype(int)
    )

    # --------------------------------------------------------
    # Add percentile-style supply score 0-100
    #
    # Makes families easier to compare geographically.
    # --------------------------------------------------------

    supply_proxy[
        "borough_supply_percentile"
    ] = (
        supply_proxy
        .groupby("service_family")[
            "vendor_supply_proxy"
        ]
        .rank(
            pct=True,
        )
        .mul(100)
        .round(1)
    )

    supply_proxy = supply_proxy.sort_values(
        [
            "service_family",
            "vendor_supply_proxy",
        ],
        ascending=[
            True,
            False,
        ],
    )

    supply_proxy.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("=" * 75)
    print("SEVERSE - LONDON SERVICE SUPPLY PROXY")
    print("=" * 75)

    print(
        f"\nService families mapped: "
        f"{supply_proxy['service_family'].nunique():,}"
    )

    print(
        f"London boroughs: "
        f"{supply_proxy['borough'].nunique():,}"
    )

    print(
        f"Output rows: "
        f"{len(supply_proxy):,}"
    )

    print(
        f"\nService-to-ONS mapping written:\n"
        f"  {MAPPING_FILE}"
    )

    print(
        f"\nSupply proxy written:\n"
        f"  {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
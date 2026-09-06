from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "raw-data" / "market-entry" / "service-categories"
OUTPUT_DIR = PROJECT_ROOT / "processed-data" / "market-entry" / "service-categories"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

V1_FILE = RAW_DIR / "service_categories_raw.csv"
V2_FILE = RAW_DIR / "service_categories_raw_v2.csv"

COMBINED_OUTPUT = OUTPUT_DIR / "service_categories_combined_clean.csv"
STANDARDISED_OUTPUT = OUTPUT_DIR / "service_categories_standardised.csv"
PLATFORM_COVERAGE_OUTPUT = OUTPUT_DIR / "service_platform_coverage.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "service_standardisation_summary.csv"


NAV_NOISE = {
    "", "home", "about", "about us", "contact", "contact us", "login",
    "log in", "sign up", "join", "help", "privacy", "terms",
    "terms & conditions", "terms and conditions", "careers", "blog", "news",
    "cookie policy", "skip to navigation", "skip to search", "skip to content",
    "skip to footer", "see all", "view all", "more", "all categories",
    "all services", "all trades", "popular services", "popular categories",
    "business", "pricing", "reviews", "advice", "advice centre", "sitemap",
    "how it works", "getting started", "legal",
}

LOCATION_ONLY = {
    "aberdeenshire", "birmingham", "bradford", "brighton", "bristol",
    "cambridge", "cardiff", "coventry", "derby", "edinburgh", "england",
    "glasgow", "greater london", "leeds", "leicester", "liverpool", "london",
    "manchester", "newcastle", "nottingham", "portsmouth", "scotland",
    "sheffield", "swansea", "wales",
}

GENERIC_BROAD_HEADINGS = {
    "events & entertainers", "health & wellness", "house & home",
    "lessons & training", "professional services", "business services", "other",
}


def normalise_space(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def clean_raw_service(value: str) -> str:
    text = normalise_space(value)
    text = re.sub(r"\s+near me$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+in\s+[A-Za-z .'-]+$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^all\s+", "", text, flags=re.IGNORECASE)
    return normalise_space(text)


def is_noise(value: str) -> bool:
    text = clean_raw_service(value)

    if not text:
        return True

    lower = text.casefold()

    if lower in NAV_NOISE or lower in LOCATION_ONLY or lower in GENERIC_BROAD_HEADINGS:
        return True

    if len(text) < 2 or len(text) > 120:
        return True

    if re.fullmatch(r"[A-Z]", text) or re.fullmatch(r"\d+", text):
        return True

    banned_contains = [
        "skip to ",
        "cookie",
        "privacy policy",
        "terms and conditions",
        "join as a professional",
        "find a local trusted trader",
    ]

    return any(token in lower for token in banned_contains)


RULES = [
    (r"\bplumb|plumber|bathroom.*plumb", "Plumbing", "Home & Trades"),
    (r"\belectric|electrical|rewir|socket|lighting", "Electrical Services", "Home & Trades"),
    (r"\broof|gutter|fascia|soffit", "Roofing & Guttering", "Home & Trades"),
    (r"\bbuild|brick|masonry|stonework|extension", "Building & Construction", "Home & Trades"),
    (r"\bcarpenter|carpentry|joiner|joinery", "Carpentry & Joinery", "Home & Trades"),
    (r"\bhandyman|odd jobs", "Handyman", "Home & Trades"),
    (r"\blocksmith|lock repair|lock installation", "Locksmith", "Home & Trades"),
    (r"\bpaint|decorat", "Painting & Decorating", "Home & Trades"),
    (r"\btiler|tiling", "Tiling", "Home & Trades"),
    (r"\bplaster|render", "Plastering & Rendering", "Home & Trades"),
    (r"\bfloor|flooring|carpet fitting|laminate|vinyl floor", "Flooring", "Home & Trades"),
    (r"\bwindow|double glazing|glazier|glass repair", "Windows & Glazing", "Home & Trades"),
    (r"\bdoor|garage door", "Doors & Door Services", "Home & Trades"),
    (r"\bboiler|central heating|heating engineer|radiator", "Heating & Boiler Services", "Home & Trades"),
    (r"\bgas engineer|gas boiler|gas appliance", "Gas Services", "Home & Trades"),
    (r"\bair conditioning|hvac|ventilation", "Air Conditioning & Ventilation", "Home & Trades"),
    (r"\bkitchen|worktop|countertop", "Kitchen Services", "Home & Trades"),
    (r"\bbathroom|shower|wet room", "Bathroom Services", "Home & Trades"),
    (r"\bgarden|gardener|landscap|lawn|hedge|tree surgeon|tree service", "Gardening & Landscaping", "Home & Trades"),
    (r"\bfence|fencing|gate installation|gate repair", "Fencing & Gates", "Home & Trades"),
    (r"\bdriveway|paving|patio|block paving", "Driveways & Paving", "Home & Trades"),
    (r"\bdamp|waterproof|mould|mold", "Damp & Waterproofing", "Home & Trades"),
    (r"\bpest|vermin|rodent|insect control", "Pest Control", "Home & Trades"),
    (r"\bsecurity|alarm|cctv|access control", "Security Systems", "Home & Trades"),
    (r"\baerial|satellite|tv installation", "Aerial & Satellite Services", "Home & Trades"),
    (r"\binsulation", "Insulation", "Home & Trades"),
    (r"\bscaffold", "Scaffolding", "Home & Trades"),
    (r"\bdemolition", "Demolition", "Home & Trades"),
    (r"\barchitect|architecture", "Architectural Services", "Professional Services"),
    (r"\bsurveyor|surveying", "Surveying", "Professional Services"),
    (r"\binterior design|interior designer", "Interior Design", "Professional Services"),
    (r"\bclean|cleaner|cleaning|maid|housekeeping", "Cleaning", "Cleaning & Household"),
    (r"\bwindow cleaning", "Window Cleaning", "Cleaning & Household"),
    (r"\bcarpet clean|upholstery clean", "Carpet & Upholstery Cleaning", "Cleaning & Household"),
    (r"\bpressure wash|jet wash", "Pressure Washing", "Cleaning & Household"),
    (r"\brubbish|waste removal|junk removal", "Rubbish Removal", "Cleaning & Household"),
    (r"\bremoval|moving|mover|house move|man and van", "Removals & Moving", "Cleaning & Household"),
    (r"\bappliance repair|washing machine repair|dishwasher repair|fridge repair|oven repair", "Appliance Repair", "Repair Services"),
    (r"\bphone repair|mobile repair|tablet repair", "Phone & Device Repair", "Repair Services"),
    (r"\bcomputer repair|pc repair|laptop repair", "Computer Repair", "Repair Services"),
    (r"\bfurniture assembly|flat pack|assembly", "Furniture Assembly", "Repair Services"),
    (r"\bfurniture repair|upholstery repair", "Furniture Repair", "Repair Services"),
    (r"\bcar repair|mechanic|auto repair|vehicle repair", "Vehicle Repair", "Automotive"),
    (r"\bauto electrician|car electrician", "Auto Electrical", "Automotive"),
    (r"\bcar wash|car valeting|detailing", "Car Cleaning & Detailing", "Automotive"),
    (r"\btyre|tire", "Tyre Services", "Automotive"),
    (r"\bmotorcycle|motorbike", "Motorcycle Services", "Automotive"),
    (r"\bhair salon|hairdresser|hair stylist|hair styling|barber|haircut", "Hair & Barber Services", "Beauty & Wellness"),
    (r"\bnail|manicure|pedicure", "Nail Services", "Beauty & Wellness"),
    (r"\bmassage", "Massage", "Beauty & Wellness"),
    (r"\bwax|hair removal|laser hair removal", "Hair Removal", "Beauty & Wellness"),
    (r"\bbeauty salon|beautician|facial|skin care|skincare|eyebrow|eyelash|lash|brow", "Beauty & Skincare", "Beauty & Wellness"),
    (r"\bspa|wellness", "Spa & Wellness", "Beauty & Wellness"),
    (r"\btattoo|piercing", "Tattoo & Piercing", "Beauty & Wellness"),
    (r"\btanning", "Tanning", "Beauty & Wellness"),
    (r"\bmedspa|aesthetic|cosmetic treatment", "Aesthetics & Medspa", "Beauty & Wellness"),
    (r"\bpersonal trainer|fitness trainer|fitness", "Fitness & Personal Training", "Health & Fitness"),
    (r"\bphysio|physical therapy|physiotherapy", "Physiotherapy", "Health & Fitness"),
    (r"\btherapy|therapist|counselling|counseling", "Therapy & Counselling", "Health & Fitness"),
    (r"\byoga", "Yoga", "Health & Fitness"),
    (r"\bdog groom|pet groom", "Pet Grooming", "Pet Services"),
    (r"\bdog walk|pet walk", "Dog Walking", "Pet Services"),
    (r"\bpet sit|dog sit|cat sit", "Pet Sitting", "Pet Services"),
    (r"\bdog train|pet train", "Pet Training", "Pet Services"),
    (r"\bphotograph|photographer", "Photography", "Events & Creative"),
    (r"\bvideograph|video production", "Videography", "Events & Creative"),
    (r"\bdj\b|disc jockey", "DJ Services", "Events & Creative"),
    (r"\bcatering|caterer", "Catering", "Events & Creative"),
    (r"\bevent planner|party planner|wedding planner", "Event Planning", "Events & Creative"),
    (r"\btutor|tuition|lessons|teacher", "Tutoring & Lessons", "Education & Lessons"),
    (r"\baccountant|bookkeep", "Accounting & Bookkeeping", "Professional Services"),
    (r"\blawyer|solicitor|legal service", "Legal Services", "Professional Services"),
    (r"\bweb design|website design|web developer", "Web Design & Development", "Professional Services"),
    (r"\bmarketing|seo|social media", "Marketing Services", "Professional Services"),
]


def canonicalise(service: str) -> tuple[str, str]:
    text = clean_raw_service(service)
    lower = text.casefold()

    for pattern, canonical, family in RULES:
        if re.search(pattern, lower, flags=re.IGNORECASE):
            return canonical, family

    return text, "Other / Unmapped"


def load_source(path: Path, version: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path).copy()

    if "platform" not in df.columns or "raw_service" not in df.columns:
        raise ValueError(f"{version} file must contain platform and raw_service columns.")

    df["source_version"] = version
    return df


def main():
    v1 = load_source(V1_FILE, "V1")
    v2 = load_source(V2_FILE, "V2")

    frames = [df for df in [v1, v2] if not df.empty]

    if not frames:
        raise RuntimeError("Neither V1 nor V2 input file was found.")

    combined = pd.concat(frames, ignore_index=True, sort=False)

    combined["platform"] = combined["platform"].map(normalise_space)
    combined["raw_service_original"] = combined["raw_service"].map(normalise_space)
    combined["raw_service"] = combined["raw_service"].map(clean_raw_service)
    combined["is_noise"] = combined["raw_service"].map(is_noise)

    clean = combined[~combined["is_noise"]].copy()

    clean["platform_key"] = clean["platform"].str.casefold()
    clean["service_key"] = clean["raw_service"].str.casefold()

    clean = (
        clean.sort_values(
            ["platform", "raw_service", "source_version"],
            ascending=[True, True, False],
        )
        .drop_duplicates(
            subset=["platform_key", "service_key"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    canonical_values = clean["raw_service"].map(canonicalise)
    clean["canonical_service"] = canonical_values.map(lambda x: x[0])
    clean["service_family"] = canonical_values.map(lambda x: x[1])

    desired_columns = [
        "platform",
        "raw_service_original",
        "raw_service",
        "canonical_service",
        "service_family",
        "source_version",
        "source_url",
        "source_type",
        "collection_date",
        "coverage_note",
        "geography_scope",
    ]

    output_columns = [col for col in desired_columns if col in clean.columns]
    clean[output_columns].to_csv(STANDARDISED_OUTPUT, index=False)

    raw_columns = [
        col
        for col in [
            "platform",
            "raw_service_original",
            "raw_service",
            "source_version",
            "source_url",
            "source_type",
            "collection_date",
            "coverage_note",
            "geography_scope",
        ]
        if col in clean.columns
    ]

    clean[raw_columns].to_csv(COMBINED_OUTPUT, index=False)

    coverage = (
        clean
        .groupby(["canonical_service", "service_family"], as_index=False)
        .agg(
            platform_count=("platform", "nunique"),
            platforms=(
                "platform",
                lambda x: " | ".join(sorted(set(x), key=str.casefold)),
            ),
            raw_label_count=("raw_service", "nunique"),
        )
        .sort_values(
            ["platform_count", "raw_label_count", "canonical_service"],
            ascending=[False, False, True],
        )
    )

    coverage.to_csv(PLATFORM_COVERAGE_OUTPUT, index=False)

    summary = pd.DataFrame(
        [
            {"metric": "combined_input_rows", "value": len(combined)},
            {"metric": "noise_rows_removed", "value": int(combined["is_noise"].sum())},
            {"metric": "clean_platform_service_rows", "value": len(clean)},
            {"metric": "unique_clean_raw_labels", "value": clean["raw_service"].nunique()},
            {"metric": "canonical_services", "value": clean["canonical_service"].nunique()},
            {"metric": "service_families", "value": clean["service_family"].nunique()},
        ]
    )

    summary.to_csv(SUMMARY_OUTPUT, index=False)

    print("=" * 76)
    print("SEVERSE SERVICE CATEGORY STANDARDISATION")
    print("=" * 76)

    print(f"\nCombined V1 + V2 rows: {len(combined):,}")
    print(f"Noise removed: {int(combined['is_noise'].sum()):,}")
    print(f"Clean platform/service rows: {len(clean):,}")
    print(f"Unique clean raw labels: {clean['raw_service'].nunique():,}")
    print(f"Canonical services: {clean['canonical_service'].nunique():,}")

    print("\nTop canonical services by platform coverage:\n")
    print(coverage.head(30).to_string(index=False))

    print("\nFiles written:")
    print(f"  {COMBINED_OUTPUT}")
    print(f"  {STANDARDISED_OUTPUT}")
    print(f"  {PLATFORM_COVERAGE_OUTPUT}")
    print(f"  {SUMMARY_OUTPUT}")

    print(
        "\nNEXT STEP: inspect the top canonical services and the "
        "'Other / Unmapped' group before market scoring."
    )


if __name__ == "__main__":
    main()

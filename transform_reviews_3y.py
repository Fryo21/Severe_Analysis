import os
import re
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "competitor_reviews/ALL_COMPETITOR_REVIEWS.txt"
OUTPUT_FOLDER = "transformed_reviews"

# Automatically means "exactly 3 years ago until today"
END_DATE = pd.Timestamp.today().normalize()
START_DATE = END_DATE - pd.DateOffset(years=3)


# ============================================================
# NORMALISE COMPETITOR NAMES
# ============================================================

def get_competitor(app_name):
    name = str(app_name).lower()

    if "yell" in name:
        return "Yell"

    if "checkatrade" in name:
        return "Checkatrade"

    if "bark" in name:
        return "Bark"

    if "yelp" in name:
        return "Yelp"

    return app_name


# ============================================================
# PARSE TXT FILE
# ============================================================

def parse_reviews_txt(file_path):

    records = []
    current = None
    mode = None

    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    for raw_line in lines:

        line = raw_line.rstrip("\n")

        # Start of a new review
        if line.startswith("APP:"):

            if current:
                records.append(current)

            current = {
                "app": line.replace("APP:", "", 1).strip(),
                "store": "",
                "country": "",
                "rating": None,
                "date": "",
                "version": "",
                "author": "",
                "title": "",
                "review": "",
                "developer_reply": ""
            }

            mode = None
            continue

        if current is None:
            continue

        # --------------------------------------------
        # Normal fields
        # --------------------------------------------

        if line.startswith("STORE:"):
            current["store"] = line.replace(
                "STORE:", "", 1
            ).strip()
            mode = None

        elif line.startswith("COUNTRY:"):
            current["country"] = line.replace(
                "COUNTRY:", "", 1
            ).strip()
            mode = None

        elif line.startswith("RATING:"):

            rating_text = line.replace(
                "RATING:", "", 1
            ).strip()

            match = re.search(r"\d+", rating_text)

            if match:
                current["rating"] = int(match.group())

            mode = None

        elif line.startswith("DATE:"):
            current["date"] = line.replace(
                "DATE:", "", 1
            ).strip()
            mode = None

        elif line.startswith("VERSION:"):
            current["version"] = line.replace(
                "VERSION:", "", 1
            ).strip()
            mode = None

        elif line.startswith("AUTHOR:"):
            current["author"] = line.replace(
                "AUTHOR:", "", 1
            ).strip()
            mode = None

        elif line.startswith("TITLE:"):
            current["title"] = line.replace(
                "TITLE:", "", 1
            ).strip()
            mode = None

        # --------------------------------------------
        # Review text
        # --------------------------------------------

        elif line.strip() == "REVIEW:":
            mode = "review"

        elif line.strip() == "DEVELOPER REPLY:":
            mode = "developer_reply"

        elif line.startswith("="):
            mode = None

        else:

            if mode == "review" and line.strip():

                if current["review"]:
                    current["review"] += " "

                current["review"] += line.strip()

            elif mode == "developer_reply" and line.strip():

                if current["developer_reply"]:
                    current["developer_reply"] += " "

                current["developer_reply"] += line.strip()

    # Add final review
    if current:
        records.append(current)

    df = pd.DataFrame(records)

    return df


# ============================================================
# CLEAN DATA
# ============================================================

def clean_reviews(df):

    # Standard competitor name
    df["competitor"] = df["app"].apply(
        get_competitor
    )

    # Numeric rating
    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce"
    )

    # Convert timestamps
    df["review_date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
        utc=True
    )

    # Remove timezone for easier analysis
    df["review_date"] = (
        df["review_date"]
        .dt.tz_convert(None)
    )

    # Date without time
    df["review_day"] = (
        df["review_date"]
        .dt.normalize()
    )

    return df


# ============================================================
# FILTER LAST 3 YEARS
# ============================================================

def filter_last_3_years(df):

    filtered = df[
        (df["review_day"] >= START_DATE)
        &
        (df["review_day"] <= END_DATE)
    ].copy()

    return filtered


# ============================================================
# BEFORE / AFTER FILTER STATISTICS
# ============================================================

def create_filter_impact(df, filtered_df):

    competitors = [
        "Yell",
        "Checkatrade",
        "Bark",
        "Yelp"
    ]

    rows = []

    # --------------------------------------------
    # ALL competitors
    # --------------------------------------------

    total_before = len(df)
    total_after = len(filtered_df)

    rows.append({
        "competitor": "ALL",
        "reviews_before_filter": total_before,
        "reviews_after_filter": total_after,
        "reviews_removed": total_before - total_after,
        "retained_percent": round(
            (total_after / total_before * 100)
            if total_before > 0 else 0,
            2
        )
    })

    # --------------------------------------------
    # Individual competitors
    # --------------------------------------------

    for competitor in competitors:

        before = len(
            df[
                df["competitor"] == competitor
            ]
        )

        after = len(
            filtered_df[
                filtered_df["competitor"]
                == competitor
            ]
        )

        rows.append({
            "competitor": competitor,
            "reviews_before_filter": before,
            "reviews_after_filter": after,
            "reviews_removed": before - after,
            "retained_percent": round(
                (after / before * 100)
                if before > 0 else 0,
                2
            )
        })

    return pd.DataFrame(rows)


# ============================================================
# REVIEW STAT FUNCTION
# ============================================================

def calculate_stats(
    df,
    competitor,
    period,
    period_start,
    period_end
):

    total = len(df)

    negative = len(
        df[
            df["rating"].between(
                1,
                3,
                inclusive="both"
            )
        ]
    )

    positive = len(
        df[
            df["rating"].between(
                4,
                5,
                inclusive="both"
            )
        ]
    )

    return {
        "competitor": competitor,
        "period": period,
        "period_start": period_start.date(),
        "period_end": period_end.date(),
        "total_reviews": total,
        "reviews_1_3_star": negative,
        "reviews_4_5_star": positive
    }


# ============================================================
# CREATE 3 YEAR STATISTICS
# ============================================================

def create_review_stats(filtered_df):

    competitors = [
        "Yell",
        "Checkatrade",
        "Bark",
        "Yelp"
    ]

    rows = []

    # ========================================================
    # YEAR WINDOWS
    # ========================================================

    year1_start = START_DATE

    year1_end = (
        START_DATE
        + pd.DateOffset(years=1)
        - pd.Timedelta(days=1)
    )

    year2_start = (
        START_DATE
        + pd.DateOffset(years=1)
    )

    year2_end = (
        START_DATE
        + pd.DateOffset(years=2)
        - pd.Timedelta(days=1)
    )

    year3_start = (
        START_DATE
        + pd.DateOffset(years=2)
    )

    year3_end = END_DATE

    periods = [
        (
            "Year 1",
            year1_start,
            year1_end
        ),
        (
            "Year 2",
            year2_start,
            year2_end
        ),
        (
            "Year 3",
            year3_start,
            year3_end
        )
    ]

    # ========================================================
    # ALL COMPETITORS COMBINED
    # ========================================================

    rows.append(
        calculate_stats(
            filtered_df,
            "ALL",
            "Past 3 Years",
            START_DATE,
            END_DATE
        )
    )

    for period_name, start, end in periods:

        period_df = filtered_df[
            (filtered_df["review_day"] >= start)
            &
            (filtered_df["review_day"] <= end)
        ]

        rows.append(
            calculate_stats(
                period_df,
                "ALL",
                period_name,
                start,
                end
            )
        )

    # ========================================================
    # EACH COMPETITOR
    # ========================================================

    for competitor in competitors:

        competitor_df = filtered_df[
            filtered_df["competitor"]
            == competitor
        ]

        # Overall 3 years
        rows.append(
            calculate_stats(
                competitor_df,
                competitor,
                "Past 3 Years",
                START_DATE,
                END_DATE
            )
        )

        # Individual years
        for period_name, start, end in periods:

            period_df = competitor_df[
                (
                    competitor_df["review_day"]
                    >= start
                )
                &
                (
                    competitor_df["review_day"]
                    <= end
                )
            ]

            rows.append(
                calculate_stats(
                    period_df,
                    competitor,
                    period_name,
                    start,
                    end
                )
            )

    return pd.DataFrame(rows)


# ============================================================
# MAIN TRANSFORMATION
# ============================================================

def transform_reviews():

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    print("=" * 70)
    print("SERVERSE REVIEW TRANSFORMATION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Parse TXT
    # --------------------------------------------------------

    df = parse_reviews_txt(
        INPUT_FILE
    )

    # --------------------------------------------------------
    # 2. Clean
    # --------------------------------------------------------

    df = clean_reviews(df)

    print(
        f"\nTotal reviews extracted: {len(df)}"
    )

    # --------------------------------------------------------
    # 3. Save FULL CSV
    # --------------------------------------------------------

    all_reviews_path = os.path.join(
        OUTPUT_FOLDER,
        "all_reviews.csv"
    )

    df.to_csv(
        all_reviews_path,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 4. Filter last 3 years
    # --------------------------------------------------------

    filtered_df = filter_last_3_years(
        df
    )

    filtered_path = os.path.join(
        OUTPUT_FOLDER,
        "reviews_last_3_years.csv"
    )

    filtered_df.to_csv(
        filtered_path,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 5. Before / after statistics
    # --------------------------------------------------------

    impact_df = create_filter_impact(
        df,
        filtered_df
    )

    impact_path = os.path.join(
        OUTPUT_FOLDER,
        "review_filter_impact.csv"
    )

    impact_df.to_csv(
        impact_path,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 6. Review statistics
    # --------------------------------------------------------

    stats_df = create_review_stats(
        filtered_df
    )

    stats_path = os.path.join(
        OUTPUT_FOLDER,
        "review_stats_3y.csv"
    )

    stats_df.to_csv(
        stats_path,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # CONSOLE SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("FILTER IMPACT")
    print("=" * 70)

    print(
        impact_df.to_string(
            index=False
        )
    )

    print()
    print(
        f"Date range: "
        f"{START_DATE.date()} "
        f"to {END_DATE.date()}"
    )

    print()
    print(
        f"Reviews before filter: {len(df)}"
    )

    print(
        f"Reviews after filter: "
        f"{len(filtered_df)}"
    )

    print(
        f"Reviews removed: "
        f"{len(df) - len(filtered_df)}"
    )

    print()
    print("=" * 70)
    print("FILES CREATED")
    print("=" * 70)

    print(all_reviews_path)
    print(filtered_path)
    print(impact_path)
    print(stats_path)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    transform_reviews()
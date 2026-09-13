from pathlib import Path
import os
import time

import numpy as np
import pandas as pd

from google.ads.googleads.client import GoogleAdsClient
from trendspy import Trends


# ============================================================
# CONFIG
# ============================================================

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
    / "data"
    / "market_demand"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

KEYWORD_OUTPUT = (
    OUTPUT_DIR
    / "keyword_planner_demand.csv"
)

TRENDS_OUTPUT = (
    OUTPUT_DIR
    / "google_trends_summary.csv"
)

TRENDS_TIMESERIES_OUTPUT = (
    OUTPUT_DIR
    / "google_trends_timeseries.csv"
)


# Google Ads customer ID
# Example:
# export GOOGLE_ADS_CUSTOMER_ID="1234567890"

CUSTOMER_ID = os.getenv(
    "GOOGLE_ADS_CUSTOMER_ID"
)


# Keyword Planner locations
LOCATIONS = {
    "UK": "United Kingdom",
    "London": "London",
}


# Google Trends
TODAY = pd.Timestamp.today().normalize()

TREND_START = (
    TODAY - pd.DateOffset(years=3)
)

TREND_TIMEFRAME = (
    f"{TREND_START.date()} "
    f"{TODAY.date()}"
)


# We do NOT need Google Trends for every tiny service.
MIN_UK_SEARCHES_FOR_TRENDS = 100

TRENDS_SLEEP_SECONDS = 3

KEYWORD_BATCH_SIZE = 500


# ============================================================
# LOAD SERVICE TAXONOMY
# ============================================================

def load_services():

    df = pd.read_csv(
        TAXONOMY_FILE
    )

    required = {
        "service_family",
        "sub_service",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = (
        df[
            [
                "service_family",
                "sub_service",
            ]
        ]
        .dropna()
        .drop_duplicates()
    )

    # ----------------------------------------
    # GENERAL SERVICES
    # ----------------------------------------

    general = (
        df[
            ["service_family"]
        ]
        .drop_duplicates()
        .copy()
    )

    general[
        "service_level"
    ] = "general"

    general[
        "keyword"
    ] = general[
        "service_family"
    ]

    general[
        "sub_service"
    ] = None


    # ----------------------------------------
    # SUB-SERVICES
    # ----------------------------------------

    specific = (
        df.copy()
    )

    specific[
        "service_level"
    ] = "sub_service"

    specific[
        "keyword"
    ] = specific[
        "sub_service"
    ]


    services = pd.concat(
        [
            general,
            specific,
        ],
        ignore_index=True,
    )

    services[
        "keyword"
    ] = (
        services[
            "keyword"
        ]
        .astype(str)
        .str.strip()
    )

    services = (
        services[
            services[
                "keyword"
            ].ne("")
        ]
        .drop_duplicates(
            [
                "service_family",
                "sub_service",
                "service_level",
                "keyword",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"Service families: "
        f"{general['service_family'].nunique():,}"
    )

    print(
        f"Sub-services: "
        f"{specific['sub_service'].nunique():,}"
    )

    print(
        f"Demand keywords: "
        f"{len(services):,}"
    )

    return services


# ============================================================
# GOOGLE ADS GEO TARGET
# ============================================================

def resolve_google_ads_location(
    client,
    location_name,
):

    service = client.get_service(
        "GeoTargetConstantService"
    )

    request = client.get_type(
        "SuggestGeoTargetConstantsRequest"
    )

    request.locale = "en"
    request.country_code = "GB"

    request.location_names.names.append(
        location_name
    )

    response = (
        service
        .suggest_geo_target_constants(
            request=request
        )
    )

    suggestions = list(
        response
        .geo_target_constant_suggestions
    )

    if not suggestions:
        raise RuntimeError(
            f"No Google Ads location found "
            f"for {location_name}"
        )

    # Prefer exact name match.
    exact = [
        x
        for x in suggestions
        if (
            x.geo_target_constant.name
            .casefold()
            ==
            location_name.casefold()
        )
    ]

    chosen = (
        exact[0]
        if exact
        else suggestions[0]
    )

    geo = (
        chosen
        .geo_target_constant
    )

    print(
        f"{location_name}: "
        f"{geo.name} "
        f"({geo.target_type})"
    )

    return geo.resource_name


# ============================================================
# GOOGLE KEYWORD PLANNER
# ============================================================

def chunks(
    values,
    size,
):

    for i in range(
        0,
        len(values),
        size,
    ):
        yield values[
            i:i + size
        ]


def get_keyword_metrics(
    client,
    customer_id,
    keywords,
    geo_resource,
):

    google_ads_service = (
        client.get_service(
            "GoogleAdsService"
        )
    )

    service = (
        client.get_service(
            "KeywordPlanIdeaService"
        )
    )

    all_rows = []


    for batch in chunks(
        keywords,
        KEYWORD_BATCH_SIZE,
    ):

        request = client.get_type(
            "GenerateKeywordHistoricalMetricsRequest"
        )

        request.customer_id = (
            customer_id
        )

        request.keywords.extend(
            batch
        )

        request.geo_target_constants.append(
            geo_resource
        )

        request.language = (
            google_ads_service
            .language_constant_path(
                "1000"
            )
        )

        request.keyword_plan_network = (
            client.enums
            .KeywordPlanNetworkEnum
            .GOOGLE_SEARCH
        )

        response = (
            service
            .generate_keyword_historical_metrics(
                request=request
            )
        )

        for result in response.results:

            metrics = (
                result.keyword_metrics
            )

            row = {
                "keyword":
                    result.text,

                "avg_monthly_searches":
                    metrics.avg_monthly_searches,

                "competition":
                    metrics.competition.name,

                "competition_index":
                    metrics.competition_index,

                "low_top_page_bid_gbp":
                    (
                        metrics
                        .low_top_of_page_bid_micros
                        / 1_000_000
                    ),

                "high_top_page_bid_gbp":
                    (
                        metrics
                        .high_top_of_page_bid_micros
                        / 1_000_000
                    ),

                "close_variants":
                    " | ".join(
                        result.close_variants
                    ),
            }

            all_rows.append(
                row
            )

    return pd.DataFrame(
        all_rows
    )


def collect_keyword_planner(
    services,
):

    if not CUSTOMER_ID:
        raise RuntimeError(
            "GOOGLE_ADS_CUSTOMER_ID "
            "is not configured."
        )

    client = (
        GoogleAdsClient
        .load_from_storage()
    )

    keywords = (
        services[
            "keyword"
        ]
        .drop_duplicates()
        .tolist()
    )

    outputs = []


    for geography, location_name \
            in LOCATIONS.items():

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"KEYWORD PLANNER: "
            f"{geography}"
        )

        print(
            "=" * 70
        )

        geo_resource = (
            resolve_google_ads_location(
                client,
                location_name,
            )
        )

        metrics = (
            get_keyword_metrics(
                client,
                CUSTOMER_ID,
                keywords,
                geo_resource,
            )
        )

        metrics[
            "geography"
        ] = geography

        outputs.append(
            metrics
        )


    result = pd.concat(
        outputs,
        ignore_index=True,
    )


    # Attach taxonomy back to metrics.
    result[
        "_key"
    ] = (
        result[
            "keyword"
        ]
        .str.casefold()
    )

    taxonomy = services.copy()

    taxonomy[
        "_key"
    ] = (
        taxonomy[
            "keyword"
        ]
        .str.casefold()
    )

    result = result.merge(
        taxonomy,
        on="_key",
        how="left",
        suffixes=(
            "_google",
            "",
        ),
    )

    result.drop(
        columns=["_key"],
        inplace=True,
    )


    result.to_csv(
        KEYWORD_OUTPUT,
        index=False,
    )

    print(
        f"\nSaved: "
        f"{KEYWORD_OUTPUT}"
    )

    return result


# ============================================================
# GOOGLE TRENDS
# ============================================================

def resolve_trends_london(
    trends,
):

    matches = trends.geo(
        find="London"
    )

    matches = [
        item
        for item in matches
        if (
            str(
                item.get(
                    "id",
                    ""
                )
            )
            .startswith(
                "GB"
            )
        )
    ]

    if not matches:
        raise RuntimeError(
            "Could not resolve London "
            "Google Trends geography."
        )

    print(
        "\nGoogle Trends London candidates:"
    )

    for item in matches[:5]:
        print(item)

    return matches[0]["id"]


def trend_summary(
    series,
):

    series = (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .dropna()
    )

    if series.empty:
        return {
            "trend_average": None,
            "trend_latest": None,
            "trend_slope": None,
            "trend_change_pct": None,
        }


    values = (
        series
        .to_numpy(
            dtype=float
        )
    )

    x = np.arange(
        len(values)
    )

    slope = (
        np.polyfit(
            x,
            values,
            1,
        )[0]
        if len(values) > 1
        else 0
    )


    window = min(
        12,
        max(
            1,
            len(values) // 4,
        ),
    )

    start_avg = (
        values[
            :window
        ].mean()
    )

    end_avg = (
        values[
            -window:
        ].mean()
    )


    change_pct = (
        (
            end_avg
            - start_avg
        )
        / start_avg
        * 100
        if start_avg > 0
        else None
    )


    return {
        "trend_average":
            float(
                values.mean()
            ),

        "trend_latest":
            float(
                values[-1]
            ),

        "trend_slope":
            float(
                slope
            ),

        "trend_change_pct":
            (
                float(
                    change_pct
                )
                if change_pct
                is not None
                else None
            ),
    }


def collect_google_trends(
    keyword_data,
):

    trends = Trends()

    london_geo = (
        resolve_trends_london(
            trends
        )
    )

    trend_geographies = {
        "UK": "GB",
        "London": london_geo,
    }


    # ----------------------------------------
    # SHORTLIST
    # ----------------------------------------
    #
    # General categories always stay.
    #
    # Sub-services only continue to Trends
    # when UK Keyword Planner demand >= 100.
    #

    uk = (
        keyword_data[
            keyword_data[
                "geography"
            ]
            == "UK"
        ]
        .copy()
    )

    candidates = uk[
        (
            uk[
                "service_level"
            ]
            == "general"
        )
        |
        (
            uk[
                "avg_monthly_searches"
            ]
            >=
            MIN_UK_SEARCHES_FOR_TRENDS
        )
    ][
        [
            "service_family",
            "sub_service",
            "service_level",
            "keyword",
        ]
    ].drop_duplicates()


    print(
        f"\nGoogle Trends candidates: "
        f"{len(candidates):,}"
    )


    summary_rows = []
    timeseries_rows = []


    for geography, geo \
            in trend_geographies.items():

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"GOOGLE TRENDS: "
            f"{geography}"
        )

        print(
            "=" * 70
        )


        for i, row in enumerate(
            candidates.itertuples(
                index=False
            ),
            start=1,
        ):

            keyword = row.keyword

            print(
                f"[{i}/{len(candidates)}] "
                f"{keyword}"
            )

            try:

                df = (
                    trends
                    .interest_over_time(
                        keyword,
                        timeframe=(
                            TREND_TIMEFRAME
                        ),
                        geo=geo,
                    )
                )

                if df is None \
                        or df.empty:

                    print(
                        "    No Trends data"
                    )

                    continue


                numeric_columns = (
                    df.select_dtypes(
                        include="number"
                    )
                    .columns
                    .tolist()
                )

                if not numeric_columns:
                    continue


                value_col = (
                    keyword
                    if keyword
                    in df.columns
                    else numeric_columns[0]
                )

                values = (
                    df[
                        value_col
                    ]
                )

                metrics = (
                    trend_summary(
                        values
                    )
                )


                summary_rows.append(
                    {
                        "geography":
                            geography,

                        "service_family":
                            row.service_family,

                        "sub_service":
                            row.sub_service,

                        "service_level":
                            row.service_level,

                        "keyword":
                            keyword,

                        **metrics,
                    }
                )


                temp = pd.DataFrame(
                    {
                        "date":
                            df.index,

                        "trend_interest":
                            values.values,
                    }
                )

                temp[
                    "geography"
                ] = geography

                temp[
                    "service_family"
                ] = row.service_family

                temp[
                    "sub_service"
                ] = row.sub_service

                temp[
                    "service_level"
                ] = row.service_level

                temp[
                    "keyword"
                ] = keyword


                timeseries_rows.extend(
                    temp.to_dict(
                        orient="records"
                    )
                )


            except Exception as exc:

                print(
                    f"    FAILED: {exc}"
                )


            time.sleep(
                TRENDS_SLEEP_SECONDS
            )


    summary_df = pd.DataFrame(
        summary_rows
    )

    timeseries_df = pd.DataFrame(
        timeseries_rows
    )


    summary_df.to_csv(
        TRENDS_OUTPUT,
        index=False,
    )

    timeseries_df.to_csv(
        TRENDS_TIMESERIES_OUTPUT,
        index=False,
    )


    print(
        f"\nSaved: "
        f"{TRENDS_OUTPUT}"
    )

    print(
        f"Saved: "
        f"{TRENDS_TIMESERIES_OUTPUT}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "SEVERSE MARKET DEMAND COLLECTION"
    )

    print(
        "=" * 70
    )


    services = (
        load_services()
    )


    keyword_data = (
        collect_keyword_planner(
            services
        )
    )


    collect_google_trends(
        keyword_data
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "DEMAND COLLECTION COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
import os

from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException


CUSTOMER_ID = os.getenv("GOOGLE_ADS_CUSTOMER_ID")

KEYWORDS = [
    "plumber",
    "electrician",
    "hairdresser",
]


def get_uk_geo(client):

    geo_service = client.get_service(
        "GeoTargetConstantService"
    )

    request = client.get_type(
        "SuggestGeoTargetConstantsRequest"
    )

    request.locale = "en"
    request.country_code = "GB"

    request.location_names.names.append(
        "United Kingdom"
    )

    response = (
        geo_service
        .suggest_geo_target_constants(
            request=request
        )
    )

    for suggestion in response.geo_target_constant_suggestions:

        geo = suggestion.geo_target_constant

        if geo.name == "United Kingdom":

            print(
                f"Using geography: "
                f"{geo.name}"
            )

            return geo.resource_name

    raise RuntimeError(
        "United Kingdom geo target not found."
    )


def main():

    if not CUSTOMER_ID:
        raise RuntimeError(
            "GOOGLE_ADS_CUSTOMER_ID is not set."
        )

    print("Loading Google Ads credentials...")

    client = (
        GoogleAdsClient
        .load_from_storage()
    )

    print("Credentials loaded.")

    uk_geo = get_uk_geo(
        client
    )

    google_ads_service = (
        client.get_service(
            "GoogleAdsService"
        )
    )

    keyword_service = (
        client.get_service(
            "KeywordPlanIdeaService"
        )
    )

    request = client.get_type(
        "GenerateKeywordHistoricalMetricsRequest"
    )

    request.customer_id = CUSTOMER_ID

    request.keywords.extend(
        KEYWORDS
    )

    request.geo_target_constants.append(
        uk_geo
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

    print(
        "\nRequesting keyword data...\n"
    )

    try:

        response = (
            keyword_service
            .generate_keyword_historical_metrics(
                request=request
            )
        )

        for result in response.results:

            metrics = result.keyword_metrics

            print(
                "=" * 50
            )

            print(
                f"Keyword: {result.text}"
            )

            print(
                "Average monthly searches:",
                metrics.avg_monthly_searches
            )

            print(
                "Competition:",
                metrics.competition.name
            )

            print(
                "Competition index:",
                metrics.competition_index
            )

            print(
                "Low top-page bid:",
                metrics.low_top_of_page_bid_micros
                / 1_000_000
            )

            print(
                "High top-page bid:",
                metrics.high_top_of_page_bid_micros
                / 1_000_000
            )

    except GoogleAdsException as error:

        print("\nGOOGLE ADS API ERROR\n")

        print(
            "Request ID:",
            error.request_id
        )

        print(
            "Status:",
            error.error.code().name
        )

        for failure in error.failure.errors:

            print(
                failure.error_code
            )

            print(
                failure.message
            )


if __name__ == "__main__":
    main()
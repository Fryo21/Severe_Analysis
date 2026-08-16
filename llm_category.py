import os
import json
from enum import Enum

from openai import OpenAI
from pydantic import BaseModel


# ============================================================
# CONFIG
# ============================================================

MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-5.6"
)

client = OpenAI()


# ============================================================
# CATEGORY DECISION
# ============================================================

class CategoryAction(str, Enum):
    EXISTING = "existing"
    NEW = "new"


class CategoryDecision(BaseModel):
    action: CategoryAction
    category_id: int | None
    category_name: str


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a review categorisation system.

Your task is to classify ONE customer review into a category.

You will receive:

1. A customer review.
2. The current list of existing categories.

Each existing category contains:
- category_id
- category_name

Your goal is to maintain a clean and reusable category taxonomy.


RULES:

1. First examine the existing category list.

2. If the review clearly belongs to an existing category:
   - action = "existing"
   - return that exact category_id
   - return that exact category_name

3. If no existing category appropriately represents the review:
   - action = "new"
   - category_id = null
   - propose a concise new category_name

4. If the category list is empty:
   - create a new category
   - action = "new"
   - category_id = null

5. Reuse existing categories whenever the underlying subject is
   substantially the same.

6. Do not create duplicate or near-duplicate categories.

7. Categories should describe WHAT the review is about,
   not whether the review is positive or negative.

For example:

Good:
- Lead Quality
- Customer Support
- Pricing
- App Experience
- Booking Process

Bad:
- Bad Lead Quality
- Great Customer Support
- Negative Experience
- Positive Review

Sentiment is handled separately by another system.

8. Keep category names short, general, and suitable for
   analytics and visualisation.

9. Do not invent details that are not present in the review.
"""


# ============================================================
# VALIDATE CATEGORY LIST
# ============================================================

def validate_categories(categories: list[dict]):

    for category in categories:

        if "category_id" not in category:
            raise ValueError(
                "Every category must contain category_id."
            )

        if "category_name" not in category:
            raise ValueError(
                "Every category must contain category_name."
            )


# ============================================================
# CATEGORISE ONE REVIEW
# ============================================================

def categorize_review(
    review: str,
    categories: list[dict]
) -> dict:

    """
    Categorise one review using the existing category list.

    If an appropriate category exists, reuse it.

    If no appropriate category exists, create a new category
    with the next available category ID.

    Args:
        review:
            Customer review text.

        categories:
            Current category list.

            Example:
            [
                {
                    "category_id": 1,
                    "category_name": "Lead Quality"
                }
            ]

    Returns:
        dict containing:
            category_id
            category_name
            categories
            created_new_category
    """

    # --------------------------------------------------------
    # VALIDATE REVIEW
    # --------------------------------------------------------

    if not isinstance(review, str):
        raise TypeError(
            "Review must be a string."
        )

    review = review.strip()

    if not review:
        raise ValueError(
            "Review cannot be empty."
        )

    # --------------------------------------------------------
    # VALIDATE CATEGORY LIST
    # --------------------------------------------------------

    if not isinstance(categories, list):
        raise TypeError(
            "Categories must be a list."
        )

    validate_categories(categories)

    # Make a copy so we do not unexpectedly modify
    # the original list.
    updated_categories = [
        category.copy()
        for category in categories
    ]

    # --------------------------------------------------------
    # PREPARE MESSAGE
    # --------------------------------------------------------

    categories_json = json.dumps(
        updated_categories,
        indent=2,
        ensure_ascii=False
    )

    user_message = f"""
REVIEW:

{review}


CURRENT CATEGORIES:

{categories_json}
"""

    # --------------------------------------------------------
    # SEND TO LLM
    # --------------------------------------------------------

    if not MODEL:
        raise ValueError(
            "OPENAI_MODEL is not configured."
        )

    response = client.responses.parse(

        model=MODEL,

        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_message
            }
        ],

        text_format=CategoryDecision
    )

    decision = response.output_parsed

    if decision is None:
        raise ValueError(
            "No structured category result returned."
        )

    # ========================================================
    # EXISTING CATEGORY
    # ========================================================

    if decision.action == CategoryAction.EXISTING:

        if decision.category_id is None:
            raise ValueError(
                "LLM selected an existing category "
                "without returning a category_id."
            )

        category = next(
            (
                item
                for item in updated_categories
                if item["category_id"]
                == decision.category_id
            ),
            None
        )

        if category is None:
            raise ValueError(
                f"LLM returned unknown category ID: "
                f"{decision.category_id}"
            )

        return {
            "category_id":
                category["category_id"],

            "category_name":
                category["category_name"],

            "categories":
                updated_categories,

            "created_new_category":
                False
        }

    # ========================================================
    # NEW CATEGORY
    # ========================================================

    new_category_name = (
        decision.category_name.strip()
    )

    if not new_category_name:
        raise ValueError(
            "LLM proposed an empty category name."
        )

    # --------------------------------------------------------
    # EXTRA DUPLICATE PROTECTION
    # --------------------------------------------------------

    for category in updated_categories:

        if (
            category["category_name"]
            .strip()
            .lower()
            == new_category_name.lower()
        ):

            return {
                "category_id":
                    category["category_id"],

                "category_name":
                    category["category_name"],

                "categories":
                    updated_categories,

                "created_new_category":
                    False
            }

    # --------------------------------------------------------
    # CREATE NEXT CATEGORY ID
    # --------------------------------------------------------

    if updated_categories:

        next_id = max(
            category["category_id"]
            for category
            in updated_categories
        ) + 1

    else:

        next_id = 1

    # --------------------------------------------------------
    # ADD CATEGORY
    # --------------------------------------------------------

    new_category = {
        "category_id": next_id,
        "category_name": new_category_name
    }

    updated_categories.append(
        new_category
    )

    # --------------------------------------------------------
    # RETURN RESULT
    # --------------------------------------------------------

    return {
        "category_id":
            next_id,

        "category_name":
            new_category_name,

        "categories":
            updated_categories,

        "created_new_category":
            True
    }
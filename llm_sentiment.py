import os
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


# ============================================================
# OPENAI CLIENT
# ============================================================

client = OpenAI()


# ============================================================
# SENTIMENT TYPES
# ============================================================

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


# ============================================================
# STRUCTURED LLM OUTPUT
# ============================================================

class SentimentResult(BaseModel):
    sentiment: Sentiment


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a sentiment classification system.

Your task is to analyse ONE customer review and classify its
overall sentiment into exactly one of these categories:

- positive
- negative
- neutral

Definitions:

POSITIVE:
The reviewer is mainly expressing satisfaction, approval,
praise, recommendation, or a good experience.

NEGATIVE:
The reviewer is mainly expressing dissatisfaction, criticism,
frustration, disappointment, or a bad experience.

NEUTRAL:
The review is mainly factual, unclear, balanced, or does not
express a clearly positive or negative opinion.

Rules:

1. Analyse the review text itself.

2. Return only the overall sentiment represented by the review.

3. Do not infer information that is not present in the review.

4. If the review contains both positive and negative statements,
   classify it according to the dominant sentiment.

5. If neither sentiment clearly dominates, classify it as neutral.

6. Do not use star ratings or other metadata to determine sentiment.
   Base the classification only on the written review.
"""


# ============================================================
# CLASSIFY ONE REVIEW
# ============================================================

def classify_sentiment(review: str) -> str:
    """
    Classify one review as positive, negative, or neutral.

    Args:
        review (str):
            The review text to analyse.

    Returns:
        str:
            "positive", "negative", or "neutral"
    """

    if not isinstance(review, str):
        raise TypeError(
            "Review must be a string."
        )

    review = review.strip()

    if not review:
        raise ValueError(
            "Review cannot be empty."
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
                "content": review
            }
        ],

        text_format=SentimentResult
    )

    result = response.output_parsed

    if result is None:
        raise ValueError(
            "No structured sentiment result returned."
        )

    return result.sentiment.value
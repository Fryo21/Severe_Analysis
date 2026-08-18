from pathlib import Path
import html
import re

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "processed-data"
    / "competitor-research"
    / "negative_reviews_2024_2026.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed-data"
    / "competitor-research"
    / "complaint-analysis-final"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

COMPETITORS = [
    "Bark",
    "Checkatrade",
    "Yell",
    "Yelp"
]


# ============================================================
# TEXT CLEANING
# ============================================================

MOJIBAKE_REPLACEMENTS = {
    "‚Äô": "'",
    "‚Äò": "'",
    "‚Äú": '"',
    "‚Äù": '"',
    "‚Äì": "-",
    "‚Äî": "-",
    "¬£": "£",
    "‚Ç¨": "€",
    "√©": "e",
    "√®": "e",
    "√†": "a",
    "√ß": "c",
    "√º": "u",
    "√∂": "o",
    "√§": "a",
}


def clean_text(text):

    if pd.isna(text):
        return ""

    text = html.unescape(
        str(text)
    )

    for bad, replacement in (
        MOJIBAKE_REPLACEMENTS.items()
    ):
        text = text.replace(
            bad,
            replacement
        )

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# REGEX HELPERS
# ============================================================

def matches(text, pattern):

    return bool(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )
    )


def matches_any(
    text,
    patterns
):

    return any(
        matches(
            text,
            pattern
        )
        for pattern in patterns
    )


def weighted_score(
    text,
    rules
):

    score = 0

    for pattern, weight in rules:

        if matches(
            text,
            pattern
        ):
            score += weight

    return score


# ============================================================
# PLATFORM RELEVANCE
# ============================================================

# Some Trustpilot results for "Bark" refer to another
# product/company such as Bark Phone rather than Bark.com.

BARK_WRONG_PRODUCT_PATTERNS = [
    r"\bbark phone\b",
    r"\bparent app\b",
    r"\bparental control\b",
    r"\bchild(?:'s)? phone\b",
    r"\bmy son(?:'s)? phone\b",
    r"\bmy daughter(?:'s)? phone\b",
    r"\bphone carrier\b",
    r"\bkeep my child safe\b",
    r"\bscreen time\b",
]


# Yelp needs a stricter relevance check because some reviews
# describe the restaurant/store itself rather than Yelp.

YELP_PLATFORM_PATTERNS = [

    (
        r"\byelp\b.{0,100}\b"
        r"(?:review|reviews|rating|ratings|account|listing|"
        r"business page|advertis|ads?|sales|support|"
        r"customer service|app|website|filter|moderation|"
        r"elite|charged|billing|recommend)"
    ),

    (
        r"\b(?:review|reviews|rating|ratings|account|listing|"
        r"business page|advertis|ads?|sales|support|"
        r"customer service|app|website|filter|moderation|"
        r"elite|charged|billing)\b"
        r".{0,100}\byelp\b"
    ),

    r"\bremoved my review\b",
    r"\bdeleted my review\b",
    r"\bwould(?:n't| not) publish my review\b",
    r"\bmy business page\b",
    r"\bbusiness listing\b",
    r"\breview filter\b",
    r"\byelp elite\b",

    (
        r"\byelp (?:is|was|are|were|seems|feels) "
        r"(?:a )?"
        r"(?:scam|terrible|awful|useless|bad|"
        r"dishonest|untrustworthy)\b"
    ),
]


YELP_THIRD_PARTY_ONLY_PATTERNS = [

    (
        r"\b(?:restaurant|cafe|hotel|car rental|dealership|"
        r"mechanic|dentist|doctor|salon|bar|store|shop)\b"
    ),

    (
        r"\b(?:food|meal|burger|pizza|waiter|waitress|"
        r"room|rental car)\b"
    ),
]


def classify_relevance(row):

    competitor = str(
        row["competitor"]
    ).strip()

    text = row[
        "clean_text"
    ]

    if not text:

        return (
            "Review",
            "Empty text"
        )

    # --------------------------------------------------------
    # Bark
    # --------------------------------------------------------

    if competitor == "Bark":

        if matches_any(
            text,
            BARK_WRONG_PRODUCT_PATTERNS
        ):

            return (
                "Exclude",
                "Different Bark product/company"
            )

        return (
            "Relevant",
            "Bark marketplace review"
        )

    # --------------------------------------------------------
    # Yelp
    # --------------------------------------------------------

    if competitor == "Yelp":

        if matches_any(
            text,
            YELP_PLATFORM_PATTERNS
        ):

            return (
                "Relevant",
                "Yelp platform signal found"
            )

        if matches_any(
            text,
            YELP_THIRD_PARTY_ONLY_PATTERNS
        ):

            return (
                "Exclude",
                "Appears to review a third-party business, not Yelp"
            )

        return (
            "Review",
            "Yelp relevance is ambiguous"
        )

    # Checkatrade and Yell reviews discussing the platform
    # or a provider obtained through the platform are useful
    # for marketplace weakness research.

    return (
        "Relevant",
        "Marketplace/platform review"
    )


# ============================================================
# USER TYPE CLASSIFICATION
# ============================================================

PROVIDER_RULES = [

    (
        r"\bas (?:a|an) "
        r"(?:supplier|service provider|professional|"
        r"freelancer|seller|tradesman|tradesperson|"
        r"contractor|business owner|business)\b",
        10
    ),

    (
        r"\bi(?: am|'m) (?:a|an) "
        r"(?:supplier|service provider|professional|"
        r"freelancer|seller|tradesman|tradesperson|"
        r"contractor|business owner)\b",
        10
    ),

    (
        r"\bmy business\b|"
        r"\bour business\b|"
        r"\bmy company\b|"
        r"\bour company\b",
        5
    ),

    (
        r"\bsmall business\b|"
        r"\bself[- ]employed\b",
        5
    ),

    (
        r"\b(?:buy|bought|buying|purchase|purchased|"
        r"purchasing|pay|paid|paying) "
        r"(?:for )?(?:bark )?"
        r"(?:credits?|leads?)\b",
        10
    ),

    (
        r"\bcredits?\b",
        4
    ),

    (
        r"\bleads?\b",
        2
    ),

    (
        r"\bprospects?\b",
        4
    ),

    (
        r"\breturn on investment\b|"
        r"\broi\b|"
        r"\bconversion rate\b",
        7
    ),

    (
        r"\b(?:advertis(?:e|ed|ing)|marketing) "
        r"(?:my|our|the) business\b",
        8
    ),

    (
        r"\b(?:advertising|marketing) "
        r"(?:package|contract|campaign|account|service)\b",
        7
    ),

    (
        r"\byell "
        r"(?:advertising|website|contract|package|sales)\b",
        8
    ),

    (
        r"\byelp "
        r"(?:advertising|ads?|sales|business page|"
        r"business account)\b",
        8
    ),

    (
        r"\bmy (?:business )?"
        r"(?:listing|profile|page)\b",
        6
    ),

    (
        r"\bmember(?:ship)? "
        r"(?:fee|fees|subscription|account)\b",
        5
    ),

    (
        r"\b(?:win|won|get|getting|land|landing) "
        r"(?:the )?"
        r"(?:job|jobs|client|clients|customer|"
        r"customers|work)\b",
        4
    ),
]


CUSTOMER_RULES = [

    (
        r"\bi (?:hired|booked|employed|used) "
        r"(?:a|an|the|this)?\s*"
        r"(?:company|contractor|tradesman|tradesperson|"
        r"provider|professional|builder|roofer|plumber|"
        r"electrician|gardener|cleaner|vendor|painter|"
        r"decorator)\b",
        10
    ),

    (
        r"\b(?:tradesman|tradesperson|contractor|"
        r"company|provider|professional) "
        r"(?:i|we) hired\b",
        10
    ),

    (
        r"\bmy "
        r"(?:contractor|tradesman|tradesperson|builder|"
        r"roofer|plumber|electrician|gardener|cleaner|"
        r"vendor|painter|decorator)\b",
        10
    ),

    (
        r"\b(?:contractor|tradesman|tradesperson|"
        r"company|provider|vendor) "
        r"(?:suggested|recommended|provided|listed) "
        r"(?:by|on)\b",
        9
    ),

    (
        r"\bi chose "
        r"(?:a|the) "
        r"(?:tradesperson|tradesman|contractor|"
        r"company|provider) "
        r"through\b",
        10
    ),

    (
        r"\bi (?:am |was )?looking for\b",
        7
    ),

    (
        r"\bi (?:need|needed|want|wanted) "
        r"(?:a|an|someone|somebody)\b",
        6
    ),

    (
        r"\bi "
        r"(?:posted|submitted|put up|put in|placed) "
        r"(?:a|my|the)?\s*"
        r"(?:job|request|enquiry|inquiry)\b",
        8
    ),

    (
        r"\bmy "
        r"(?:job|request|enquiry|inquiry)\b",
        5
    ),

    (
        r"\bi asked for (?:a )?quote\b|"
        r"\bi requested (?:a )?quote\b",
        7
    ),

    (
        r"\bmy "
        r"(?:home|house|property|garden|roof|"
        r"kitchen|bathroom|drive|patio)\b",
        2
    ),

    (
        r"\bmy deposit\b|"
        r"\bpaid "
        r"(?:the|a) "
        r"(?:contractor|tradesman|company|provider)\b",
        8
    ),

    (
        r"\bno "
        r"(?:cleaner|plumber|electrician|builder|gardener|"
        r"roofer|vendor|provider|company|contractor|"
        r"tradesman|tradesperson)s? "
        r"(?:called|contacted|replied|responded)\b",
        10
    ),

    (
        r"\b(?:vendors?|providers?|companies|contractors?|"
        r"trades(?:men|people|persons?)) "
        r"(?:contacted|called|quoted) me\b",
        5
    ),

    (
        r"\bwork "
        r"(?:done|carried out|completed) "
        r"(?:for|at) "
        r"(?:me|my)\b",
        5
    ),

    (
        r"\bguarantee\b"
        r".{0,80}"
        r"\b(?:job|work|trade|tradesperson|contractor)\b",
        4
    ),
]


def classify_user_type(
    text,
    competitor
):

    provider_score = weighted_score(
        text,
        PROVIDER_RULES
    )

    customer_score = weighted_score(
        text,
        CUSTOMER_RULES
    )

    # --------------------------------------------------------
    # Yell
    # --------------------------------------------------------

    if competitor == "Yell":

        if matches_any(
            text,
            [
                r"\badvertis(?:e|ed|ing)\b",
                r"\bmarketing\b",
                (
                    r"\bwebsite "
                    r"(?:package|service|contract|built|design)\b"
                ),
                r"\bsales rep\b|\bsales team\b",
                r"\bbusiness account\b",
            ]
        ):

            provider_score += 6

    # --------------------------------------------------------
    # Yelp
    # --------------------------------------------------------

    if competitor == "Yelp":

        if matches_any(
            text,
            [
                r"\bmy business\b",
                r"\bbusiness owner\b",
                r"\bbusiness page\b",
                r"\badvertis(?:e|ed|ing)\b",
                r"\bsales rep\b|\bsales team\b",
            ]
        ):

            provider_score += 7

        elif matches_any(
            text,
            [
                r"\bmy review\b",
                r"\breviews? "
                r"(?:removed|deleted|filtered)\b",
                r"\bi use yelp\b|"
                r"\bi used yelp\b",
                r"\bfound "
                r"(?:a|the) business\b",
            ]
        ):

            customer_score += 5

    # --------------------------------------------------------
    # Checkatrade
    # --------------------------------------------------------

    if competitor == "Checkatrade":

        if matches_any(
            text,
            [
                r"\btrade member\b",
                r"\btradesperson account\b",
                r"\bmy checkatrade profile\b",
                (
                    r"\bmembership\b"
                    r".{0,40}"
                    r"\b(?:fee|cost|renew|subscription)\b"
                ),
            ]
        ):

            provider_score += 7

        elif matches_any(
            text,
            [
                r"\btradesman hired\b",
                r"\btradesperson hired\b",
                r"\bi hired\b",
                r"\bi chose a tradesperson\b",
                r"\bmy property\b",
            ]
        ):

            customer_score += 5

    highest = max(
        provider_score,
        customer_score
    )

    difference = abs(
        provider_score
        - customer_score
    )

    if highest < 4:

        return (
            "Unknown",
            "Low",
            provider_score,
            customer_score
        )

    if provider_score > customer_score:

        confidence = (
            "High"
            if difference >= 5
            else "Medium"
        )

        return (
            "Service Provider / Business",
            confidence,
            provider_score,
            customer_score
        )

    if customer_score > provider_score:

        confidence = (
            "High"
            if difference >= 5
            else "Medium"
        )

        return (
            "Customer",
            confidence,
            provider_score,
            customer_score
        )

    return (
        "Unknown",
        "Low",
        provider_score,
        customer_score
    )


# ============================================================
# FINAL WEAKNESS CATEGORIES
# ============================================================

THEME_PATTERNS = {

    # ========================================================
    # CUSTOMER SIDE
    # ========================================================

    "Finding a provider / low response": [

        (
            r"\bno "
            r"(?:one|one ever|companies?|providers?|vendors?|"
            r"contractors?|trades(?:men|people|persons?)|"
            r"cleaners?|plumbers?|electricians?|builders?|"
            r"gardeners?|roofers?) "
            r"(?:called|contacted|replied|responded)\b"
        ),

        (
            r"\bnever received "
            r"(?:a|one|any) repl(?:y|ies)\b"
        ),

        (
            r"\bno repl(?:y|ies) from "
            r"(?:potential )?"
            r"(?:providers?|companies|contractors?|"
            r"trades(?:men|people))\b"
        ),

        (
            r"\bno responses? from "
            r"(?:my )?"
            r"(?:request|enquiry|inquiry)\b"
        ),

        r"\brequested replies?.{0,40}heard nothing\b",

        (
            r"\b(?:companies|providers|contractors?) "
            r"(?:failed to|didn't|did not) respond\b"
        ),

        (
            r"\bfew "
            r"(?:applicants|responses|providers|companies)\b"
        ),
    ],


    "Poor matching / location": [

        r"\bpoor match(?:ing)?\b",
        r"\bbad match(?:ing)?\b",
        r"\bnot relevant\b",
        r"\birrelevant\b",
        r"\bwrong service\b",
        r"\bwrong provider\b",
        r"\bnot suitable\b",
        r"\boutside (?:the|my|our) area\b",
        r"\bwell outside (?:the|my|our) area\b",
        r"\bnot local\b",
        r"\bno local responses?\b",

        (
            r"\b(?:[1-9][0-9]{1,3})\s*"
            r"(?:miles?|km) away\b"
        ),

        r"\bmiles away\b",

        (
            r"\basked for "
            r".{0,50}"
            r"(?:but|instead) "
            r"(?:got|received)\b"
        ),

        (
            r"\bnot what "
            r"(?:i|we) "
            r"(?:asked for|wanted|needed)\b"
        ),
    ],


    "Provider quality": [

        r"\bpoor workmanship\b",
        r"\bbad workmanship\b",
        r"\bshoddy\b",
        r"\bpoor quality work\b",
        r"\bbad job\b",
        r"\bvery bad job\b",
        r"\bunfinished work\b",
        r"\bfailed to complete\b",
        r"\bdidn't complete\b",
        r"\bdid not complete\b",
        r"\bwork deteriorated\b",
        r"\bdamaged (?:my|the)\b",
        r"\bbroke (?:my|the)\b",

        (
            r"\bripped off by "
            r"(?:the|a|this) "
            r"(?:contractor|tradesman|company|provider)\b"
        ),

        (
            r"\bnever showed\b|"
            r"\bdidn't show\b|"
            r"\bdid not show\b|"
            r"\bno[- ]show\b|"
            r"\bnever turned up\b|"
            r"\bdidn't turn up\b|"
            r"\bdid not turn up\b|"
            r"\bfailed to turn up\b|"
            r"\bghosted (?:me|us)\b"
        ),
    ],


    "Provider verification / safety": [

        r"\bnot vetted\b",
        r"\bnot verified\b",
        r"\bneeds? to vet\b",
        r"\bshould (?:have )?vet(?:ted)?\b",
        r"\bwhat checks did\b",
        r"\bbackground check\b",
        r"\bcrb check\b",
        r"\bpublic liability\b",
        r"\bscammer\b",

        (
            r"\bfraudulent "
            r"(?:business|contractor|provider)\b"
        ),

        r"\busing another name\b",
        r"\bnot the same company\b",
        r"\bno basic details\b",

        (
            r"\blost "
            r"(?:£|\$|€)\s*[0-9]"
            r".{0,80}"
            r"\b(?:contractor|provider|business)\b"
        ),
    ],


    "Too many provider contacts / spam": [

        r"\btoo many (?:calls|texts|messages|emails)\b",
        r"\bbombarded\b",
        r"\binundated\b",
        r"\bflood of (?:emails|calls|messages)\b",
        r"\bconstant calls\b",
        r"\brelentless calls\b",
        r"\bphone (?:was|is) blowing up\b",

        (
            r"\b(?:20|30|40|50|60|70)"
            r"[–-]?"
            r"(?:20|30|40|50|60|70)? "
            r"calls\b"
        ),
    ],


    "Pricing / quote transparency": [

        r"\bfar too expensive\b",
        r"\btoo expensive\b",
        r"\boverpriced\b",
        r"\bprice increased\b",
        r"\bcost increased\b",
        r"\bfinal cost increased\b",

        (
            r"\bmore than "
            r"(?:the )?"
            r"(?:agreed|quoted) price\b"
        ),

        r"\bovercharged\b",

        (
            r"\bquote"
            r".{0,25}"
            r"(?:too high|expensive)\b"
        ),

        (
            r"\bprice"
            r".{0,40}"
            r"(?:without|no prior) discussion\b"
        ),

        r"\bprice (?:was|is) not clear\b",
    ],


    "Review trust": [

        r"\bfake reviews?\b",
        r"\breviews? (?:are|were) fake\b",
        r"\bremoved my review\b",
        r"\bdeleted my review\b",
        r"\bwon't publish my review\b",
        r"\bwill not publish my review\b",
        r"\bwouldn't publish my review\b",
        r"\bonly positive reviews?\b",
        r"\breview manipulation\b",
        r"\bcannot trust (?:the )?reviews?\b",
        r"\bcan't trust (?:the )?reviews?\b",
        r"\bmy review was not calculated\b",
        r"\breview filter\b",
    ],


    "Dispute protection": [

        r"\bno protection\b",
        r"\bno redress\b",
        r"\brefused to help\b",
        r"\bwouldn't help\b",
        r"\bwould not help\b",
        r"\bdid nothing about\b",

        (
            r"\bcomplaint"
            r".{0,40}"
            r"(?:ignored|unresolved|not resolved)\b"
        ),

        (
            r"\bdispute"
            r".{0,40}"
            r"(?:ignored|unresolved|not resolved)\b"
        ),

        (
            r"\bguarantee"
            r".{0,50}"
            r"(?:not honoured|not honored|not respected|"
            r"refused|failed)\b"
        ),

        r"\bno assistance\b",
        r"\bno responsibility\b",
        r"\bno escalation\b",
    ],


    # ========================================================
    # PROVIDER / BUSINESS SIDE
    # ========================================================

    "Poor lead quality": [

        r"\bpoor(?: quality)? leads?\b",
        r"\bbad leads?\b",
        r"\blow[- ]quality leads?\b",
        r"\blow[- ]value leads?\b",
        r"\buseless leads?\b",
        r"\bquality of (?:the )?leads?\b",
        r"\bleads? (?:are|were) poor\b",

        (
            r"\bnot serious "
            r"(?:customers?|buyers?|prospects?)\b"
        ),

        (
            r"\bjust "
            r"(?:curious|looking for prices|shopping around|"
            r"getting quotes)\b"
        ),

        (
            r"\bno intention of "
            r"(?:hiring|proceeding|buying)\b"
        ),

        r"\bunrealistic (?:budget|budgets)\b",
    ],


    "Fake / invalid leads": [

        r"\bfake leads?\b",
        r"\bbogus leads?\b",
        r"\bfalse leads?\b",
        r"\bghost accounts?\b",
        r"\bleads? (?:are|were) not legitimate\b",
        r"\bleads? (?:are|were) fake\b",

        (
            r"\bfake "
            r"(?:phone|number|email|customer|enquiry|inquiry)\b"
        ),

        (
            r"\binvalid "
            r"(?:phone|number|email|contact)\b"
        ),

        (
            r"\bincorrect "
            r"(?:phone|number|email|contact information)\b"
        ),

        r"\bwrong (?:phone|number|email)\b",

        (
            r"\bnumber "
            r"(?:does not|doesn't|did not|didn't) work\b"
        ),

        r"\bdead (?:number|phone line)\b",

        (
            r"\bnever "
            r"(?:posted|submitted) "
            r"(?:the|a) "
            r"(?:job|request)\b"
        ),

        (
            r"\bdoesn't remember "
            r"(?:posting|submitting)\b"
        ),

        (
            r"\bdid not "
            r"(?:post|submit) "
            r"(?:the|a) "
            r"(?:job|request)\b"
        ),
    ],


    "Unresponsive leads": [

        (
            r"\bleads? "
            r"(?:do not|don't|did not|didn't|never) "
            r"(?:reply|respond|answer)\b"
        ),

        (
            r"\bcustomers? "
            r"(?:do not|don't|did not|didn't|never) "
            r"(?:reply|respond|answer)\b"
        ),

        (
            r"\bclients? "
            r"(?:do not|don't|did not|didn't|never) "
            r"(?:reply|respond|answer)\b"
        ),

        (
            r"\bprospects? "
            r"(?:do not|don't|did not|didn't|never) "
            r"(?:reply|respond|answer)\b"
        ),

        r"\bmessages? (?:were |are )?(?:unread|ignored)\b",

        (
            r"\bphone calls? "
            r"(?:were )?"
            r"(?:not answered|unanswered)\b"
        ),

        r"\bzero responses?\b",
        r"\bradio silence\b",
    ],


    "Poor ROI / expensive leads": [

        (
            r"\bcredits? "
            r"(?:cost|costs|are expensive|were expensive)\b"
        ),

        (
            r"\bleads? "
            r"(?:cost|costs|are expensive|were expensive)\b"
        ),

        (
            r"\bpay(?:ing|ed)? "
            r"(?:just )?"
            r"to (?:contact|message|unlock)\b"
        ),

        r"\bpay(?:ing|ed)? for (?:a )?lead\b",
        r"\bno return on investment\b",
        r"\bpoor return on investment\b",
        r"\bnot good value for money\b",

        (
            r"\bnot worth "
            r"(?:the )?"
            r"(?:money|cost|credits?)\b"
        ),

        r"\bwaste of (?:time and )?money\b",
        r"\bzero (?:jobs?|clients?|customers?)\b",

        (
            r"\bspent "
            r".{0,30}"
            r"(?:£|\$|€|credits?)"
            r".{0,80}"
            r"(?:no jobs?|nothing|zero|no return|no work)\b"
        ),

        (
            r"\badvertising"
            r".{0,70}"
            r"(?:no leads?|no customers?|no return|"
            r"waste|not worth|lost business)\b"
        ),

        (
            r"\bspent "
            r".{0,40}"
            r"(?:advertising|marketing)"
            r".{0,60}"
            r"(?:waste|no return|no leads?|no customers?)\b"
        ),
    ],


    "Too much provider competition": [

        r"\boversaturated\b",
        r"\bsaturated\b",

        (
            r"\btoo many "
            r"(?:providers?|professionals?|trades(?:men|people))\b"
        ),

        r"\btoo much competition\b",

        (
            r"\balready spoken to "
            r"(?:two|three|2|3|several|multiple)\b"
        ),

        (
            r"\bonly (?:five|5) "
            r"(?:people|professionals|companies) "
            r"can contact\b"
        ),

        r"\bsold to (?:everyone|multiple)\b",

        (
            r"\bif you don't respond in "
            r"(?:the )?"
            r"(?:same minute|seconds|minutes)\b"
        ),
    ],


    "Billing / refunds": [

        r"\brefund\b",
        r"\brefunded\b",
        r"\brefunds\b",
        r"\bmoney back\b",
        r"\bcredit refund\b",
        r"\brefused to refund\b",
        r"\bwouldn't refund\b",
        r"\bwould not refund\b",
        r"\bnot refunding\b",
        r"\bno refund\b",
        r"\bunexpected charges?\b",
        r"\bcharged without\b",
        r"\bcharged again\b",
        r"\bautomatic renewal\b",
        r"\bauto[- ]renewal\b",
        r"\brenewed without\b",
        r"\bdirect debit\b",
        r"\bmoney (?:was )?taken\b",
        r"\bdebt collector\b|"
        r"\bdebt-collector\b",
    ],


    # ========================================================
    # BOTH SIDES
    # ========================================================

    "Customer support": [

        (
            r"\bcustomer "
            r"(?:service|support)"
            r".{0,50}"
            r"(?:poor|bad|terrible|awful|rude|unhelpful|"
            r"ignored|no response|useless)\b"
        ),

        (
            r"\bsupport "
            r"(?:team )?"
            r".{0,50}"
            r"(?:rude|unhelpful|ignored|no response|"
            r"did nothing|useless)\b"
        ),

        r"\bno meaningful (?:response|help)\b",
        r"\bno escalation path\b",
        r"\bcopy[- ]and[- ]paste (?:response|reply)\b",
        r"\bgeneric (?:response|reply)\b",

        (
            r"\bcontacted "
            r".{0,25}"
            r"support"
            r".{0,50}"
            r"(?:no response|nothing|ignored|did nothing)\b"
        ),

        (
            r"\bno response from "
            r"(?:customer service|support)\b"
        ),

        (
            r"\bimpossible to "
            r"(?:contact|reach) "
            r"(?:them|support|customer service)\b"
        ),
    ],


    "Account / cancellation": [

        (
            r"\baccount "
            r"(?:was |is |got )?"
            r"(?:suspended|blocked|locked|closed|removed)\b"
        ),

        (
            r"\bprofile "
            r"(?:was |is )?"
            r"(?:offline|removed|blocked)\b"
        ),

        (
            r"\bcannot log ?in\b|"
            r"\bcan't log ?in\b|"
            r"\bunable to log ?in\b|"
            r"\bimpossible to log ?in\b"
        ),

        r"\bcannot access (?:my )?(?:account|messages)\b",

        (
            r"\bemail address"
            r".{0,25}"
            r"(?:rejected|not accept)\b"
        ),

        (
            r"\bcannot cancel\b|"
            r"\bcan't cancel\b|"
            r"\bdifficult to cancel\b|"
            r"\bimpossible to cancel\b"
        ),

        (
            r"\bdelete (?:my|the) account\b|"
            r"\bclose (?:my|the) account\b"
        ),

        (
            r"\bremove my (?:account|details|data)\b|"
            r"\bdelete my (?:details|data)\b"
        ),

        (
            r"\bcannot unsubscribe\b|"
            r"\bcan't unsubscribe\b|"
            r"\bwill not unsubscribe\b|"
            r"\bwon't unsubscribe\b"
        ),
    ],


    "Privacy / unwanted contact": [

        r"\bunwanted (?:emails?|calls?|messages?|ads?)\b",
        r"\bunsolicited (?:emails?|calls?|messages?)\b",
        r"\bspam(?:ming|med)?\b",

        (
            r"\bscraped "
            r"(?:my|our) "
            r"(?:email|details|data|contact)\b"
        ),

        (
            r"\bharvested "
            r"(?:my|our) "
            r"(?:email|details|data|contact)\b"
        ),

        r"\bnever signed up\b",
        r"\bdid not sign up\b|"
        r"\bdidn't sign up\b",
        r"\bwithout my consent\b",
        r"\bpersonal data\b",
        r"\bgdpr\b|"
        r"\bico\b|"
        r"\bprivacy\b",
        r"\bkeep getting (?:emails|calls|messages)\b",
    ],


    "Platform usability": [

        (
            r"\bwebsite "
            r"(?:doesn't|does not|didn't|did not) work\b"
        ),

        (
            r"\bapp "
            r"(?:doesn't|does not|didn't|did not) work\b"
        ),

        (
            r"\bsite "
            r"(?:doesn't|does not|didn't|did not) work\b"
        ),

        r"\bimpossible to use\b",
        r"\bcould not access\b|"
        r"\bcouldn't access\b",
        r"\btechnical (?:problem|issue|fault)\b",

        (
            r"\bsystem "
            r"(?:is |was )?"
            r"(?:faulty|broken)\b"
        ),

        r"\bglitch\b",

        (
            r"\bai bot\b|"
            r"\bai chatbot\b|"
            r"\bai chat\b|"
            r"\bai[- ]generated\b|"
            r"\bautomated response\b|"
            r"\bautomated responses\b"
        ),

        r"\breal human\b",
    ],


    "Misleading claims / transparency": [

        r"\bmisleading\b",
        r"\bdeceptive\b",
        r"\bcomplete lie\b",
        r"\bthey (?:are|were) lying\b",
        r"\bfalse promise\b|"
        r"\bfalse promises\b",

        (
            r"\bpromises?"
            r".{0,40}"
            r"(?:not kept|not honoured|not honored|"
            r"false|misleading)\b"
        ),

        (
            r"\bdoes not match "
            r"(?:the )?"
            r"(?:promise|claim)\b"
        ),

        r"\bno transparency\b",
        r"\black of transparency\b",
        r"\bnot clear (?:that|about)\b",
        r"\bmis[- ]sold\b|"
        r"\bmis-selling\b|"
        r"\bmisselling\b",
    ],
}


# ============================================================
# THEME SIDES
# ============================================================

CUSTOMER_ONLY_THEMES = {
    "Finding a provider / low response",
    "Poor matching / location",
    "Provider quality",
    "Provider verification / safety",
    "Too many provider contacts / spam",
    "Pricing / quote transparency",
    "Review trust",
    "Dispute protection",
}


PROVIDER_ONLY_THEMES = {
    "Poor lead quality",
    "Fake / invalid leads",
    "Unresponsive leads",
    "Poor ROI / expensive leads",
    "Too much provider competition",
    "Billing / refunds",
}


BOTH_THEMES = {
    "Customer support",
    "Account / cancellation",
    "Privacy / unwanted contact",
    "Platform usability",
    "Misleading claims / transparency",
}


THEME_SIDE = {
    **{
        theme: "Customer"
        for theme in CUSTOMER_ONLY_THEMES
    },

    **{
        theme: "Service Provider / Business"
        for theme in PROVIDER_ONLY_THEMES
    },

    **{
        theme: "Both"
        for theme in BOTH_THEMES
    },
}


# ============================================================
# CLASSIFY COMPLAINT THEMES
# ============================================================

def classify_themes(
    text,
    user_type
):

    themes = []

    for theme, patterns in (
        THEME_PATTERNS.items()
    ):

        if matches_any(
            text,
            patterns
        ):

            themes.append(
                theme
            )

    # --------------------------------------------------------
    # Generic "no response" depends on which side is speaking.
    # --------------------------------------------------------

    generic_no_response = (
        matches_any(
            text,
            [
                r"\bno response\b",
                r"\bno reply\b",
                r"\bnever replied\b",
                r"\bnever responded\b",
                r"\bdidn't respond\b",
                r"\bdid not respond\b",
                r"\bheard nothing\b",
            ]
        )
    )

    if generic_no_response:

        if (
            user_type
            == "Service Provider / Business"
        ):

            if (
                "Unresponsive leads"
                not in themes
            ):

                themes.append(
                    "Unresponsive leads"
                )

        elif user_type == "Customer":

            if (
                "Finding a provider / low response"
                not in themes
            ):

                themes.append(
                    "Finding a provider / low response"
                )

    # --------------------------------------------------------
    # Credits + no work = provider ROI problem.
    # --------------------------------------------------------

    if (
        user_type
        == "Service Provider / Business"

        and matches(
            text,
            r"\bcredits?\b"
        )

        and matches_any(
            text,
            [
                r"\bno jobs?\b",
                r"\bno clients?\b",
                r"\bno customers?\b",
                r"\bno work\b",
                r"\bzero responses?\b",
                r"\bwaste of money\b",
            ]
        )

        and (
            "Poor ROI / expensive leads"
            not in themes
        )
    ):

        themes.append(
            "Poor ROI / expensive leads"
        )

    # --------------------------------------------------------
    # Customer provider no-show = provider quality.
    # --------------------------------------------------------

    if (
        user_type == "Customer"

        and matches_any(
            text,
            [
                r"\bnever turned up\b",
                r"\bdidn't turn up\b",
                r"\bdid not turn up\b",
                r"\bnever showed\b",
                r"\bghosted (?:me|us)\b",
            ]
        )
    ):

        if (
            "Provider quality"
            not in themes
        ):

            themes.append(
                "Provider quality"
            )

    if not themes:

        themes = [
            "Other / Manual Review"
        ]

    # Remove duplicate themes
    # while preserving order.

    return list(
        dict.fromkeys(
            themes
        )
    )


# ============================================================
# SECOND PASS USER TYPE
# ============================================================

def infer_user_type_from_themes(
    user_type,
    confidence,
    themes
):

    if user_type != "Unknown":

        return (
            user_type,
            confidence
        )

    customer_hits = len(
        set(themes)
        & CUSTOMER_ONLY_THEMES
    )

    provider_hits = len(
        set(themes)
        & PROVIDER_ONLY_THEMES
    )

    if (
        customer_hits > 0
        and provider_hits == 0
    ):

        return (
            "Customer",
            "Medium"
        )

    if (
        provider_hits > 0
        and customer_hits == 0
    ):

        return (
            "Service Provider / Business",
            "Medium"
        )

    return (
        "Unknown",
        "Low"
    )


# ============================================================
# SEVERITY
# ============================================================

BASE_SEVERITY = {

    "Finding a provider / low response":
        "Medium",

    "Poor matching / location":
        "Medium",

    "Provider quality":
        "Medium",

    "Provider verification / safety":
        "High",

    "Too many provider contacts / spam":
        "Medium",

    "Pricing / quote transparency":
        "Medium",

    "Review trust":
        "Medium",

    "Dispute protection":
        "High",

    "Poor lead quality":
        "Medium",

    "Fake / invalid leads":
        "Medium",

    "Unresponsive leads":
        "Medium",

    "Poor ROI / expensive leads":
        "Medium",

    "Too much provider competition":
        "Medium",

    "Billing / refunds":
        "Medium",

    "Customer support":
        "Medium",

    "Account / cancellation":
        "Medium",

    "Privacy / unwanted contact":
        "Medium",

    "Platform usability":
        "Low",

    "Misleading claims / transparency":
        "Medium",

    "Other / Manual Review":
        "Low",
}


SEVERITY_ORDER = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
}


SERIOUS_HARM_PATTERNS = [

    (
        r"\bfraudulent "
        r"(?:business|contractor|provider)\b"
    ),

    (
        r"\bscammer\b"
        r".{0,100}"
        r"\b(?:lost|paid|deposit|money|£|\$|€)\b"
    ),

    (
        r"\b(?:lost|stole|stolen|duped|ripped off)\b"
        r".{0,80}"
        r"(?:£|\$|€)\s*[0-9]"
    ),

    r"\btook my deposit\b",
    r"\bunsafe\b",
    r"\bdangerous\b",
    r"\binjury\b",
    r"\bpolice\b",
    r"\blegal action\b",
]


def extract_money_amounts(
    text
):

    amounts = []

    matches_found = re.findall(
        (
            r"(?:£|\$|€)\s*"
            r"([0-9]+"
            r"(?:,[0-9]{3})*"
            r"(?:\.[0-9]+)?)"
        ),
        text
    )

    for amount in matches_found:

        try:

            amounts.append(
                float(
                    amount.replace(
                        ",",
                        ""
                    )
                )
            )

        except ValueError:
            pass

    return amounts


def classify_severity(
    text,
    themes
):

    if matches_any(
        text,
        SERIOUS_HARM_PATTERNS
    ):

        return "High"

    amounts = extract_money_amounts(
        text
    )

    if (
        amounts
        and max(amounts) >= 500
        and matches_any(
            text,
            [
                r"\blost\b",
                r"\bscam\b",
                r"\bripped off\b",
                r"\brefund\b",
                r"\bcharged\b",
                r"\bpaid\b",
                r"\bdeposit\b",
            ]
        )
    ):

        return "High"

    severity = "Low"

    for theme in themes:

        candidate = (
            BASE_SEVERITY.get(
                theme,
                "Low"
            )
        )

        if (
            SEVERITY_ORDER[
                candidate
            ]
            >
            SEVERITY_ORDER[
                severity
            ]
        ):

            severity = candidate

    return severity


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n"
            f"{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    print(
        f"\nLoaded "
        f"{len(df):,} "
        f"negative reviews."
    )

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    df = df.copy()

    required = {
        "reviewId",
        "rating",
        "publishedDate",
        "text",
        "competitor",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    df["publishedDate"] = (
        pd.to_datetime(
            df["publishedDate"],
            errors="coerce",
            utc=True
        )
    )

    df["rating"] = (
        pd.to_numeric(
            df["rating"],
            errors="coerce"
        )
    )

    df["year"] = (
        df[
            "publishedDate"
        ]
        .dt.year
    )

    df["clean_text"] = (
        df["text"]
        .apply(
            clean_text
        )
    )

    # --------------------------------------------------------
    # RELEVANCE
    # --------------------------------------------------------

    relevance_results = (
        df.apply(
            classify_relevance,
            axis=1
        )
    )

    df["relevance"] = (
        relevance_results
        .apply(
            lambda result:
            result[0]
        )
    )

    df[
        "relevance_reason"
    ] = (
        relevance_results
        .apply(
            lambda result:
            result[1]
        )
    )

    # --------------------------------------------------------
    # USER TYPE
    # --------------------------------------------------------

    user_results = (
        df.apply(
            lambda row:
            classify_user_type(
                row[
                    "clean_text"
                ],
                row[
                    "competitor"
                ]
            ),
            axis=1
        )
    )

    df["user_type"] = (
        user_results
        .apply(
            lambda result:
            result[0]
        )
    )

    df[
        "user_type_confidence"
    ] = (
        user_results
        .apply(
            lambda result:
            result[1]
        )
    )

    df["provider_score"] = (
        user_results
        .apply(
            lambda result:
            result[2]
        )
    )

    df["customer_score"] = (
        user_results
        .apply(
            lambda result:
            result[3]
        )
    )

    # --------------------------------------------------------
    # FIRST COMPLAINT PASS
    # --------------------------------------------------------

    df[
        "complaint_themes"
    ] = (
        df.apply(
            lambda row:
            classify_themes(
                row[
                    "clean_text"
                ],
                row[
                    "user_type"
                ]
            ),
            axis=1
        )
    )

    # --------------------------------------------------------
    # SECOND USER-TYPE PASS
    # --------------------------------------------------------

    second_pass = (
        df.apply(
            lambda row:
            infer_user_type_from_themes(
                row[
                    "user_type"
                ],
                row[
                    "user_type_confidence"
                ],
                row[
                    "complaint_themes"
                ]
            ),
            axis=1
        )
    )

    df["user_type"] = (
        second_pass
        .apply(
            lambda result:
            result[0]
        )
    )

    df[
        "user_type_confidence"
    ] = (
        second_pass
        .apply(
            lambda result:
            result[1]
        )
    )

    # --------------------------------------------------------
    # RE-RUN THEMES AFTER USER TYPE IMPROVEMENT
    # --------------------------------------------------------

    df[
        "complaint_themes"
    ] = (
        df.apply(
            lambda row:
            classify_themes(
                row[
                    "clean_text"
                ],
                row[
                    "user_type"
                ]
            ),
            axis=1
        )
    )

    # --------------------------------------------------------
    # SEVERITY
    # --------------------------------------------------------

    df["severity"] = (
        df.apply(
            lambda row:
            classify_severity(
                row[
                    "clean_text"
                ],
                row[
                    "complaint_themes"
                ]
            ),
            axis=1
        )
    )

    # --------------------------------------------------------
    # MANUAL REVIEW FLAG
    # --------------------------------------------------------

    df[
        "needs_manual_review"
    ] = (

        df[
            "relevance"
        ].eq(
            "Review"
        )

        |

        df[
            "user_type"
        ].eq(
            "Unknown"
        )

        |

        df[
            "user_type_confidence"
        ].eq(
            "Low"
        )

        |

        df[
            "complaint_themes"
        ]
        .apply(
            lambda themes:
            "Other / Manual Review"
            in themes
        )
    )

    return df


# ============================================================
# EXPLODE COMPLAINT THEMES
# ============================================================

def explode_complaints(df):

    relevant = (
        df[
            df[
                "relevance"
            ].eq(
                "Relevant"
            )
        ]
        .copy()
    )

    complaint_df = (
        relevant
        .explode(
            "complaint_themes"
        )
        .rename(
            columns={
                "complaint_themes":
                    "weakness"
            }
        )
    )

    complaint_df[
        "theme_side"
    ] = (
        complaint_df[
            "weakness"
        ]
        .map(
            THEME_SIDE
        )
        .fillna(
            "Manual review"
        )
    )

    return complaint_df


# ============================================================
# FREQUENCY
# ============================================================

def calculate_frequency(
    complaint_df
):

    useful = (
        complaint_df[
            complaint_df[
                "weakness"
            ].ne(
                "Other / Manual Review"
            )
        ]
        .copy()
    )

    totals = (
        complaint_df
        .groupby(
            "competitor"
        )[
            "reviewId"
        ]
        .nunique()
        .rename(
            "total_relevant_negative_reviews"
        )
    )

    frequency = (
        useful
        .groupby(
            [
                "competitor",
                "weakness"
            ]
        )[
            "reviewId"
        ]
        .nunique()
        .reset_index(
            name="frequency"
        )
        .merge(
            totals,
            on="competitor",
            how="left"
        )
    )

    frequency[
        "percent_of_collected_negative_sample"
    ] = (

        frequency[
            "frequency"
        ]

        /

        frequency[
            "total_relevant_negative_reviews"
        ]

        * 100

    ).round(1)

    frequency = (
        frequency
        .sort_values(
            [
                "competitor",
                "frequency"
            ],
            ascending=[
                True,
                False
            ]
        )
    )

    return frequency


# ============================================================
# USER TYPE DISTRIBUTION
# ============================================================

def calculate_user_type_distribution(
    df
):

    relevant = (
        df[
            df[
                "relevance"
            ].eq(
                "Relevant"
            )
        ]
        .copy()
    )

    result = (
        relevant
        .groupby(
            [
                "competitor",
                "user_type"
            ]
        )[
            "reviewId"
        ]
        .nunique()
        .reset_index(
            name="reviews"
        )
    )

    return result


# ============================================================
# YEARLY FREQUENCY
# ============================================================

def calculate_yearly_frequency(
    complaint_df
):

    useful = (
        complaint_df[
            complaint_df[
                "weakness"
            ].ne(
                "Other / Manual Review"
            )
        ]
        .copy()
    )

    result = (
        useful
        .groupby(
            [
                "competitor",
                "year",
                "weakness"
            ]
        )[
            "reviewId"
        ]
        .nunique()
        .reset_index(
            name="frequency"
        )
        .sort_values(
            [
                "competitor",
                "year",
                "frequency"
            ],
            ascending=[
                True,
                True,
                False
            ]
        )
    )

    return result


# ============================================================
# PERSISTENCE
# ============================================================

def calculate_persistence(
    yearly_frequency
):

    persistence = (
        yearly_frequency
        .groupby(
            [
                "competitor",
                "weakness"
            ]
        )
        .agg(
            years_present=(
                "year",
                "nunique"
            ),
            total_mentions=(
                "frequency",
                "sum"
            ),
            first_year=(
                "year",
                "min"
            ),
            last_year=(
                "year",
                "max"
            ),
        )
        .reset_index()
    )

    persistence = (
        persistence
        .sort_values(
            [
                "competitor",
                "years_present",
                "total_mentions"
            ],
            ascending=[
                True,
                False,
                False
            ]
        )
    )

    return persistence


# ============================================================
# WEAKNESS MATRIX
# ============================================================

def build_competitor_weakness_matrix(
    frequency
):

    matrix = (
        frequency
        .pivot_table(
            index="weakness",
            columns="competitor",
            values=(
                "percent_of_collected_negative_sample"
            ),
            fill_value=0
        )
    )

    for competitor in COMPETITORS:

        if competitor not in matrix.columns:

            matrix[
                competitor
            ] = 0.0

    matrix = matrix[
        COMPETITORS
    ]

    matrix[
        "marketplace_side"
    ] = (
        matrix
        .index
        .map(
            THEME_SIDE
        )
        .fillna(
            "Both"
        )
    )

    matrix = (
        matrix
        .reset_index()
    )

    matrix = matrix[
        [
            "marketplace_side",
            "weakness",
            *COMPETITORS
        ]
    ]

    return matrix


# ============================================================
# VALIDATION SAMPLE
# ============================================================

def create_validation_sample(
    df,
    sample_size_per_competitor=25
):

    candidates = (
        df[
            df[
                "relevance"
            ]
            .isin(
                [
                    "Relevant",
                    "Review"
                ]
            )
        ]
        .copy()
    )

    samples = []

    for competitor in COMPETITORS:

        subset = (
            candidates[
                candidates[
                    "competitor"
                ].eq(
                    competitor
                )
            ]
        )

        if subset.empty:
            continue

        uncertain = (
            subset[
                subset[
                    "needs_manual_review"
                ]
            ]
        )

        confident = (
            subset[
                ~subset[
                    "needs_manual_review"
                ]
            ]
        )

        uncertain_n = min(
            15,
            len(
                uncertain
            ),
            sample_size_per_competitor
        )

        if uncertain_n > 0:

            chosen_uncertain = (
                uncertain
                .sample(
                    n=uncertain_n,
                    random_state=42
                )
            )

        else:

            chosen_uncertain = (
                uncertain
            )

        remaining = (
            sample_size_per_competitor
            - len(
                chosen_uncertain
            )
        )

        confident_n = min(
            remaining,
            len(
                confident
            )
        )

        if confident_n > 0:

            chosen_confident = (
                confident
                .sample(
                    n=confident_n,
                    random_state=42
                )
            )

        else:

            chosen_confident = (
                confident
            )

        chosen = (
            pd.concat(
                [
                    chosen_uncertain,
                    chosen_confident
                ],
                ignore_index=True
            )
        )

        samples.append(
            chosen
        )

    if samples:

        validation = (
            pd.concat(
                samples,
                ignore_index=True
            )
        )

    else:

        validation = (
            pd.DataFrame()
        )

    if not validation.empty:

        validation[
            "manual_relevance"
        ] = ""

        validation[
            "manual_user_type"
        ] = ""

        validation[
            "manual_complaint_themes"
        ] = ""

        validation[
            "manual_severity"
        ] = ""

        validation[
            "validation_notes"
        ] = ""

    return validation


# ============================================================
# GRAPH - WEAKNESSES
# ============================================================

def create_weakness_graph(
    frequency
):

    overall = (
        frequency
        .groupby(
            "weakness"
        )[
            "frequency"
        ]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(12)
        .index
    )

    graph_data = (
        frequency[
            frequency[
                "weakness"
            ]
            .isin(
                overall
            )
        ]
        .pivot(
            index="competitor",
            columns="weakness",
            values=(
                "percent_of_collected_negative_sample"
            )
        )
        .fillna(0)
    )

    ax = graph_data.plot(
        kind="bar",
        figsize=(15, 8)
    )

    ax.set_title(
        "Competitor Weaknesses in Collected "
        "1★–3★ Trustpilot Reviews"
    )

    ax.set_xlabel(
        "Competitor"
    )

    ax.set_ylabel(
        "% of collected negative review sample"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "competitor_weaknesses_final.png",
        dpi=300
    )

    plt.close()


# ============================================================
# GRAPH - USER TYPE
# ============================================================

def create_user_type_graph(
    user_types
):

    graph_data = (
        user_types
        .pivot(
            index="competitor",
            columns="user_type",
            values="reviews"
        )
        .fillna(0)
    )

    ax = graph_data.plot(
        kind="bar",
        figsize=(10, 6)
    )

    ax.set_title(
        "Negative Reviews by Marketplace Side"
    )

    ax.set_xlabel(
        "Competitor"
    )

    ax.set_ylabel(
        "Number of collected negative reviews"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "user_type_distribution_final.png",
        dpi=300
    )

    plt.close()


# ============================================================
# QUALITY CHECK
# ============================================================

def print_quality_check(
    df,
    frequency
):

    relevant = (
        df[
            df[
                "relevance"
            ].eq(
                "Relevant"
            )
        ]
    )

    excluded = (
        df[
            df[
                "relevance"
            ].eq(
                "Exclude"
            )
        ]
    )

    review = (
        df[
            df[
                "relevance"
            ].eq(
                "Review"
            )
        ]
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "FINAL CLASSIFICATION QUALITY CHECK"
    )

    print(
        "=" * 72
    )

    print(
        f"Total negative reviews: "
        f"{len(df):,}"
    )

    print(
        f"Relevant platform reviews: "
        f"{len(relevant):,}"
    )

    print(
        f"Excluded off-topic reviews: "
        f"{len(excluded):,}"
    )

    print(
        f"Ambiguous relevance - manual review: "
        f"{len(review):,}"
    )

    if len(relevant) > 0:

        unknown = (
            relevant[
                "user_type"
            ]
            .eq(
                "Unknown"
            )
            .sum()
        )

        manual = (
            relevant[
                "needs_manual_review"
            ]
            .sum()
        )

        other = (
            relevant[
                "complaint_themes"
            ]
            .apply(
                lambda themes:
                "Other / Manual Review"
                in themes
            )
            .sum()
        )

        print(
            f"Unknown user type: "
            f"{unknown:,} "
            f"({unknown / len(relevant):.1%})"
        )

        print(
            f"Other / Manual Review theme: "
            f"{other:,} "
            f"({other / len(relevant):.1%})"
        )

        print(
            f"Needs manual review overall: "
            f"{manual:,} "
            f"({manual / len(relevant):.1%})"
        )

    print(
        "\nTOP WEAKNESSES BY COMPETITOR"
    )

    print(
        "=" * 72
    )

    for competitor in COMPETITORS:

        subset = (
            frequency[
                frequency[
                    "competitor"
                ].eq(
                    competitor
                )
            ]
            .head(8)
        )

        if subset.empty:
            continue

        print(
            f"\n{competitor}"
        )

        print(
            subset[
                [
                    "weakness",
                    "frequency",
                    (
                        "percent_of_collected_"
                        "negative_sample"
                    )
                ]
            ]
            .to_string(
                index=False
            )
        )


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(
    df,
    complaint_df,
    frequency,
    user_types,
    yearly_frequency,
    persistence,
    matrix,
    validation,
):

    # --------------------------------------------------------
    # ALL CLASSIFIED REVIEWS
    # --------------------------------------------------------

    classified = (
        df.copy()
    )

    classified[
        "complaint_themes"
    ] = (
        classified[
            "complaint_themes"
        ]
        .apply(
            lambda themes:
            " | ".join(
                themes
            )
        )
    )

    classified.to_csv(
        OUTPUT_DIR
        / "classified_negative_reviews_final.csv",
        index=False
    )

    # --------------------------------------------------------
    # EXCLUDED OFF-TOPIC
    # --------------------------------------------------------

    (
        df[
            df[
                "relevance"
            ].eq(
                "Exclude"
            )
        ]
        .to_csv(
            OUTPUT_DIR
            / "excluded_offtopic_reviews.csv",
            index=False
        )
    )

    # --------------------------------------------------------
    # MANUAL REVIEW QUEUE
    # --------------------------------------------------------

    manual_queue = (
        df[
            df[
                "needs_manual_review"
            ]

            |

            df[
                "relevance"
            ].eq(
                "Review"
            )
        ]
        .copy()
    )

    manual_queue[
        "complaint_themes"
    ] = (
        manual_queue[
            "complaint_themes"
        ]
        .apply(
            lambda themes:
            " | ".join(
                themes
            )
        )
    )

    manual_queue.to_csv(
        OUTPUT_DIR
        / "manual_review_queue.csv",
        index=False
    )

    # --------------------------------------------------------
    # COMPLAINT MENTIONS
    # --------------------------------------------------------

    complaint_df.to_csv(
        OUTPUT_DIR
        / "complaint_mentions_final.csv",
        index=False
    )

    # --------------------------------------------------------
    # FREQUENCY
    # --------------------------------------------------------

    frequency.to_csv(
        OUTPUT_DIR
        / "complaint_frequency_by_competitor_final.csv",
        index=False
    )

    # --------------------------------------------------------
    # USER TYPES
    # --------------------------------------------------------

    user_types.to_csv(
        OUTPUT_DIR
        / "user_type_distribution_final.csv",
        index=False
    )

    # --------------------------------------------------------
    # YEARLY FREQUENCY
    # --------------------------------------------------------

    yearly_frequency.to_csv(
        OUTPUT_DIR
        / "complaint_frequency_by_year_final.csv",
        index=False
    )

    # --------------------------------------------------------
    # PERSISTENCE
    # --------------------------------------------------------

    persistence.to_csv(
        OUTPUT_DIR
        / "weakness_persistence_final.csv",
        index=False
    )

    # --------------------------------------------------------
    # FINAL MATRIX
    # --------------------------------------------------------

    matrix.to_csv(
        OUTPUT_DIR
        / "competitor_weakness_matrix_final.csv",
        index=False
    )

    # --------------------------------------------------------
    # VALIDATION SAMPLE
    # --------------------------------------------------------

    if not validation.empty:

        validation_save = (
            validation.copy()
        )

        validation_save[
            "complaint_themes"
        ] = (
            validation_save[
                "complaint_themes"
            ]
            .apply(
                lambda themes:
                " | ".join(
                    themes
                )
            )
        )

        validation_save.to_csv(
            OUTPUT_DIR
            / "manual_validation_sample_final.csv",
            index=False
        )

    print(
        f"\nFiles saved to:\n"
        f"{OUTPUT_DIR}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\nStarting final competitor "
        "complaint analysis..."
    )

    df = load_data()

    df = prepare_data(
        df
    )

    complaint_df = (
        explode_complaints(
            df
        )
    )

    frequency = (
        calculate_frequency(
            complaint_df
        )
    )

    user_types = (
        calculate_user_type_distribution(
            df
        )
    )

    yearly_frequency = (
        calculate_yearly_frequency(
            complaint_df
        )
    )

    persistence = (
        calculate_persistence(
            yearly_frequency
        )
    )

    matrix = (
        build_competitor_weakness_matrix(
            frequency
        )
    )

    validation = (
        create_validation_sample(
            df,
            sample_size_per_competitor=25
        )
    )

    print_quality_check(
        df,
        frequency
    )

    create_weakness_graph(
        frequency
    )

    create_user_type_graph(
        user_types
    )

    save_outputs(
        df,
        complaint_df,
        frequency,
        user_types,
        yearly_frequency,
        persistence,
        matrix,
        validation,
    )

    print(
        "\nFinal competitor complaint "
        "analysis complete."
    )

    print(
        "\nIMPORTANT: Validate "
        "manual_validation_sample_final.csv "
        "before treating the weakness matrix "
        "as a final research finding."
    )


if __name__ == "__main__":
    main()
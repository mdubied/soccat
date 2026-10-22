"""
Step 2 taxonomy: broad category -> specific labels, and normalisation of the
raw human-annotation column (specific_group_new) to canonical taxonomy labels.

TAXONOMY must stay in sync with prompts/step_2/step_2_*.txt (the categories
listed there are these, verbatim). LABEL_MAP is copied from
src/step_2/convert_annotations.py's mapping (the script that builds the
mDeBERTa NLI training pairs from data/manual_annotations/step_2/
annotations_ground_truth.csv), with "others" added since that script's
taxonomy only covers the 8 trained broad categories -- "Others" is a human
annotation category the mDeBERTa models were never trained to predict, but
our prompt does ask the LLM to use it, so it must round-trip for ground
truth comparison too. Any specific_group_new value not in LABEL_MAP (e.g.
"enterprises", "victims of crimes" -- categories dropped from the final
taxonomy) is treated as no codable category, exactly as it is for mDeBERTa.
"""

TAXONOMY = {
    "Socio-economic position": [
        "lower class", "middle class", "upper class",
        "capital owners, investors and shareholders",
        "unskilled or unqualified", "skilled or qualified",
    ],
    "Labor market position": [
        "wage and salary earners", "civil servants", "CEOs and corporate leaders",
        "employers", "entrepreneurs", "self-employed and freelancers",
        "unemployed", "retirees", "housewives and househusbands",
    ],
    "Age and family status": [
        "parents and families", "minors, including children and pupils",
        "youth, including students and apprentices",
        "middle-aged and pre-retirement age groups", "elderly", "couples", "singles",
    ],
    "Identities and minority/majority status": [
        "men", "women", "cisgender and heterosexuals", "LGBTQIA+", "disabled people",
        "people with an immigration background, including immigrants",
        "ethnic and racial minorities", "Christians", "Jews", "Muslims",
        "multiple (or other) religious or minority groups",
    ],
    "Profession": [
        "athletes", "authors and artists", "doctors", "farmers and fishermen",
        "health and care professionals", "journalists", "legal professionals",
        "politicians and high-ranking officials", "sex workers",
        "scientists and professors", "security forces", "soldiers",
        "teachers and educators", "other professions",
    ],
    "Social roles and behavior": ["consumers and clients", "car drivers", "patients"],
    "Social deviance": [
        "extremists",
        "terrorists, rebels, revolutionaries and/or movements of armed resistance",
        "offenders, criminals, prisoners and/or accused people", "drug addicts",
    ],
    "Real estate ownership": ["real-estate owners", "tenants", "homeless"],
    "Others": ["others"],
}

# raw specific_group_new value (lowercase, stripped) -> canonical taxonomy label.
# Copied from src/step_2/convert_annotations.py::LABEL_MAP, plus "others".
LABEL_MAP = {
    "lower class": "lower class",
    "middle class": "middle class",
    "upper class": "upper class",
    "capital owners, investors and shareholders": "capital owners, investors and shareholders",
    "unskilled or unqualified": "unskilled or unqualified",
    "skilled or qualified": "skilled or qualified",

    "wage and salary earners": "wage and salary earners",
    "civil servants": "civil servants",
    "ceos and corporate leaders": "CEOs and corporate leaders",
    "employers": "employers",
    "entrepreneur": "entrepreneurs",
    "entrepreneurs": "entrepreneurs",
    "entrepreneurs (smes)": "entrepreneurs",
    "entrepreneurs in [specific] sector": "entrepreneurs",
    "entrepreneurs (large enterprises)": "CEOs and corporate leaders",
    "self-employed and freelancers": "self-employed and freelancers",
    "unemployed": "unemployed",
    "retirees": "retirees",
    "housewife and househusband": "housewives and househusbands",
    "housewives and househusbands": "housewives and househusbands",

    "parents and families": "parents and families",
    "minors": "minors, including children and pupils",
    "minors, including children and pupils": "minors, including children and pupils",
    "youth": "youth, including students and apprentices",
    "youth, including students and apprentices": "youth, including students and apprentices",
    "middle-aged and pre-retirement age groups": "middle-aged and pre-retirement age groups",
    "elderly": "elderly",
    "couples": "couples",
    "singles": "singles",

    "men": "men",
    "women": "women",
    "cisgender and heterosexuals": "cisgender and heterosexuals",
    "lgbtqia+": "LGBTQIA+",
    "lgbtqqia+": "LGBTQIA+",
    "disabled people": "disabled people",
    "people with an immigration background, including immigrants":
        "people with an immigration background, including immigrants",
    "ethnic and racial minorities": "ethnic and racial minorities",
    "christians": "Christians",
    "jews": "Jews",
    "muslims": "Muslims",
    "multiple (or other specific) religious or minority groups":
        "multiple (or other) religious or minority groups",
    "multiple (or other) religious or minority groups":
        "multiple (or other) religious or minority groups",

    "athletes": "athletes",
    "authors and artists": "authors and artists",
    "doctors": "doctors",
    "farmers and fishermen": "farmers and fishermen",
    "health and care professionals": "health and care professionals",
    "journalists": "journalists",
    "legal professionals": "legal professionals",
    "politicians and high-ranking officials": "politicians and high-ranking officials",
    "sex workers": "sex workers",
    "prostitutes": "sex workers",
    "scientists and professors": "scientists and professors",
    "security forces": "security forces",
    "soldiers": "soldiers",
    "teachers and educators": "teachers and educators",
    "other profession": "other professions",
    "other professions": "other professions",

    "consumers and clients": "consumers and clients",
    "car drivers": "car drivers",
    "patients": "patients",

    "extremists": "extremists",
    "terrorists": "terrorists, rebels, revolutionaries and/or movements of armed resistance",
    "terrorists, rebels, revolutionaries and/or movements of armed resistance":
        "terrorists, rebels, revolutionaries and/or movements of armed resistance",
    "offenders, criminals, prisoners and/or accused people":
        "offenders, criminals, prisoners and/or accused people",
    "drug addicts": "drug addicts",

    "real-estate owner": "real-estate owners",
    "real-estate owners": "real-estate owners",
    "real estate owners": "real-estate owners",
    "tenants": "tenants",
    "homeless": "homeless",

    "others": "others",
}

# specific taxonomy label (lowercase) -> broad category display name.
LABEL_TO_BROAD = {
    label.lower(): broad for broad, labels in TAXONOMY.items() for label in labels
}

ALL_SPECIFIC_LABELS = sorted({label for labels in TAXONOMY.values() for label in labels})
ALL_BROAD_CATEGORIES = list(TAXONOMY.keys())


def parse_true_categories(raw) -> list:
    """Parse a semicolon-separated specific_group_new cell into a sorted, deduplicated
    list of (broad_category, specific_category) tuples using canonical taxonomy labels.
    Values with no taxonomy match (e.g. "enterprises", "victims of crimes" -- categories
    dropped from the final taxonomy) are dropped, matching how the mDeBERTa NLI training
    pairs were built."""
    if raw is None or (isinstance(raw, float) and raw != raw):  # NaN
        return []
    seen = set()
    for part in str(raw).split(";"):
        key = part.strip().lower()
        specific = LABEL_MAP.get(key)
        if specific:
            broad = LABEL_TO_BROAD[specific.lower()]
            seen.add((broad, specific))
    return sorted(seen)

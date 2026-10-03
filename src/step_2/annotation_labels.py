"""
annotation_labels.py

Step 2 manual annotations: codebook, label normalisation, span parsing and loading of
data/manual_annotations/step_2/annotation_step2.csv (ground truth + independent coders).

Shared by tables/icr_step2_krippendorff.ipynb and figures/annotation_confusion_matrix.py.
"""
import ast
import re
from pathlib import Path

import pandas as pd

ANNOTATION_FILE = Path(__file__).resolve().parents[2] / 'data' / 'manual_annotations' / 'step_2' / 'annotation_step2.csv'


# ── Codebook ─────────────────────────────────────────────────────────────────
CODEBOOK: dict[str, list[str]] = {
    'Socio-economic position': [
        'lower class', 'middle class', 'upper class',
        'capital owners, investors and shareholders',
        'unskilled or unqualified', 'skilled or qualified',
    ],
    'Labor market position': [
        'wage and salary earners', 'civil servants', 'CEOs and corporate leaders',
        'employers', 'entrepreneurs', 'self-employed and freelancers',
        'unemployed', 'retirees', 'housewives and househusbands',
    ],
    'Age and family status': [
        'parents and families', 'minors', 'youth',
        'middle-aged and pre-retirement age groups', 'elderly', 'couples', 'singles',
    ],
    'Gender, sexuality, and sociocultural characteristics': [
        'men', 'women', 'cisgender and heterosexuals', 'lgbtqia+', 'disabled people',
        'people with an immigration background, including immigrants',
        'ethnic and racial minorities',
        'christians', 'jews', 'muslims',
        'multiple (or other) religious or minority groups',
    ],
    'Profession': [
        'athletes', 'authors and artists', 'doctors', 'farmers and fishermen',
        'health and care professionals', 'journalists', 'legal professionals',
        'politicians and high-ranking officials', 'sex workers',
        'scientists and professors', 'security forces', 'soldiers',
        'teachers and educators', 'other professions',
    ],
    'Social roles and behavior': [
        'consumers and clients', 'car drivers', 'patients',
    ],
    'Social deviance': [
        'extremists',
        'terrorists, rebels, revolutionaries and/or movements of armed resistance',
        'offenders, criminals, prisoners and/or accused people',
        'drug addicts',
    ],
    'Real estate ownership': ['real-estate owners', 'tenants', 'homeless'],
    'Others': ['others'],
}

CANONICAL = {lbl for lbls in CODEBOOK.values() for lbl in lbls}


# ── Label normalisation ──────────────────────────────────────────────────────
# Legacy and variant label spellings are mapped to canonical codebook labels.
# Unknown labels fall back to 'others'.
COMMA_LABELS = [
    'offenders, criminals, prisoners and/or accused people',
    'terrorists, rebels, revolutionaries and/or movements of armed resistance',
    'capital owners, investors and shareholders',
    'multiple (or other) religious or minority groups',
    'people with an immigration background, including immigrants',
    'minors, including children and pupils',
    'youth, including students and apprentices',
]

NORMALISE: dict[str, str | None] = {
    'poor': 'lower class', 'underprivileged': 'lower class',
    'unskilled and/or underprivileged': 'unskilled or unqualified',
    'qualified': 'skilled or qualified', 'qualified and graduates': 'skilled or qualified',
    'investors and stakeholders': 'capital owners, investors and shareholders',
    'employees': 'wage and salary earners', 'precarious employees': 'wage and salary earners',
    'working active population': 'wage and salary earners',
    'housewife and househusband': 'housewives and househusbands',
    'self-employed/freelancers': 'self-employed and freelancers',
    'leaders': 'CEOs and corporate leaders',
    'ceos and corporate leaders': 'CEOs and corporate leaders',
    'enterprises': 'entrepreneurs', 'large enterprises': 'entrepreneurs',
    'small- and middle-size enterprises': 'entrepreneurs', 'specific sector': 'entrepreneurs',
    'entrepreneurs (smes)': 'entrepreneurs', 'entrepreneurs (large enterprises)': 'entrepreneurs',
    'entrepreneurs in [specific] sector': 'entrepreneurs',
    'minors, including children and pupils': 'minors',
    'youth, including students and apprentices': 'youth',
    'middle-aged': 'middle-aged and pre-retirement age groups',
    'older age group': 'elderly',
    'seniors': 'retirees',  # legacy label: matches consensus 'retirees', never 'elderly'
    'immigrants': 'people with an immigration background, including immigrants',
    'people with an immigration background': 'people with an immigration background, including immigrants',
    'racial and ethnic minorities': 'ethnic and racial minorities',
    'visible and ethnic minorities': 'ethnic and racial minorities',
    'ethnic minorities': 'ethnic and racial minorities',
    'minorities': 'ethnic and racial minorities',
    'east germans': 'ethnic and racial minorities',
    'west germans': 'ethnic and racial minorities',
    'ethnic germans': 'ethnic and racial minorities',
    'expatriates': 'ethnic and racial minorities',
    'white': 'ethnic and racial minorities',
    'white people': 'ethnic and racial minorities',
    'visible minorities': 'ethnic and racial minorities',
    'language and ethnic minorities': 'ethnic and racial minorities',
    'lgbtqi*': 'lgbtqia+', 'lgbtqqia+': 'lgbtqia+',
    'cis & heterosexuals': 'cisgender and heterosexuals',
    'religious groups': 'multiple (or other) religious or minority groups',
    'religious minorities': 'multiple (or other) religious or minority groups',
    'territorial language minorities': 'multiple (or other) religious or minority groups',
    'other minorities': 'multiple (or other) religious or minority groups',
    'other minorities / religious groups': 'multiple (or other) religious or minority groups',
    'disabled': 'disabled people',
    'other profession': 'other professions',
    'scientists': 'scientists and professors', 'professors': 'scientists and professors',
    'teachers': 'teachers and educators', 'educators': 'teachers and educators',
    'farmers': 'farmers and fishermen', 'fishermen': 'farmers and fishermen',
    'high-ranking officials': 'politicians and high-ranking officials',
    'politicians': 'politicians and high-ranking officials',
    'prostitutes': 'sex workers',
    'social professions': 'other professions',
    'engineers': 'other professions',
    'lobbyists': 'other professions',
    'hunters': 'other professions',
    'people working in the public sector': 'civil servants',
    'commuters': 'consumers and clients',
    'cyclists': 'car drivers', 'pedestrians': 'car drivers',
    'road carriers': 'consumers and clients',
    'air travellers': 'consumers and clients',
    'users of certain transportation modes': 'consumers and clients',
    'public transport passengers': 'consumers and clients',
    'insured persons': 'patients',
    'consumers': 'consumers and clients',
    'tax payers': 'others', 'gun owners': 'others',
    'tax evaders and white collar criminals': 'offenders, criminals, prisoners and/or accused people',
    'offenders': 'offenders, criminals, prisoners and/or accused people',
    'offenders or criminals': 'offenders, criminals, prisoners and/or accused people',
    'criminals': 'offenders, criminals, prisoners and/or accused people',
    'criminals and/or prisoners': 'offenders, criminals, prisoners and/or accused people',
    'prisoners': 'offenders, criminals, prisoners and/or accused people',
    'terrorists': 'terrorists, rebels, revolutionaries and/or movements of armed resistance',
    'home owner': 'real-estate owners', 'land owner': 'real-estate owners',
    'landlords': 'real-estate owners', 'real-estate owner': 'real-estate owners',
    'real estate owners': 'real-estate owners',
    'inhabitants of cities': 'others',
    'inhabitants of rural or underserved areas': 'others',
    'inhabitants of other areas': 'others',
    'inhabitants of overseas': 'others',
    'inhabitants of specific sites': 'others',
    'inhabitants of underprivileged areas': 'others',
    'victims of crimes': 'others', 'victims of state violence': 'others',
    'victims of german history': 'others', 'whistle-blower and witnesses': 'others',
    'volunteers': 'others', 'people without public social protection': 'others',
    'heirs': 'others',
    'other': 'others',
    'target abroad': None,
}


def normalise_label(raw: str) -> str | None:
    raw = raw.strip().lower()
    if raw in ('target abroad', 'nan', ''):
        return None
    if raw in NORMALISE:
        return NORMALISE[raw]
    if raw in CANONICAL:
        return raw
    for canon in CANONICAL:
        if canon.lower() == raw:
            return canon
    return 'others'


# ── Span parsing ─────────────────────────────────────────────────────────────
def _protect(t: str) -> str:
    for cl in sorted(COMMA_LABELS, key=len, reverse=True):
        t = t.replace(cl, cl.replace(', ', '|||'))
    return t

def _unprotect(t: str) -> str:
    return t.replace('|||', ', ')


def parse_ann_label(val) -> frozenset:
    """Individual annotator label field (semicolon- and comma-separated strings)."""
    if pd.isna(val):
        return frozenset()
    val = str(val).strip()
    if val in ('nan', ''):
        return frozenset()
    protected = _protect(val.lower())
    labels = set()
    for chunk in re.split(r';\s*', protected):
        chunk = _unprotect(chunk).strip()
        if not chunk or chunk == 'nan':
            continue
        for piece in _protect(chunk).split(', '):
            piece = _unprotect(piece).strip()
            if piece:
                canon = normalise_label(piece)
                if canon is not None:
                    labels.add(canon)
    return frozenset(labels)


def load_spans(val) -> list:
    """Span list [[start, end, 'label'], ...]. A single span stored without the outer list is wrapped.
    Malformed values raise instead of being read as 'no group'. Only labels are used: span offsets
    (a few are shifted or run past the end of the text) and duplicate spans do not matter here."""
    spans = ast.literal_eval(str(val))
    if spans and not isinstance(spans[0], list):
        spans = [spans]
    return [s for s in spans if isinstance(s, list) and len(s) >= 3]


def parse_consensus_label(val) -> frozenset:
    """Ground-truth span list; empty labels and 'target abroad' are dropped by normalise_label."""
    if pd.isna(val):
        return frozenset()
    labels = {normalise_label(str(s[2])) for s in load_spans(val)}
    return frozenset(labels - {None})


def parse_coder_spans(val) -> frozenset:
    """Independent coder span list [[start, end, 'label(s)'], ...]; a span label may hold several comma-separated labels."""
    return parse_ann_label('; '.join(str(s[2]) for s in load_spans(val)))


# ── Loading ──────────────────────────────────────────────────────────────────
def load_sentences(path=ANNOTATION_FILE) -> pd.DataFrame:
    """One row per sentence with an outlet and at least one independent coder.

    Adds: gt_labels (frozenset), coders (list of names), coder_labels (list of frozensets,
    aligned with coders), values (tuple: ground truth followed by each coder's labels).
    """
    df = pd.read_csv(path)
    CODER_COLS = [c for c in df.columns if c.startswith('annotation_') and c != 'annotation_sources']

    df['outlet'] = df['outlet'].astype(str)
    df = df[df['outlet'] != 'nan']
    df = df[df[CODER_COLS].notna().any(axis=1)].copy()  # sentences with a ground truth but no independent coder

    df['gt_labels'] = df['ground_truth'].apply(parse_consensus_label)
    df['coders'] = df.apply(lambda r: [c.removeprefix('annotation_') for c in CODER_COLS if pd.notna(r[c])], axis=1)
    df['coder_labels'] = df.apply(lambda r: [parse_coder_spans(r[c]) for c in CODER_COLS if pd.notna(r[c])], axis=1)
    df['values'] = df.apply(lambda r: (r['gt_labels'], *r['coder_labels']), axis=1)  # one unit per sentence
    return df.reset_index(drop=True)

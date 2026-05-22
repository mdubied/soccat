"""
table_manual_annotation_top_groups.py

Description:
Read manual annotation file and compute the top-N most frequent "specific_group_new" categories
separately for France and Germany, then export a side-by-side LaTeX table.

Input:
- data/manual_annotations/annotations_ground_truth.csv

Output:
- tables/manual_annotation_top_XX.tex   (XX = top_n argument)

Usage (from this directory):
python manual_annotation_top_groups.py --top_n 10
python manual_annotation_top_groups.py --top_n 20

"""

# ============================================================
# CONFIG
# ============================================================
import pandas as pd
import argparse
import os

DEFAULT_INPUT_PATH = "../data/manual_annotations/step_2/annotations_ground_truth.csv"
DEFAULT_OUT_DIR = "../tables"
DISCARD_CATEGORIES = [
    "others",
    "other",
    "other profession",
    "other professions",
    "target abroad",
]

# Maps raw specific_group_new values to canonical labels (aligned with ICR notebooks).
NORMALISE = {
    "poor": "lower class", "underprivileged": "lower class",
    "unskilled and/or underprivileged": "unskilled or unqualified",
    "qualified": "skilled or qualified", "qualified and graduates": "skilled or qualified",
    "investors and stakeholders": "capital owners, investors and shareholders",
    "employees": "wage and salary earners", "precarious employees": "wage and salary earners",
    "working active population": "wage and salary earners",
    "housewife and househusband": "housewives and househusbands",
    "self-employed/freelancers": "self-employed and freelancers",
    "leaders": "CEOs and corporate leaders",
    "ceos and corporate leaders": "CEOs and corporate leaders",
    "enterprises": "entrepreneurs", "large enterprises": "entrepreneurs",
    "small- and middle-size enterprises": "entrepreneurs",
    "specific sector": "entrepreneurs",
    "entrepreneurs (smes)": "entrepreneurs",
    "entrepreneurs (large enterprises)": "CEOs and corporate leaders",
    "entrepreneurs in [specific] sector": "entrepreneurs",
    "minors, including children and pupils": "minors",
    "youth, including students and apprentices": "youth",
    "middle-aged": "middle-aged and pre-retirement age groups",
    "older age group": "elderly", "seniors": "elderly",
    "immigrants": "people with an immigration background, including immigrants",
    "visible and ethnic minorities": "ethnic and racial minorities",
    "ethnic minorities": "ethnic and racial minorities",
    "minorities": "ethnic and racial minorities",
    "east germans": "ethnic and racial minorities",
    "west germans": "ethnic and racial minorities",
    "ethnic germans": "ethnic and racial minorities",
    "expatriates": "ethnic and racial minorities",
    "white": "ethnic and racial minorities",
    "white people": "ethnic and racial minorities",
    "visible minorities": "ethnic and racial minorities",
    "language and ethnic minorities": "ethnic and racial minorities",
    "lgbtqi*": "LGBTQIA+", "lgbtqqia+": "LGBTQIA+", "lgbtqia+": "LGBTQIA+",
    "cis & heterosexuals": "cisgender and heterosexuals",
    "religious groups": "multiple (or other) religious or minority groups",
    "religious minorities": "multiple (or other) religious or minority groups",
    "territorial language minorities": "multiple (or other) religious or minority groups",
    "multiple (or other specific) religious or minority groups":
        "multiple (or other) religious or minority groups",
    "disabled": "disabled people",
    "scientists": "scientists and professors",
    "prostitutes": "sex workers",
    "social professions": "other professions",
    "engineers": "other professions",
    "lobbyists": "other professions",
    "hunters": "other professions",
    "people working in the public sector": "civil servants",
    "commuters": "consumers and clients",
    "cyclists": "car drivers", "pedestrians": "car drivers",
    "road carriers": "consumers and clients",
    "air travellers": "consumers and clients",
    "users of certain transportation modes": "consumers and clients",
    "public transport passengers": "consumers and clients",
    "insured persons": "patients",
    "consumers": "consumers and clients",
    "tax evaders and white collar criminals":
        "offenders, criminals, prisoners and/or accused people",
    "offenders or criminals": "offenders, criminals, prisoners and/or accused people",
    "terrorists":
        "terrorists, rebels, revolutionaries and/or movements of armed resistance",
    "home owner": "real-estate owners", "land owner": "real-estate owners",
    "landlords": "real-estate owners", "real-estate owner": "real-estate owners",
    "real estate owners": "real-estate owners",
}

# ============================================================
# PARSING / COUNTING
# ============================================================

def parse_multi_categories(cell, sep=" ; "):
    """
    Parse a cell that may contain:
    - NaN / empty
    - one category
    - multiple categories separated by ' ; ' (with spaces)

    Returns a list of cleaned category strings.
    """
    if cell is None:
        return []

    if pd.isna(cell):
        return []

    s = str(cell).strip()
    if s == "":
        return []

    parts = s.split(sep)
    out = []
    for p in parts:
        p = str(p).strip()
        if p != "":
            out.append(p)
    return out


def count_categories_by_country(
    df,
    col_categories="specific_group_new",
    col_country="country",
    discard_categories=None,
):
    """
    Returns:
      counts = {
        "France": {category: count, ...},
        "Germany": {category: count, ...},
      }
    """
    if discard_categories is None:
        discard_set = set()
    else:
        discard_set = {str(x).strip() for x in discard_categories}

    counts = {"France": {}, "Germany": {}}

    for _, row in df.iterrows():
        country = row.get(col_country)
        if pd.isna(country):
            continue

        country = str(country).strip()
        if country not in counts:
            continue

        cats = parse_multi_categories(row.get(col_categories))
        for c in cats:
            c_clean = str(c).strip()
            if c_clean == "":
                continue
            c_clean = NORMALISE.get(c_clean.lower(), c_clean)
            if c_clean in discard_set:
                continue

            counts[country][c_clean] = counts[country].get(c_clean, 0) + 1

    return counts



def top_n_from_counts(counts_dict, n):
    """
    counts_dict: {category: count}
    Returns list of tuples: [(category, count), ...] length <= n
    Sorted by: count desc, category asc.
    """
    items = list(counts_dict.items())
    items.sort(key=lambda x: (-x[1], x[0]))
    return items[:n]


# ============================================================
# LATEX TABLE
# ============================================================

def latex_escape(s):
    """
    Minimal LaTeX escaping for common special characters.
    """
    if s is None:
        return ""
    s = str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def build_side_by_side_latex_table(top_fr, top_de, top_n, caption=None, label=None):
    """
    Creates LaTeX code for a 5-column table:
      Rank | (France: Category, #) | (Germany: Category, #)
    With merged headers for France and Germany.
    """
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabularx}{\textwidth}{r X r X r}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{2}{c}{France} & \multicolumn{2}{c}{Germany} \\")
    lines.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}")
    lines.append(r"Rank & Specific category & \# & Specific category & \# \\")
    lines.append(r"\midrule")

    # pad to top_n
    fr_pad = top_fr + [("", "")] * (top_n - len(top_fr))
    de_pad = top_de + [("", "")] * (top_n - len(top_de))

    for i in range(top_n):
        fr_cat, fr_ct = fr_pad[i]
        de_cat, de_ct = de_pad[i]

        fr_cat = latex_escape(fr_cat)
        de_cat = latex_escape(de_cat)

        fr_ct = "" if fr_ct == "" else str(fr_ct)
        de_ct = "" if de_ct == "" else str(de_ct)

        lines.append(f"{i+1} & {fr_cat} & {fr_ct} & {de_cat} & {de_ct} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")

    if caption:
        lines.append(rf"\caption{{{latex_escape(caption)}}}")
    if label:
        lines.append(rf"\label{{{latex_escape(label)}}}")

    lines.append(r"\end{table}")
    lines.append("")  # trailing newline

    return "\n".join(lines)


# ============================================================
# FULL PIPELINE
# ============================================================

def generate_top_groups_table(
    input_path=DEFAULT_INPUT_PATH,
    out_dir=DEFAULT_OUT_DIR,
    top_n=10,
    col_categories="specific_group_new",
    col_country="country",
):
    df = pd.read_csv(input_path)

    counts = count_categories_by_country(
        df,
        col_categories=col_categories,
        col_country=col_country,
        discard_categories=DISCARD_CATEGORIES,
    )

    top_fr = top_n_from_counts(counts.get("France", {}), top_n)
    top_de = top_n_from_counts(counts.get("Germany", {}), top_n)

    latex = build_side_by_side_latex_table(
        top_fr=top_fr,
        top_de=top_de,
        top_n=top_n,
        caption=f"Top {top_n} most frequent annotated specific categories for France and Germany. The numbers indicate how many sentences contain at least one mention of the respective categories. Residual categories are excluded.",
        label=f"tab:manual-annotation-top-{top_n}",
    )

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"manual_annotation_top_{top_n}.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex)

    print(f"[saved] {out_path}")

    return counts


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--input_path", type=str, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    generate_top_groups_table(
        input_path=args.input_path,
        out_dir=args.out_dir,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()

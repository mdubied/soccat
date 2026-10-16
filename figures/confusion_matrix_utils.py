"""
confusion_matrix_utils.py

Confusion matrices and outcome tables by broad class, shared by
tables/icr_step2_krippendorff.ipynb (ground truth vs independent coders) and
figures/step_2_confusion_matrix.py (ground truth vs SOCCAT), so that both sets of
figures are built and drawn identically.

A unit is a pair (ground-truth labels, compared labels) of label sets for one
sentence, the compared side being an independent coder or the model.
"""
import re
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

import utils as su  # font configuration shared with the other paper figures

OTHER, NONE, OTHERS = 'Other broad class', 'None', 'Others'  # OTHERS: the residual label 'others'
RESIDUAL = 'others'
CELL_CM = 0.66            # cell size (largest matrix, Profession, about 14 cm wide), the same in every figure so that font sizes match
WRAP_CHARS = 38
TEXTWIDTH_CM = 14.10     # LaTeX \textwidth, for the figure widths written to figure_widths.txt
DISPLAY = {  # shorter display names, as in figures/step_2_boxplot.py
    'lgbtqia+': 'LGBTQIA+',
    'people with an immigration background, including immigrants': 'people with immigration background',
    'offenders, criminals, prisoners and/or accused people': 'offenders, criminals, prisoners, accused people',
    'terrorists, rebels, revolutionaries and/or movements of armed resistance':
        'terrorists, revolutionaries, rebels, armed resistance',
}

# What the compared side used for each ground-truth label, in this order of precedence
OUTCOMES = ['same label', 'other label, same class', 'other broad class',
            'other broad class + others', 'others', 'none']


def class_slug(broad: str) -> str:
    return re.sub(r'[^a-z]+', '_', broad.lower()).strip('_')


def display_label(label: str, width: int = WRAP_CHARS) -> str:
    text = DISPLAY.get(label, label)
    text = text[:1].upper() + text[1:]
    return '\n'.join(textwrap.wrap(text, width))


def class_confusion(units, class_labels: list) -> pd.DataFrame:
    """Ground truth (rows) x compared side (columns) for one broad class.

    Specific labels of other broad classes are collapsed into OTHER, the residual label
    'others' into OTHERS; units where neither side uses a label of the class are skipped.
    Labels used by both sides are matched first (diagonal) and never paired with anything
    else. Each side's unmatched labels of the class are paired with the other side's
    unmatched labels of the class or, if there are none, with the other side's unmatched
    labels outside the class: OTHER and/or OTHERS (both cells if both), or NONE if the
    other side has no unmatched label at all (row NONE: label added by the compared side;
    column NONE: ground-truth label missed).
    """
    in_class = set(class_labels)
    order = class_labels + [OTHER, OTHERS, NONE]
    idx = {l: i for i, l in enumerate(order)}
    mat = np.zeros((len(order), len(order)), dtype=np.int64)
    for gt, coder in units:
        g, c = gt & in_class, coder & in_class
        if not g and not c:
            continue
        for l in g & c:
            mat[idx[l], idx[l]] += 1
        g_only, c_only = g - c, c - g
        def fill(labels, other):  # what one side used instead, among its UNMATCHED labels
            out = (labels - other) - in_class
            return ([OTHER] if out - {RESIDUAL} else []) + ([OTHERS] if RESIDUAL in out else []) or [NONE]
        rows = [idx[l] for l in g_only] or ([idx[f] for f in fill(gt, coder)] if c_only else [])
        cols = [idx[l] for l in c_only] or ([idx[f] for f in fill(coder, gt)] if g_only else [])
        for r in rows:
            for k in cols:
                mat[r, k] += 1
    return pd.DataFrame(mat, index=order, columns=order)


def plot_class_confusion(df: pd.DataFrame, path: Path, xlabel: str = 'Independent coder') -> None:
    su.configure_fonts()
    counts = df.values.astype(float)
    row_tot = counts.sum(axis=1, keepdims=True)
    share = np.divide(counts, row_tot, out=np.zeros_like(counts), where=row_tot > 0)
    n = len(df)
    side = n * CELL_CM / 2.54
    fig, ax = plt.subplots(figsize=(side, side))
    # vector cells with their own white border (pcolormesh): no image resampling, so cell
    # edges stay sharp in every PDF viewer; empty cells are left white
    ax.pcolormesh(np.arange(n + 1) - 0.5, np.arange(n + 1) - 0.5,
                  np.ma.masked_where(counts == 0, share), cmap='Oranges', norm=Normalize(0, 1),
                  edgecolors='white', linewidth=0.6)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)  # first label at the top
    ax.set_aspect('equal')
    # bottom-right block (other broad class / others / none on both sides) is empty by
    # construction: such units do not involve the class and are skipped
    ax.add_patch(plt.Rectangle((n - 3.5, n - 3.5), 3, 3, facecolor='#EEEEEE', edgecolor='none', zorder=0.5))
    for i in range(n):
        for j in range(n):
            if counts[i, j] > 0:
                ax.text(j, i, f'{int(counts[i, j])}', ha='center', va='center',
                        fontsize=7 if counts[i, j] < 1000 else 6,  # 4-digit counts must fit the cell
                        color='white' if share[i, j] > 0.55 else 'black')
    labels = [display_label(l) for l in df.index]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_linespacing(1.0)  # two-line labels must fit one cell
    ax.tick_params(length=0)
    ax.axhline(n - 3.5, color='black', linewidth=0.8)  # separates the class's own labels
    ax.axvline(n - 3.5, color='black', linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Ground truth')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'[saved] {path}')


def write_class_confusions(units, codebook: dict, out_dir: Path, xlabel: str = 'Independent coder') -> dict:
    """One CSV + PDF per broad class (residual 'Others' class excluded: it has its own
    row/column, OTHERS, in every matrix); returns {broad class: matrix}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {}
    for broad, lbls in codebook.items():
        if broad == 'Others':
            continue
        df_cm = class_confusion(units, list(lbls))
        tables[broad] = df_cm
        df_cm.to_csv(out_dir / f'confusion_{class_slug(broad)}.csv')
        plot_class_confusion(df_cm, out_dir / f'confusion_{class_slug(broad)}.pdf', xlabel=xlabel)
    return tables


def pdf_size_cm(path: Path) -> tuple[float, float]:
    """Width and height of the saved (tight-cropped) PDF page, from its MediaBox (pt)."""
    x0, y0, x1, y1 = map(float, re.search(rb'/MediaBox \[([^\]]+)\]', path.read_bytes()).group(1).split())
    return (x1 - x0) / 72 * 2.54, (y1 - y0) / 72 * 2.54


def write_figure_widths(out_dir: Path) -> str:
    """figure_widths.txt: width/height of each confusion_*.pdf in cm and as a share of \\textwidth."""
    width_lines = [f'Confusion matrix PDF sizes (\\textwidth = {TEXTWIDTH_CM:.2f} cm)', '',
                   f'{"file":<66} {"width cm":>8} {"height cm":>9} {"share of textwidth":>18}']
    for pdf in sorted(out_dir.glob('confusion_*.pdf')):
        w, h = pdf_size_cm(pdf)
        width_lines.append(f'{pdf.name:<66} {w:>8.2f} {h:>9.2f} {w / TEXTWIDTH_CM:>18.3f}')
    text = '\n'.join(width_lines)
    (out_dir / 'figure_widths.txt').write_text(text + '\n', encoding='utf-8')
    return text


def label_outcome(label, class_labels, coder, gt):
    """What the compared side used for one ground-truth label (see OUTCOMES), among the
    labels it used that are NOT in the ground truth: a correctly found label elsewhere in
    the sentence never counts as a substitution, so a label missed without any wrong label
    in its place is 'none'."""
    if label in coder:
        return 'same label'
    wrong = coder - gt
    if wrong & class_labels:
        return 'other label, same class'
    out = wrong - class_labels
    specific, residual = bool(out - {RESIDUAL}), RESIDUAL in out
    if specific and residual:
        return 'other broad class + others'
    if specific:
        return 'other broad class'
    return 'others' if residual else 'none'


def outcome_table(units, codebook: dict) -> pd.DataFrame:
    """Counts of OUTCOMES per broad class (one unit per ground-truth label x unit) plus an
    'All' row, n, and each disagreement outcome as a share of the disagreements."""
    rows = []
    for broad, lbls in codebook.items():
        if broad == 'Others':
            continue
        in_class = set(lbls)
        counts = Counter(label_outcome(l, in_class, coder, gt)
                         for l in lbls for gt, coder in units if l in gt)
        rows.append({'Broad class': broad, **{o: counts.get(o, 0) for o in OUTCOMES}})
    df = pd.DataFrame(rows).set_index('Broad class')
    df.loc['All'] = df.sum()
    df['n'] = df[OUTCOMES].sum(axis=1)
    disagree = df[OUTCOMES[1:]].sum(axis=1)
    for o in OUTCOMES[1:]:  # share of the disagreements (units where the label was not used)
        df[f'{o} (% of disagreements)'] = (100 * df[o] / disagree).round(1)
    return df


def outcome_summary(df: pd.DataFrame, who: str = 'independent coder') -> str:
    """Text summary of outcome_table's 'All' row; `who` names the compared side."""
    short = who.split()[-1]  # 'coder' / 'SOCCAT'
    subject = f'the {short}' if short.islower() else short  # 'the coder', but 'SOCCAT'
    tot = df.loc['All']
    n_dis = int(tot[OUTCOMES[1:]].sum())
    lines = [
        f'{who[:1].upper() + who[1:]} vs ground truth, per ground-truth label',
        f'(units: ground-truth label x sentence x {who}; residual "Others" class excluded)',
        '',
        f'Ground-truth label units: {int(tot["n"]):,}',
        f'Same label used by {subject}: {int(tot["same label"]):,} ({100 * tot["same label"] / tot["n"]:.1f}%)',
        f'Disagreements: {n_dis:,} ({100 * n_dis / tot["n"]:.1f}%), of which:',
    ]
    lines += [f'  {o:<28} {int(tot[o]):>5}  {100 * tot[o] / n_dis:5.1f}%' for o in OUTCOMES[1:]]
    lines += ['', f'{short[:1].upper() + short[1:]} used the residual "others" (alone or with a label of another broad class): '
              f'{int(tot["others"] + tot["other broad class + others"]):,} '
              f'({100 * (tot["others"] + tot["other broad class + others"]) / n_dis:.1f}% of disagreements)']
    return '\n'.join(lines)

# Measuring Discursive Constructions of Social Categories in the Press

## Description

This repository accompanies the paper "**Detecting Social Categories in Multilingual Newspaper
Corpora**" (2026) by S. Šarenkapa, I. Guinaudeau, E. Deiss-Helbig, M. Dubied, R. Heiberger, and T. Matthieß.

It provides the codebase required to reproduce the results reported in the paper.

## How to run the code

To prepare your working environment, follow these steps (only required once):

1. Install Python

Make sure Python 3.10+ is installed (3.12 recommended):
```
python --version
```

2. Clone this repository

Either download the project, or use the terminal:
```
git clone https://github.com/mdubied/social-categorization.git
cd social-categorization
```

3. Create a virtual environment

From the repository root:
```
python -m venv venv
```

4. Activate the virtual environment

Windows
```
venv\Scripts\Activate
```

macOS / Linux
```
source venv/bin/activate
```

5. Install required Python packages
```
pip install -r requirements.txt
```

To run the code, navigate to the repository root and open the desired script. Each script contains inline documentation describing its purpose, expected inputs, and configurable parameters.

## Structure of the repository

The repository is structured as follows:

```
.
├── data/                          data used to produce results
│   ├── annotated_corpus/          aggregated corpus statistics by outlet and year
│   ├── manual_annotations/        raw and pre-processed manual annotation files
│   └── model_performance/         model evaluation outputs
│
├── figures/                       scripts and PDF outputs for all paper figures
│   ├── findings_fig1_broad_classes.py      figure 2 — broad category distributions
│   ├── findings_fig2_top_categories.py     figure 4 — top 20 categories
│   ├── findings_fig3_selected_groups_trends.py  figure 5 — group trends over time
│   ├── step_1_heatmap.py                   figure A1 — step 1 performance heatmap
│   ├── step_2_heatmap.py                   figure A10-A13 — step 2 performance heatmaps
│   ├── step_2_boxplot.py                   figure 3, A2-9 — step 2 performance boxplots
│   ├── utils.py                            shared plotting utilities
│   ├── findings/                           PDF outputs for findings figures
│   ├── step_1/                             PDF output for step 1
│   └── step_2/                             PDF outputs for step 2
│
├── src/                           model training and pipeline code
│   ├── step_1/
│   │   └── SOCCAT_mDeBERTa_replication.py/.ipynb   step 1 replication binary classifier (mDeBERTa)
│   └── step_2/
│       ├── replicate_from_hub_step_2.py/.ipynb     step 2 replication from Hugging Face Hub
│       ├── step_2_cv_pipeline.py/.ipynb            step 2 cross-validation pipeline
│       └── convert_annotations.py/.ipynb           converts annotations to NLI pairs
│
├── tables/                        scripts and outputs for all paper tables
│   ├── manual_annotation_top_groups.py     table A2 — top annotated groups per country
│   ├── manual_annotation_top_10.tex        generated LaTeX output
│   ├── icr_step1_krippendorff.ipynb        table 3 — inter-coder reliability, step 1
│   ├── icr_step2_krippendorff.ipynb        table 3 — inter-coder reliability, step 2
│   ├── icr_step1_by_outlet.csv             ICR step 1 results by outlet
│   ├── icr_step1_by_annotator_pair.csv     ICR step 1 pairwise results
│   └── icr_step2_by_outlet.csv             ICR step 2 results by outlet
```

## Citations

If you use this codebase, cite us as follows:

TODO: add exact citation
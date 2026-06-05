# SOCCAT: Detecting Social Categories in Multilingual Newspaper Corpora

## Description

This repository accompanies the paper "**Detecting Social Categories in Multilingual Newspaper Corpora**" (2026) by S. Šarenkapa, I. Guinaudeau, E. Deiss-Helbig, M. Dubied, R. Heiberger, and T. Matthieß. [Preprint on OSF/SocArXiv](https://osf.io/preprints/socarxiv/bmqxt_v1)

It provides the codebase required to reproduce the results reported in the paper. The pipeline detects mentions of social groups in French and German sentences using transformer-based NLP models. It has been trained and validated using representative French and German newspaper articles. It has two stages:

1. **Step 1 — Mention detection.** A binary classifier (fine-tuned mDeBERTa) predicts whether a sentence mentions any social group.
2. **Step 2 — Category classification.** For sentences flagged in step 1, eight per-category NLI classifiers predict which specific social groups are mentioned. The taxonomy covers 57 categories (labels) across 8 broad classes.

All fine-tuned models are publicly available at https://huggingface.co/selsar.

## Quick start

For a short end-to-end demonstration of the pipeline on five example sentences, open `demo/demo.ipynb` after completing the setup steps below. The demo runs three of the eight categories and is the recommended starting point.

## Setup

These steps prepare your working environment and only need to be run once.

1. **Install Python 3.10 or higher** (3.12 recommended):
   ```
   python3 --version
   ```

2. **Clone this repository:**
   ```
   git clone https://github.com/mdubied/social-categorization.git
   cd social-categorization
   ```

3. **Create a virtual environment** from the repository root:
   ```
   python3 -m venv venv
   ```

4. **Activate the virtual environment:**

   macOS / Linux:
   ```
   source venv/bin/activate
   ```

   Windows:
   ```
   venv\Scripts\Activate
   ```

5. **Install required Python packages:**
   ```
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Running the code

The jupyter notebook `demo/demo.ipynb` shows an example on how to use the SOCCAT pipeline on your own sentences.

To reproduce results from the paper, each script contains inline documentation describing its purpose, expected inputs, and configurable parameters. Aside from the tables (`tables/`) and figures (`figures/`) contained in the paper, the code contains replication code for the SOCCAT pipeline itself in `src/`. SOCCAT relies on two main steps:

**Step 1 — Mention detection.** Reproduce the evaluation reported in the paper without retraining:
```
python src/step_1/SOCCAT_mDeBERTa_replication.py \
  --test_path data/model_performance/step_1/test_with_all_outlets.json \
  --output_dir output/step_1 \
  --inference_only
```

**Step 2 — Category classification.** First generate the per-category NLI pair CSVs from the ground-truth annotations, then run inference using the models on the Hub:
```
cd src/step_2/
python convert_annotations.py
python replicate_from_hub_step_2.py \
  --data_root nli_pairs_by_category/ \
  --out output/step_2 \
  --batch_size 8
```

To run a single category instead of all eight, add `--category <name>` where `<name>` is one of the eight category names listed in `src/step_2/categories.json`. Use `--help` on any script to see all available flags.

## Repository structure

```
.
├── demo/                          short end-to-end demo of the pipeline
│   └── demo.ipynb                 5 example sentences through steps 1 and 2
│
├── data/                          data used to produce results
│   ├── annotated_corpus/          aggregated corpus statistics by outlet and year
│   ├── manual_annotations/        raw and pre-processed manual annotation files
│   └── model_performance/         model evaluation outputs
│
├── figures/                       scripts and PDF outputs for all paper figures
│   ├── findings_fig1_broad_classes.py        figure 2 — broad category distributions
│   ├── findings_fig2_top_categories.py       figure 4 — top 20 categories
│   ├── findings_fig3_selected_groups_trends.py  figure 5 — group trends over time
│   ├── step_1_heatmap.py                     figure A1 — step 1 performance heatmap
│   ├── step_2_heatmap.py                     figures A10–A13 — step 2 performance heatmaps
│   ├── step_2_boxplot.py                     figure 3, A2–A9 — step 2 performance boxplots
│   ├── utils.py                              shared plotting utilities
│   ├── findings/                             PDF outputs for findings figures
│   ├── step_1/                               PDF output for step 1
│   └── step_2/                               PDF outputs for step 2
│
├── src/                           model training and pipeline code
│   ├── step_1/
│   │   └── SOCCAT_mDeBERTa_replication.py
│   │           Fine-tunes mDeBERTa for binary mention detection and evaluates
│   │           against held-out data. Use --inference_only to skip training and
│   │           load the published model from the Hub.
│   │
│   └── step_2/
│       ├── categories.json
│       │       Manifest listing the 8 categories, their labels, and the
│       │       corresponding Hugging Face Hub repositories. Consumed by
│       │       replicate_from_hub_step_2.py.
│       ├── convert_annotations.py
│       │       Converts the ground-truth annotation file into 8 per-category
│       │       NLI pair CSVs (sentence × label pairs with entailment labels).
│       │       Must be run before replicate_from_hub_step_2.py.
│       ├── replicate_from_hub_step_2.py
│       │       Loads each fine-tuned NLI model from the Hub and runs inference
│       │       on the per-category CSVs. Produces per-category metrics and a
│       │       cross-category summary.
│       └── step_2_cv_pipeline.py
│               Fine-tunes a category-specific NLI classifier with stratified
│               5-fold cross-validation. Used to produce the published models.
│               Optional Hub upload via --push_to_hub.
│
├── tables/                        scripts and outputs for all paper tables
│   ├── manual_annotation_top_groups.py       table A2 — top annotated groups per country
│   ├── manual_annotation_top_10.tex          generated LaTeX output
│   ├── icr_step1_krippendorff.ipynb          table 3 — inter-coder reliability, step 1
│   ├── icr_step2_krippendorff.ipynb          table 3 — inter-coder reliability, step 2
│   ├── icr_step1_by_outlet.csv               ICR step 1 results by outlet
│   ├── icr_step1_by_annotator_pair.csv       ICR step 1 pairwise results
│   └── icr_step2_by_outlet.csv               ICR step 2 results by outlet
```

## Troubleshooting

**`pyexpat` import error on macOS Tahoe (`Symbol not found: _XML_SetAllocTrackerActivationThreshold`).** Some Homebrew Python builds are linked against a newer expat than the one shipped with macOS Tahoe. Install Homebrew's expat and rebind the pyexpat module:
```
brew install expat
install_name_tool -change /usr/lib/libexpat.1.dylib "$(brew --prefix expat)/lib/libexpat.1.dylib" \
  $(python3 -c "import pyexpat; print(pyexpat.__file__)")
codesign --force --sign - $(python3 -c "import pyexpat; print(pyexpat.__file__)")
```

**Out-of-memory or `zsh: killed` during inference.** The default batch sizes are tuned for CPU/MPS Macs with 16 GB RAM. If inference is killed by the OS, reduce the batch size further with `--batch_size 4` (or 2 if needed).

## Citation

If you use this codebase, please cite our paper. The [preprint](https://osf.io/preprints/socarxiv/bmqxt_v1) can be cited as:

```bibtex
@misc{sarenkapa_detecting_2026,
	title = {Detecting {Social} {Categories} in {Multilingual} {Newspaper} {Corpora}},
	url = {https://osf.io/preprints/socarxiv/bmqxt_v1/},
	urldate = {2026-06-05},
	publisher = {SocArXiv},
	author = {Šarenkapa, Selma and Guinaudeau, Isabelle and Deiss-Helbig, Elisa and Dubied, Mathieu and Heiberger, Raphael and Matthieß, Theres},
	month = jun,
	year = {2026},
	keywords = {multilingual text classification, natural language inference, social categories, text as data},
}
```

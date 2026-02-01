# Measuring Discursive Constructions of Social Categories in the Press

## Description

This repository accompanies the paper "**Measuring Discursive Constructions of Social Categories in the Press**" (2025) by S. Šarenkapa, I. Guinaudeau, E. Deiss-Helbig, M. Dubied, R. Heiberger, and T. Matthieß.

It provides the trained models as well as the codebase required to reproduce the results reported in the paper.

## How to run the code

To prepare your working environment, follow these steps (only required once):

1. Install Python

Make sure Python 3.9+ is installed:
```
python --version
```

2. Clone this repository

Either download the project, or use the terminal:
```
git clone https://github.com/<your-organization>/<your-repository>.git
cd <your-repository>
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
├── data: contains data used to produce the figures and tables of the paper.
│   ├── manual_annotations
│   ├── step_1
│   └── step_2
│
├── figures: contains the figures of the paper.
│   ├── step_1
│   └── step_2
│       ├── boxplots
│       └── heatmaps
│
├── step_1_heatmap.py: used to create figure XY in the paper
├── step_2_boxplot.py: used to create figure XY in the paper
├── step_2_heatmap.py: used to create figure XY in the paper
│
├── tables: contains the table of the paper.
│   ├── manual_annotation_top_10.tex
│   └── table_manual_annotation_top_groups.py: used to create figure XY in the paper
```

## Citations

If you use this codebase, cite us as follows:
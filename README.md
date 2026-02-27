# Inferring Map Generalization Operations from User Prompts

Master’s Thesis · MSc Geoinformatics Engineering  
Politecnico di Milano (2025–2026), in collaboration with the University of Bonn  
Author: Amirhossein Donyadidegan

---

## Overview

This repository investigates a practical question in AI-assisted cartography:

> Can we infer **which deterministic map generalization operator** to apply (and with what parameter) from a **natural-language prompt**?

Instead of learning geometry transformations end-to-end, this project predicts:

1. a classical operator (classification), and
2. its parameter value (operator-specific regression),

then applies the selected geometric algorithm deterministically.

This preserves interpretability, reproducibility, and compatibility with established cartographic workflows.

---

## What is implemented

The codebase includes:

- user-study data preparation utilities,
- prompt preprocessing,
- geometric feature extraction,
- text + map embeddings,
- multimodal model training,
- experiment artifacts and trained bundles,
- a local FastAPI demo web application.

Core package: `imgofup` (in `src/`).

---

## Supported generalization operators

The current system predicts and applies four operators:

- **Selection**
- **Simplification**
- **Aggregation**
- **Displacement**

Each operator is parameterized and executed with deterministic geometric logic.

---

## Repository structure

```text
.
├── src/imgofup/
│   ├── config/          # constants and path management
│   ├── datasets/        # data loading, labels, splits
│   ├── embeddings/      # prompt/map embeddings + fusion
│   ├── features/        # handcrafted polygon/map descriptors
│   ├── models/          # classifier/regressor training + bundle save
│   ├── pipelines/       # end-to-end experiment scripts
│   ├── preprocessing/   # cleaning and normalization utilities
│   ├── userstudy/       # sample generation and prompt cleaning tools
│   └── webapp/          # FastAPI API + static frontend
├── models/              # trained model bundles
├── notebooks/           # research and experiment notebooks
├── docs/                # project website assets
├── pyproject.toml
└── README.md
```

---

## Requirements

- Python **3.10+**
- pip (or uv)

Main dependencies are managed in `pyproject.toml` and include `geopandas`, `shapely`, `scikit-learn`, `fastapi`, and `uvicorn`.

---

## Installation

From repository root:

```bash
# create and activate your virtual environment first (recommended)
pip install -e .
```

For optional groups:

```bash
# notebook/dev utilities
pip install -e .[dev]

# optional embedding-related extras
pip install -e .[use]
```

---

## Quickstart

### 1) Run feature/embedding pipelines

```bash
python -m imgofup.pipelines.run_map_embeddings
python -m imgofup.pipelines.run_prompt_embeddings
python -m imgofup.pipelines.run_concat_features
```

### 2) Train models

```bash
python -m imgofup.models.train_classifier
python -m imgofup.models.train_regressors
```

### 3) Save reusable model bundle

```bash
python -m imgofup.models.save_bundle
```

---

## Run the local web app

Start the FastAPI app:

```bash
python -m imgofup.webapp.app
```

Then open:

- App / UI: `http://127.0.0.1:8000/`
- API docs: `http://127.0.0.1:8000/docs`

The app loads model bundles from the top-level `models/` directory.

---

## Experimental framing

The project evaluates three settings:

- **Prompt-only**
- **Geometry-only**
- **Multimodal (prompt + geometry)**

High-level finding: prompt semantics are strong for operator selection, while geometric context is especially useful for parameter estimation.

---

## Notes

- This repository is research-oriented and thesis-driven; interfaces may evolve as experiments progress.
- For module-level details, see `src/README.md`.
- For web app details, see `src/imgofup/webapp/README.md`.

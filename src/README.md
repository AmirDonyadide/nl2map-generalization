This directory contains the complete implementation of the nl2map-generalization pipeline, including:

- Dataset preparation
- Feature extraction
- Multimodal embeddings
- Model training (classification + regression)
- Evaluation
- User study utilities
- FastAPI web application backend

The code is structured as a modular Python package: `imgofup`.

---

# Package Structure

```text
src/
├── imgofup/
│   ├── config/
│   ├── datasets/
│   ├── embeddings/
│   ├── features/
│   ├── models/
│   ├── pipelines/
│   ├── preprocessing/
│   ├── userstudy/
│   ├── webapp/
│   └── __init__.py
└── README.md
```

The system follows a data → features → embeddings → fusion → model → execution workflow.

---

# Module Overview

## config/

Central configuration utilities:

- `constants.py` – global constants (operators, parameter ranges, labels)
- `paths.py` – filesystem path management

Purpose:
- Avoid hard-coded paths
- Keep experiment configuration consistent
- Ensure reproducibility across training and inference

---

## datasets/

Dataset construction and loading utilities:

- `load_training_data.py`
- `splitting.py`
- `labels_and_weights.py`
- `utils/`

Responsibilities:
- Load cleaned user study data
- Generate operator labels and regression targets
- Create train/validation/test splits
- Handle class balancing

---

## features/

Handcrafted geometric feature extraction:

- `polygons.py`
- `pooling.py`

Computes map-level descriptors such as:

- Number of buildings
- Mean area
- Compactness
- Density
- Shape statistics
- Spatial dispersion metrics

These features encode geometric structure of map tiles.

---

## embeddings/

Multimodal representation building:

- `maps.py` – geometric feature embeddings
- `prompts.py` – textual embeddings
- `concat.py` – fusion utilities

Responsibilities:
- Encode user prompts into vector space
- Transform map descriptors into feature vectors
- Concatenate representations for multimodal learning

Supports:
- Prompt-only
- Geometry-only
- Multimodal configurations

---

## preprocessing/

Data cleaning and normalization:

- `preprocessing.py`
- `utils/`

Includes:
- Prompt normalization
- Text cleaning
- Feature scaling
- Missing value handling

Ensures consistent input to downstream models.

---

## models/

Training and persistence:

- `train_classifier.py`
- `train_regressors.py`
- `save_bundle.py`
- `utils/`

Responsibilities:
- Operator classification (multi-class)
- Parameter regression (operator-specific)
- Model serialization
- Bundle export for inference

Models are saved in a reusable format for integration in the web application.

---

## pipelines/

High-level experiment orchestration:

- `run_concat_features.py`
- `run_map_embeddings.py`
- `run_prompt_embeddings.py`

These scripts execute complete workflows:

1. Load dataset
2. Extract embeddings
3. Train models
4. Evaluate performance
5. Store results

This directory reflects the experimental setups described in the thesis.

---

## userstudy/

Utilities used during dataset generation:

- `sample_generation.py`
- `tiling.py`
- `rendering.py`
- `prompt_cleaning.py`
- `operators.py`
- `param_search.py`

Responsibilities:
- Generate map tiles
- Apply controlled generalisation operators
- Render stimuli images
- Clean collected prompts
- Search parameter ranges

This module supports the User Study and Dataset Construction chapter.

---

## webapp/

FastAPI backend for interactive demo:

- `app.py`
- `api.py`
- `schemas.py`
- `services/`
- `frontend/`

Provides:
- Model loading
- Inference endpoints
- GeoJSON input handling
- Operator prediction
- Parameter prediction
- Execution of deterministic geometric operators

This module connects the trained models to an interactive prototype system.

---

# Conceptual Workflow

```text
User Study Data
      ↓
Preprocessing
      ↓
Feature Extraction (maps)
Text Embedding (prompts)
      ↓
Fusion
      ↓
Classifier + Regressor
      ↓
Model Bundle
      ↓
Web Application Inference
```

---

# Design Principles

The `src/` architecture follows these principles:

- Modularity – clear separation between features, embeddings, models, and pipelines
- Reproducibility – central configuration and deterministic splits
- Interpretability – structured prediction (operator + parameter)
- Extensibility – easy addition of new operators or embedding strategies
- Separation of concerns – research code vs. web application logic

---

# Usage Examples

Train models via pipeline:

```bash
python -m imgofup.pipelines.run_concat_features
```

Train classifier only:

```bash
python -m imgofup.models.train_classifier
```

Train regressors:

```bash
python -m imgofup.models.train_regressors
```

Run web application:

```bash
uvicorn imgofup.webapp.app:app --reload
```

---

# Relation to Thesis

This directory implements the methodology described in:

- Methodology
- User Study and Dataset Creation
- Experimental Results
- Web Application Prototype

It constitutes the full reproducible baseline for:

Inferring Map Generalisation Operations from User Prompts

---

# Maintainer

Amirhossein Donyadidegan  
MSc Geoinformatics Engineering  
Politecnico di Milano


# Inferring Map Generalisation Operations from User Prompts

Master’s Thesis  
MSc Geoinformatics Engineering  
Politecnico di Milano (2025–2026)  
In collaboration with the University of Bonn  

Author: Amirhossein Donyadidegan  

---

## Abstract

Cartographic map generalisation enables spatial data to be adapted to different visualisation scales and purposes; however, the effective use of many generalisation algorithms requires expert knowledge, particularly for operator selection and parameter tuning, making the process difficult and unintuitive for non-expert users.

This thesis proposes a supervised machine learning workflow that links natural-language user prompts to established geometric generalisation algorithms. Rather than replacing established methods, the approach combines their interpretability and robustness with the flexibility of multimodal learning.

A dedicated dataset is constructed through a controlled user study in which participants describe, in natural language, how pairs of input and generalised building maps are related. Each sample is annotated with a generalisation operator and its parameter value, enabling supervised learning for both operator classification and parameter regression.

The multimodal input combines semantic embeddings of user prompts with handcrafted geometric descriptors extracted from map tiles. Different embedding strategies are evaluated, and a multi-layer perceptron(MLP) architecture is employed to predict operator type and parameter magnitude. Experiments compare prompt-only, geometry-only, and multimodal configurations.

Results indicate that operator selection can be reliably inferred from natural-language prompts, while accurate parameter estimation benefits from incorporating geometric context. A prototype web application further demonstrates the integration of the proposed models into an interactive, prompt-driven generalisation workflow for non-expert users.

---

## Research Motivation

Recent work in AI-based cartography attempts to learn full geometry transformations end-to-end. While powerful, such approaches:

- Require large training datasets
- Lack interpretability
- Discard decades of cartographic algorithmic research

This thesis explores an alternative paradigm:

> Learn *which* algorithm to apply and *how* to parameterise it —  
> not how to regenerate geometry from scratch.

This preserves:

- Algorithmic transparency  
- Deterministic execution  
- Reproducibility  
- Scientific interpretability  

---

## Problem Formulation

Given:

- A vector map tile (building footprints)
- A natural-language user prompt

Predict:

1. The appropriate generalisation operator  
2. The corresponding parameter value  

Then execute the deterministic geometric operator to generate the final output.

This is formulated as a **multimodal structured prediction task** consisting of:

- Multi-class classification (operator)
- Operator-specific regression (parameter)

---

## Methodological Framework

The system follows a modular pipeline:

### 1. Input
- Polygonal building map tiles
- Free-form natural-language prompts

### 2. Representation

#### Geometric Embedding
Handcrafted spatial descriptors extracted from vector data, including:
- Feature counts
- Area statistics
- Density measures
- Compactness indicators
- Spatial proximity metrics

#### Textual Embedding
Semantic sentence embeddings derived from:
- Pre-trained sentence encoders
- Transformer-based models

### 3. Fusion
Concatenation of geometric and textual representations into a joint feature vector.

### 4. Prediction
- Operator classification (multi-class)
- Parameter regression (continuous)

### 5. Execution
Application of deterministic geometric algorithms implemented with:

- GeoPandas
- Shapely

---

## Supported Generalisation Operators

The system currently supports four classical operators:

- **Selection** – removal of less relevant features
- **Simplification** – reduction of geometric detail
- **Aggregation** – merging of nearby features
- **Displacement** – spatial adjustment to resolve conflicts

Each operator is parameterised and executed deterministically.

---

## Dataset

A dedicated dataset was constructed through a controlled user study.

Participants were shown:
- An input building map
- A generalised output

They were asked to describe, in natural language, how the transformation could be achieved.

Each sample contains:

- Map tile geometry
- User prompt
- Labelled operator
- Corresponding parameter value

This enables supervised learning for:

- Operator classification
- Parameter regression

---

## Experimental Design

Three experimental configurations are evaluated:

### 1. Prompt-Only
Uses textual embeddings exclusively.

### 2. Geometry-Only
Uses handcrafted spatial descriptors exclusively.

### 3. Multimodal
Combines both representations.

### Key Findings

- Operator selection is primarily driven by textual semantics.
- Parameter estimation benefits significantly from geometric context.
- Multimodal models outperform unimodal baselines.

---

## Repository Structure

```
.
├── data/                # Dataset and processed tiles
├── src/
│   ├── embeddings/      # Text and geometry embedding modules
│   ├── features/        # Geometric feature extraction
│   ├── models/          # MLP architectures and training logic
│   ├── pipeline/        # Training and inference scripts
│   ├── generalisation/  # Deterministic operator implementations
│   └── utils/
├── app/                 # FastAPI web prototype
├── notebooks/           # Exploratory analysis
├── results/             # Experimental outputs
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/AmirDonyadide/nl2map-generalization
cd nl2map-generalization
```

Create environment and install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Train the model

```bash
python src/pipeline/train.py
```

### Run inference

```bash
python src/pipeline/inference.py
```

### Launch the web application

```bash
uvicorn app.main:app --reload
```

Then open:

```
http://127.0.0.1:8000
```

---

## Web Application Prototype

An interactive FastAPI-based prototype demonstrates:

- Upload of GeoJSON building datasets
- Free-text prompt input
- Model selection
- Operator and parameter prediction
- Visualisation of generalised output

The prototype illustrates the practical applicability of the trained models for non-expert users.

---

## Reproducibility

This repository includes:

- Full training pipeline
- Feature extraction code
- Model definitions
- Experimental configurations
- Web inference pipeline

All deterministic geometric operators are implemented explicitly and do not rely on black-box transformations.

Random seeds can be fixed for reproducibility.

---

## Limitations

- Assumes a single dominant operator per sample
- Limited to polygonal building data
- Simple feature concatenation used for multimodal fusion
- Dataset size constrained by user study scope

---

## Future Work

- Multi-operator sequential workflows
- Graph-based map embeddings
- Advanced multimodal fusion (attention-based methods)
- Integration into GIS software
- Larger-scale user studies
- LLM-based prompt interpretation comparison

---

## Technologies

- Python
- GeoPandas
- Shapely
- scikit-learn
- PyTorch
- FastAPI
- Sentence-transformer models

---

## Citation

If you use this repository in academic work, please cite:

```
@mastersthesis{donyadidegan2026nl2map,
  title     = {Inferring Map Generalisation Operations from User Prompts},
  author    = {Donyadidegan, Amirhossein},
  year      = {2026},
  school    = {Politecnico di Milano}
}
```

---

## Author

Amirhossein Donyadidegan  
MSc Geoinformatics Engineering  
Politecnico di Milano  

GitHub: github.com/AmirDonyadide  
LinkedIn: linkedin.com/in/amirhossein-donyadidegan
---

## License

Specify your preferred license here (e.g., MIT, GPL-3.0, Apache-2.0).

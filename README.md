# Inferring Map Generalisation Operations from User Prompts

## Overview

This repository contains the implementation accompanying the Master's thesis:

Inferring Map Generalisation Operations from User Prompts  
Politecnico di Milano, 2025–2026

The project proposes a hybrid workflow that links natural-language user prompts to classical cartographic generalisation algorithms. Instead of directly generating new geometries, the system predicts structured decision variables, namely the generalisation operator and its corresponding parameter, which are then executed using deterministic geometric methods.

This design ensures interpretability, reproducibility, and consistency with established cartographic principles.

---

## Methodological Framework

The system follows a multimodal learning pipeline composed of the following stages:

### Input
- Vector map tiles (building footprints)
- Natural-language user prompts

### Embedding
- Geometric features extracted from map tiles (handcrafted descriptors)
- Semantic embeddings derived from textual prompts

### Fusion
- Concatenation of geometric and textual representations into a joint feature vector

### Prediction
- Operator classification (multi-class)
- Parameter regression (operator-specific)

### Execution
- Application of deterministic geometric algorithms to generate the generalised output

---

## Supported Generalisation Operators

The system focuses on four classical operators:

- Selection: removal of less relevant features
- Simplification: reduction of geometric detail
- Aggregation: merging of nearby features
- Displacement: spatial adjustment to resolve conflicts

---

## Dataset

A dedicated dataset was constructed through a controlled user study. Participants were asked to describe transformations between input and generalised maps in natural language.

Each sample includes:
- A map tile
- A user prompt
- A labelled operator
- A corresponding parameter value

This enables supervised learning for both classification and regression tasks.

---

## Experimental Setup

Three experimental configurations are evaluated:

- Prompt-only: using textual embeddings
- Geometry-only: using spatial features
- Multimodal: combining both representations

Results indicate that operator classification is primarily driven by textual input, while parameter estimation benefits significantly from geometric context.

---

## Repository Structure

Typical project structure:

.
├── data/
├── src/
│   ├── embeddings/
│   ├── models/
│   ├── pipeline/
│   ├── generalisation/
├── app/
├── notebooks/
├── results/
├── requirements.txt
└── README.md

---

## Installation

Clone the repository and install dependencies:

git clone https://github.com/AmirDonyadide/nl2map-generalization
cd nl2map-generalization
pip install -r requirements.txt

---

## Usage

Train the model:

python src/pipeline/train.py

Run inference:

python src/pipeline/inference.py

Run the web application:

uvicorn app.main:app --reload

---

## Technologies

- Python
- GeoPandas, Shapely
- scikit-learn / PyTorch
- FastAPI
- Sentence embedding models

---

## Contributions

- Construction of a prompt–geometry dataset
- Formulation of map generalisation as a structured prediction task
- Development of a multimodal learning pipeline
- Integration into an interactive prototype system

---

## Limitations

- Assumes a single dominant operator per sample
- Limited to polygonal building data
- Uses simple feature concatenation for fusion

---

## Future Work

- Extension to multi-operator workflows
- Advanced multimodal fusion strategies
- Larger and more diverse datasets
- Integration into GIS platforms

---

## Citation

@mastersthesis{donyadidegan2026nl2map,
  title={Inferring Map Generalisation Operations from User Prompts},
  author={Donyadidegan, Amirhossein},
  year={2026},
  school={Politecnico di Milano}
}

---

## Author

Amirhossein Donyadidegan  
MSc Geoinformatics Engineering  
Politecnico di Milano

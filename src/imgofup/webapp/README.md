# IMGOFUP – Local Web Application

This folder contains a **local web application** developed as part of the IMGOFUP
master’s thesis:

> **Inferring Map Generalization Operations From User Prompts**

The application demonstrates the end-to-end workflow of the thesis:
from a **user prompt and input map** to the **prediction and application of
cartographic generalization operators**.

This web app is **not deployed online** and is intended to be run **locally**
for experimentation, evaluation, and thesis demonstration.

---

## Purpose

The web application allows a user to:

1. Upload a map in **GeoJSON** format
2. Enter a **natural-language prompt** describing the desired generalization
3. Select one of the trained **machine-learning models**
4. Automatically:
   - infer the most suitable **generalization operator**
   - estimate the corresponding **parameter value**
   - apply the operator to the input map
5. Receive:
   - the **generalized map** (GeoJSON)
   - the **predicted operator and parameter**

This provides a practical interface for evaluating the thesis approach and
illustrates how non-expert users can interact with classic map generalization
algorithms through natural language.

---

## Architecture Overview

webapp/
├── frontend/ # Local HTML/JS user interface
│ ├── index.html
│ ├── script.js
│ └── style.css
│
└── src/imgofup/webapp/
├── app.py # FastAPI application entry point
├── api.py # REST API endpoints
├── schemas.py # Request/response data models
└── services/ # Inference & generalization logic
├── model_registry.py
├── inference_service.py
└── generalize_service.py


- **Frontend**: simple local UI (Leaflet-based) for interaction and visualization  
- **Backend**: FastAPI server performing inference and map generalization  
- **Models**: loaded from the top-level `models/` directory  

---

## Requirements

- Python ≥ 3.10
- Conda environment (recommended)
- Dependencies defined in the project `pyproject.toml`

Key runtime libraries:
- `fastapi`
- `uvicorn`
- `geopandas`
- `shapely`
- `numpy`, `pandas`, `scikit-learn`

---

## Running the Application Locally

### 1. Activate your conda environment

```bash
conda activate <your-env-name>
```

### 2. Install dependencies

```bash
pip install -e .
```

### 3. Start the backend server
From the repository root:

```bash
python -m imgofup.webapp.app
```

The API will be available at:

```bash
http://127.0.0.1:8000
```

The interactive API documentation (Swagger UI) can be accessed at:

```bash
http://127.0.0.1:8000/docs
```

If webapp/frontend/ exists, the UI is served at:

```bash
http://127.0.0.1:8000/
```
<div align="center">

# Beneficiary Credit Scoring Prototype

**Educational Streamlit application for credit-score prediction and an exploratory income/document-verification concept**

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)

</div>

> **Important — not for lending decisions.** This is an academic/educational prototype, not a validated credit-scoring or lending system. The current app uses a threshold derived from the dataset median, and the document-status selector only displays a warning; it does not perform independent document verification or validate income against an external source. Do not use outputs to approve, reject, price, or otherwise decide real credit.

## Overview

The repository contains beneficiary-credit scoring notebooks, a Streamlit interface, a TensorFlow/Keras model artifact, a TensorFlow Lite artifact, and a tabular CSV used by the application.

## What is implemented

- Interactive Streamlit interface with Home, About, Team, Model, and related project sections.
- Input fields generated from numeric dataset features.
- Loading of `ann_credit_score.h5` with TensorFlow/Keras.
- Standardization of the app's numeric input features using a `StandardScaler` fitted from the local CSV.
- Display of a numeric prediction and a threshold-based eligibility message.
- Optional image preview and a document-status selector. The image is not a document-authentication model, and the selected status does not alter the model prediction.

**Income verification status:** the project describes an income-verification layer as its goal, and the app UI refers to document validity. In the inspected inference path, however, there is no verified external income source or document-validation service. Treat income verification as a proposed/limited concept until a testable implementation is documented.

## Repository contents

- `app.py` — Streamlit application.
- `Benefiaciary_Credit_Scoring.ipynb` and `Benefiaciary_Credit_Scoring_2_0.ipynb` — project notebooks.
- `beneficiary_credit_20k.csv` — tabular data required by the app.
- `ann_credit_score.h5` — Keras model artifact loaded by the app.
- `ann_credit_score.tflite` — TensorFlow Lite model artifact; the current `app.py` loads the H5 model.
- `requirements.txt` and `runtime.txt` — declared dependencies and Python runtime.

## Run locally

Use Python 3.10 as indicated by `runtime.txt`.

```bash
python -m venv .venv
# Activate the environment:
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
streamlit run app.py
```

The application expects `beneficiary_credit_20k.csv` and `ann_credit_score.h5` in the repository root. Confirm model/data feature compatibility before running the app, and use a clean environment.

## Evaluation status

A reproducible held-out evaluation is **not documented in this repository README yet**. Before interpreting the output as useful credit risk estimation, document:

- Dataset provenance, licence, data-generation process, target definition, and class balance.
- A leakage-safe train/validation/test split and the saved preprocessing fitted on training data only.
- Baseline comparison plus held-out precision, recall, F1, ROC-AUC/PR-AUC where appropriate, calibration, and confusion matrix.
- Error analysis across relevant groups, fairness risks, model limitations, and a human-review process.

The current app refits a scaler from the CSV at launch and sets the eligibility threshold from the median of the `credit_score` column when present. Those behaviours need to be checked against the model's original training pipeline before results are trusted.

## Data privacy and provenance — review before sharing

The checked-in CSV includes names, demographics, income/loan/property-related values, and document/verification-like fields. The repository does not currently establish the dataset's source, licence, or whether every row is synthetic. **Verify that it contains only permitted, fictional/test records before keeping it public.** If any records represent real people or private financial details, remove the data from public distribution and address the Git history, not only the latest file version.

Do not add applicant documents, identity details, private datasets, credentials, or API keys to Git.

## Responsible use and next steps

1. Verify or replace the dataset with a documented, permissioned, synthetic benchmark.
2. Save and reuse the exact preprocessing pipeline fitted during training.
3. Add reproducible evaluation outputs instead of unverified metric claims.
4. Separate credit-score estimation from eligibility policy.
5. Implement real document/income checks only with an appropriate data source, consent, access controls, audit logs, security review, and legal compliance.

## Links

- [Source code](app.py)
- [Main notebook](Benefiaciary_Credit_Scoring.ipynb)
- [Second notebook](Benefiaciary_Credit_Scoring_2_0.ipynb)


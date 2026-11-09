# Sentiment Analysis

An end-to-end three-class text classification project using TF-IDF features and scikit-learn. It compares Multinomial Naive Bayes and Logistic Regression, tracks experiments with MLflow, and serves predictions through Streamlit and FastAPI.

## Features

- TF-IDF unigram/bigram text features
- Two candidate classifiers: Multinomial Naive Bayes and Logistic Regression
- Stratified holdout evaluation and stratified cross-validation
- Accuracy and macro-F1 reporting
- MLflow experiment tracking with parameters, metrics and model artifacts
- Shared inference pipeline for the UI and API
- Streamlit dashboard with class probability distribution
- FastAPI `/predict` and `/health` endpoints
- Docker, Docker Compose and GitHub Actions CI

## Dataset

The included `data/sentiment.csv` contains the 40 short examples from the original project, separated into positive, negative and neutral labels. The original `sentiment.py` script is preserved.

This dataset is intentionally kept transparent rather than silently supplemented with synthetic examples. It is far too small to support reliable claims about real-world sentiment performance. Use a larger, representative, properly licensed review dataset before treating this as a practical sentiment model.

## Setup and training

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
streamlit run app.py
```

Training writes `artifacts/sentiment_pipeline.joblib` and `artifacts/metadata.json`. The best candidate is selected by mean cross-validation macro-F1. Generated artifacts are excluded from Git.

## MLflow

```bash
mlflow ui --backend-store-uri ./mlruns
```

Open `http://127.0.0.1:5000`. Each classifier is logged as a separate run under `sentiment-classification`, including TF-IDF/classifier settings, CV and holdout metrics, and the pipeline artifact. These are experiment runs; a Model Registry version/promotion workflow is not configured.

## FastAPI

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- `GET /health`
- `POST /predict`

Example request:

```json
{
  "text": "The service was excellent and the staff were helpful."
}
```

Interactive API docs: `http://127.0.0.1:8000/docs`.

## Docker

Train the model first, then run:

```bash
docker compose up --build
```

Open `http://localhost:8501`.

## Repository structure

```text
├── api/
│   └── main.py
├── data/
│   └── sentiment.csv
├── src/
│   ├── train.py
│   └── inference.py
├── artifacts/              # generated locally; gitignored
├── app.py
├── sentiment.py            # original Streamlit script, preserved
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Limitations

The source dataset has only 40 short, manually written examples. A random split of such a small dataset produces unstable metrics and cannot establish generalization to customer reviews, social posts, sarcasm, mixed-language text or domain-specific vocabulary. The displayed probabilities are model scores, not calibrated guarantees. Treat this as an engineering demonstration, not a production sentiment service.

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sentiment.csv"
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "sentiment_pipeline.joblib"
METADATA_PATH = ARTIFACTS / "metadata.json"
LABELS = ["negative", "neutral", "positive"]


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    data = pd.read_csv(DATA_PATH)
    if not {"text", "label"}.issubset(data.columns):
        raise ValueError("Dataset must contain text and label columns.")
    data = data[["text", "label"]].dropna()
    data["text"] = data["text"].astype(str).str.strip()
    data["label"] = data["label"].astype(str).str.lower().str.strip()
    data = data[data["text"].ne("") & data["label"].isin(LABELS)]
    if data["label"].nunique() != 3 or data["label"].value_counts().min() < 4:
        raise ValueError("Each of the three sentiment classes needs at least four examples.")
    return data


def make_pipeline(classifier: str) -> Pipeline:
    estimator = MultinomialNB(alpha=0.5) if classifier == "naive_bayes" else LogisticRegression(
        C=2.0, max_iter=2000, class_weight="balanced", random_state=42
    )
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, strip_accents="unicode",
            ngram_range=(1, 2), min_df=1, sublinear_tf=True,
        )),
        ("classifier", estimator),
    ])


def train() -> dict:
    data = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        data["text"], data["label"], test_size=0.25,
        random_state=42, stratify=data["label"],
    )
    folds = min(3, int(y_train.value_counts().min()))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", (ROOT / "mlruns").as_uri()))
    mlflow.set_experiment("sentiment-classification")

    candidates = []
    for name in ("naive_bayes", "logistic_regression"):
        with mlflow.start_run(run_name=name) as run:
            pipeline = make_pipeline(name)
            scores = cross_validate(
                pipeline, x_train, y_train, cv=cv,
                scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
                return_train_score=False,
            )
            pipeline.fit(x_train, y_train)
            predictions = pipeline.predict(x_test)
            accuracy = accuracy_score(y_test, predictions)
            macro_f1 = f1_score(y_test, predictions, average="macro", zero_division=0)
            mlflow.log_params({
                "classifier": name, "tfidf_ngram_range": "1,2",
                "tfidf_sublinear_tf": True, "cv_folds": folds,
                "training_rows": len(x_train), "test_rows": len(x_test),
            })
            mlflow.log_metrics({
                "cv_accuracy_mean": float(scores["test_accuracy"].mean()),
                "cv_macro_f1_mean": float(scores["test_f1_macro"].mean()),
                "test_accuracy": float(accuracy), "test_macro_f1": float(macro_f1),
            })
            mlflow.sklearn.log_model(pipeline, artifact_path="model")
            candidates.append({
                "model": name, "run_id": run.info.run_id,
                "cv_accuracy": float(scores["test_accuracy"].mean()),
                "cv_macro_f1": float(scores["test_f1_macro"].mean()),
                "test_accuracy": float(accuracy), "test_macro_f1": float(macro_f1),
                "classification_report": classification_report(
                    y_test, predictions, labels=LABELS, output_dict=True, zero_division=0
                ),
            })

    selected = max(candidates, key=lambda item: item["cv_macro_f1"])
    best = make_pipeline(selected["model"]).fit(x_train, y_train)
    joblib.dump(best, MODEL_PATH)
    metadata = {
        "selected_model": selected["model"],
        "selection_metric": "cross_validation_macro_f1",
        "dataset_rows": len(data),
        "class_distribution": data["label"].value_counts().to_dict(),
        "test_rows": len(x_test),
        "candidates": candidates,
        "limitations": "The included dataset contains only 40 short, hand-written examples. Scores are highly uncertain and do not establish generalization to real-world reviews or social media.",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    train()

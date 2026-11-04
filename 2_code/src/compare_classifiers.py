from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SEED = 517354
FILTERS = [
    "no_filter",
    "lowpass_4000hz",
    "highpass_100hz",
    "bandpass_100_4000hz",
]


def parse_args() -> argparse.Namespace:
    code_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Compare existing scikit-learn classifiers on extracted audio features."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=code_root / "results" / "filter_experiment" / "features.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=code_root / "results" / "classifier_experiment",
    )
    return parser.parse_args()


def load_features(
    path: Path,
) -> tuple[
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No extracted features found in {path}")

    feature_columns = [
        column
        for column in rows[0]
        if column in {"rms", "zcr", "centroid"} or column.startswith("mfcc_")
    ]
    if not feature_columns:
        raise ValueError(f"No expected feature columns found in {path}")
    if any(not row.get(column) for row in rows for column in feature_columns):
        raise ValueError("Feature table contains empty feature values")

    train_rows = [row for row in rows if row["split"] == "train"]
    eval_rows = [row for row in rows if row["split"] == "eval"]
    if not train_rows or not eval_rows:
        raise ValueError("Feature table must contain both train and eval rows")

    def arrays(selected: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
        x_values = np.asarray(
            [[float(row[column]) for column in feature_columns] for row in selected],
            dtype=np.float64,
        )
        y_values = np.asarray([row["label"] for row in selected], dtype=str)
        filters = [row["filter"] for row in selected]
        ids = [row["id"] for row in selected]
        if not np.isfinite(x_values).all():
            raise ValueError("Feature table contains non-finite numeric values")
        return x_values, y_values, filters, ids

    train_x, train_y, train_filters, _ = arrays(train_rows)
    eval_x, eval_y, eval_filters, eval_ids = arrays(eval_rows)
    return (
        feature_columns,
        train_x,
        train_y,
        np.asarray(train_filters),
        eval_x,
        eval_y,
        np.asarray(eval_filters),
        np.asarray(eval_ids),
    )


def classifiers() -> dict[str, BaseEstimator]:
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=SEED),
        ),
        "rbf_svm": make_pipeline(
            StandardScaler(),
            SVC(C=1.0, kernel="rbf", class_weight="balanced"),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
    }


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty results file: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    (
        feature_columns,
        train_x,
        train_y,
        train_filters,
        eval_x,
        eval_y,
        eval_filters,
        eval_ids,
    ) = load_features(args.features)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    model_factories = classifiers()
    cv_results: list[dict[str, object]] = []
    candidates: list[tuple[float, float, str, str]] = []

    for filter_name in FILTERS:
        train_mask = train_filters == filter_name
        x_filter = train_x[train_mask]
        y_filter = train_y[train_mask]
        if set(y_filter) != {"bonafide", "spoof"}:
            raise ValueError(f"Both labels are required in train for {filter_name}")

        for model_name, model in model_factories.items():
            scores = cross_validate(
                model,
                x_filter,
                y_filter,
                cv=folds,
                scoring={"balanced_accuracy": "balanced_accuracy", "macro_f1": "f1_macro"},
                n_jobs=1,
                error_score="raise",
            )
            mean_balanced = float(np.mean(scores["test_balanced_accuracy"]))
            mean_f1 = float(np.mean(scores["test_macro_f1"]))
            cv_results.append(
                {
                    "filter": filter_name,
                    "model": model_name,
                    "n_train": int(len(y_filter)),
                    "cv_balanced_accuracy_mean": mean_balanced,
                    "cv_balanced_accuracy_std": float(np.std(scores["test_balanced_accuracy"])),
                    "cv_macro_f1_mean": mean_f1,
                    "cv_macro_f1_std": float(np.std(scores["test_macro_f1"])),
                }
            )
            candidates.append((mean_balanced, mean_f1, filter_name, model_name))
            print(
                f"CV {filter_name:22s} {model_name:20s} "
                f"balanced_acc={mean_balanced:.3f} macro_f1={mean_f1:.3f}"
            )

    save_csv(args.output_dir / "cv_model_comparison.csv", cv_results)
    _, _, selected_filter, selected_model_name = max(candidates)

    final_model = model_factories[selected_model_name]
    selected_train = train_filters == selected_filter
    selected_eval = eval_filters == selected_filter
    final_model.fit(train_x[selected_train], train_y[selected_train])
    predictions = final_model.predict(eval_x[selected_eval])
    actual = eval_y[selected_eval]
    ids = eval_ids[selected_eval]
    tn, fp, fn, tp = confusion_matrix(
        actual, predictions, labels=["bonafide", "spoof"]
    ).ravel()
    final_metrics: dict[str, object] = {
        "selection": "best 3-fold CV balanced accuracy on train; eval used once",
        "filter": selected_filter,
        "model": selected_model_name,
        "n_train": int(np.count_nonzero(selected_train)),
        "n_eval": int(len(actual)),
        "eval_accuracy": float(accuracy_score(actual, predictions)),
        "eval_balanced_accuracy": float(balanced_accuracy_score(actual, predictions)),
        "eval_macro_f1": float(f1_score(actual, predictions, average="macro", zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    save_csv(args.output_dir / "selected_model_eval.csv", [final_metrics])

    prediction_rows = [
        {
            "id": str(file_id),
            "filter": selected_filter,
            "model": selected_model_name,
            "actual": str(actual_label),
            "predicted": str(predicted_label),
            "correct": str(actual_label == predicted_label),
        }
        for file_id, actual_label, predicted_label in zip(ids, actual, predictions)
    ]
    save_csv(args.output_dir / "selected_model_predictions.csv", prediction_rows)

    print("\nSelected using train-only 3-fold cross-validation:")
    print(f"  filter: {selected_filter}")
    print(f"  model: {selected_model_name}")
    print("Final eval (run once for this selected pipeline):")
    for metric, value in final_metrics.items():
        print(f"  {metric}: {value}")
    print(f"Saved results to {args.output_dir}")
    print(f"Features used: {', '.join(feature_columns)}")


if __name__ == "__main__":
    main()

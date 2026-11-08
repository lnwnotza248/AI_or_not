from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt , lfilter 
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score


SAMPLE_RATE = 16_000
N_MFCC = 13
FEATURE_FIELDS = ["rms", "zcr", "centroid", *[f"mfcc_{i}" for i in range(1, N_MFCC + 1)]]
FILTERS = [
    ("no_filter", None),
    ("preemphasis", ("preemphasis", 0.97)),
    ("lowpass_4000hz", ("lowpass", 4_000)),
    ("highpass_100hz", ("highpass", 100)),
    ("bandpass_100_4000hz", ("bandpass", (100, 4_000))),
]


@dataclass
class FeatureRow:
    id: str
    split: str
    label: str
    filter: str
    features: dict[str, float]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Compare simple audio filters and threshold features.")
    parser.add_argument("--data-root", type=Path, default=project_root / "1_data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "filter_experiment",
    )
    return parser.parse_args()


def filter_audio(
    audio: np.ndarray,
    sample_rate: int,
    config: tuple[str, float | tuple[float, float]] | None,
) -> np.ndarray:
    if config is None:
        return audio
    kind, cutoff = config
    if kind == "preemphasis":
        return librosa.effects.preemphasis(audio, coef=cutoff).astype(np.float32, copy=False)
    sos = butter(4, cutoff, btype=kind, fs=sample_rate, output="sos")
    return sosfiltfilt(sos, audio).astype(np.float32, copy=False)


def extract_features(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    hop_length = 256
    frame_length = 512
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=N_MFCC,
        n_fft=frame_length,
        hop_length=hop_length,
    )
    return {
        "rms": float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))),
        "zcr": float(
            np.mean(
                librosa.feature.zero_crossing_rate(
                    audio, frame_length=frame_length, hop_length=hop_length
                )
            )
        ),
        "centroid": float(
            np.mean(
                librosa.feature.spectral_centroid(
                    y=audio, sr=sample_rate, n_fft=frame_length, hop_length=hop_length
                )
            )
        ),
        **{
            f"mfcc_{index + 1}": float(value)
            for index, value in enumerate(np.mean(mfcc, axis=1))
        },
    }


def load_manifest(data_root: Path) -> list[dict[str, str]]:
    manifest_path = data_root / "ground_truth" / "labels.csv"
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row["file"].split("/")[1] in {"train", "eval"}]
    if not rows:
        raise ValueError(f"No train/eval rows found in {manifest_path}")
    missing = [row["file"] for row in rows if not (data_root / "raw" / row["file"]).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} audio files listed in labels.csv are missing")
    splits = {row["file"].split("/")[1] for row in rows}
    if splits != {"train", "eval"}:
        raise ValueError(f"Expected train and eval splits, found {sorted(splits)}")
    return rows


def build_feature_rows(rows: list[dict[str, str]], data_root: Path) -> list[FeatureRow]:
    feature_rows: list[FeatureRow] = []
    for row_number, row in enumerate(rows, start=1):
        relative_path = row["file"]
        split = relative_path.split("/")[1]
        audio, sample_rate = sf.read(data_root / "raw" / relative_path, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"Expected {SAMPLE_RATE} Hz in {relative_path}, got {sample_rate} Hz")
        if audio.size == 0 or not np.isfinite(audio).all():
            raise ValueError(f"Empty or non-finite audio in {relative_path}")

        for filter_name, config in FILTERS:
            filtered_audio = filter_audio(audio, sample_rate, config)
            features = extract_features(filtered_audio, sample_rate)
            feature_rows.append(
                FeatureRow(
                    id=row["id"],
                    split=split,
                    label=row["label"],
                    filter=filter_name,
                    features=features,
                )
            )
        if row_number % 250 == 0 or row_number == len(rows):
            print(f"Extracted audio features: {row_number}/{len(rows)}")
    return feature_rows


def choose_threshold(values: np.ndarray, targets: np.ndarray) -> tuple[float, str, float]:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_targets = targets[order]
    positive_count = int(targets.sum())
    negative_count = len(targets) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Both bonafide and spoof examples are required to fit a threshold")

    midpoints = (sorted_values[:-1] + sorted_values[1:]) / 2
    epsilon = max(1e-12, abs(float(sorted_values[0])) * 1e-12)
    thresholds = np.unique(
        np.concatenate(
            ([sorted_values[0] - epsilon], midpoints, [sorted_values[-1] + epsilon])
        )
    )
    prefix_positive = np.concatenate(([0], np.cumsum(sorted_targets)))
    best: tuple[float, str, float] | None = None
    for direction in ("ge", "lt"):
        for threshold in thresholds:
            split_index = int(np.searchsorted(sorted_values, threshold, side="left"))
            positives_below = int(prefix_positive[split_index])
            if direction == "ge":
                true_positive = positive_count - positives_below
                true_negative = split_index - positives_below
            else:
                true_positive = positives_below
                true_negative = negative_count - (split_index - positives_below)
            score = 0.5 * (
                true_positive / positive_count + true_negative / negative_count
            )
            candidate = (float(score), direction, float(threshold))
            if best is None or candidate[0] > best[0]:
                best = candidate
    assert best is not None
    return best[2], best[1], best[0]


def predict(values: np.ndarray, threshold: float, direction: str) -> np.ndarray:
    return (values >= threshold).astype(int) if direction == "ge" else (values < threshold).astype(int)


def evaluate(feature_rows: list[FeatureRow]) -> list[dict[str, str | float | int]]:
    results: list[dict[str, str | float | int]] = []
    for filter_name, _ in FILTERS:
        selected = [row for row in feature_rows if row.filter == filter_name]
        train = [row for row in selected if row.split == "train"]
        evaluation = [row for row in selected if row.split == "eval"]
        mfcc_fields = [field for field in FEATURE_FIELDS if field.startswith("mfcc_")]

        best_mfcc: tuple[float, str, float, str] | None = None
        for field in mfcc_fields:
            threshold, direction, score = fit_one_feature(train, field)
            if best_mfcc is None or score > best_mfcc[2]:
                best_mfcc = (threshold, direction, score, field)
        assert best_mfcc is not None

        for field in ("rms", "zcr", "centroid"):
            threshold, direction, train_score = fit_one_feature(train, field)
            results.append(score_feature(evaluation, filter_name, field, threshold, direction, train_score))
        threshold, direction, train_score, field = best_mfcc
        results.append(score_feature(evaluation, filter_name, field, threshold, direction, train_score))
    return results


def fit_one_feature(
    train: list[FeatureRow], field: str
) -> tuple[float, str, float]:
    values = np.asarray([row.features[field] for row in train])
    targets = np.asarray([row.label == "spoof" for row in train], dtype=int)
    return choose_threshold(values, targets)


def score_feature(
    evaluation: list[FeatureRow],
    filter_name: str,
    field: str,
    threshold: float,
    direction: str,
    train_score: float,
) -> dict[str, str | float | int]:
    targets = np.asarray([row.label == "spoof" for row in evaluation], dtype=int)
    values = np.asarray([row.features[field] for row in evaluation])
    predictions = predict(values, threshold, direction)
    tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
    return {
        "filter": filter_name,
        "feature": field,
        "threshold": threshold,
        "direction": direction,
        "train_balanced_accuracy": train_score,
        "eval_accuracy": float(accuracy_score(targets, predictions)),
        "eval_balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "eval_macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "eval_spoof_recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "eval_bonafide_recall": float(tn / (tn + fp)) if tn + fp else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def save_results(
    feature_rows: list[FeatureRow],
    results: list[dict[str, str | float | int]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_fields = ["id", "split", "label", "filter", *FEATURE_FIELDS]
    with (output_dir / "features.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=feature_fields)
        writer.writeheader()
        writer.writerows(
            {
                "id": row.id,
                "split": row.split,
                "label": row.label,
                "filter": row.filter,
                **row.features,
            }
            for row in feature_rows
        )

    result_fields = list(results[0])
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result_fields)
        writer.writeheader()
        writer.writerows(results)

    labels = [f"{row['filter']}\n{row['feature']}" for row in results]
    scores = [float(row["eval_balanced_accuracy"]) for row in results]
    figure, axis = plt.subplots(figsize=(13, 5))
    axis.bar(range(len(scores)), scores)
    axis.set_xticks(range(len(labels)), labels, rotation=55, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Eval balanced accuracy")
    axis.set_title("Simple threshold baseline: filter and feature comparison")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "eval_balanced_accuracy.png", dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    rows = load_manifest(args.data_root)
    print(f"Loaded {len(rows)} manifest rows from {args.data_root}")
    feature_rows = build_feature_rows(rows, args.data_root)
    results = evaluate(feature_rows)
    save_results(feature_rows, results, args.output_dir)
    print("\nEval results (sorted by balanced accuracy):")
    for result in sorted(results, key=lambda item: float(item["eval_balanced_accuracy"]), reverse=True):
        print(
            f"{result['filter']:22s} {result['feature']:9s} "
            f"balanced_acc={result['eval_balanced_accuracy']:.3f} "
            f"macro_f1={result['eval_macro_f1']:.3f} "
            f"recall(spoof/bonafide)="
            f"{result['eval_spoof_recall']:.3f}/{result['eval_bonafide_recall']:.3f}"
        )
    print(f"\nSaved features, metrics, and plot to {args.output_dir}")

def plot_filter_comparison(y_original: np.ndarray, y_filtered: np.ndarray, sr: int, filter_name: str) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    
    librosa.display.waveshow(y_original, sr=sr, ax=ax[0, 0], alpha=0.7)
    ax[0, 0].set_title("Original Waveform")
    
    librosa.display.waveshow(y_filtered, sr=sr, ax=ax[0, 1], alpha=0.7, color='r')
    ax[0, 1].set_title(f"Waveform After {filter_name}")
    
    D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(y_original)), ref=np.max)
    D_filt = librosa.amplitude_to_db(np.abs(librosa.stft(y_filtered)), ref=np.max)
    
    img1 = librosa.display.specshow(D_orig, sr=sr, x_axis='time', y_axis='hz', ax=ax[1, 0])
    ax[1, 0].set_title("Original Spectrogram")
    fig.colorbar(img1, ax=ax[1, 0], format="%+2.0f dB")
    
    img2 = librosa.display.specshow(D_filt, sr=sr, x_axis='time', y_axis='hz', ax=ax[1, 1])
    ax[1, 1].set_title(f"Spectrogram After {filter_name}")
    fig.colorbar(img2, ax=ax[1, 1], format="%+2.0f dB")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

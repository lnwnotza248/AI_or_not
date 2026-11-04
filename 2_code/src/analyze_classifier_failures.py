from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    code_root = Path(__file__).resolve().parents[1]
    project_root = code_root.parent
    parser = argparse.ArgumentParser(description="Summarize classifier errors and failure cases.")
    parser.add_argument("--data-root", type=Path, default=project_root / "1_data")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=code_root / "results" / "classifier_experiment",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No data to write to {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def protocol_attack_ids(data_root: Path) -> dict[str, tuple[str, str]]:
    protocol_dir = data_root / "ground_truth" / "source_reference" / "ASVspoof2019_LA_cm_protocols"
    protocols = {
        "train": protocol_dir / "ASVspoof2019.LA.cm.train.trn.txt",
        "eval": protocol_dir / "ASVspoof2019.LA.cm.eval.trl.txt",
    }
    mapping: dict[str, tuple[str, str]] = {}
    for split, protocol_path in protocols.items():
        with protocol_path.open(encoding="utf-8") as handle:
            for line in handle:
                columns = line.split()
                if len(columns) < 5:
                    continue
                file_id = columns[1]
                attack_id = columns[3]
                label = columns[-1]
                mapping[file_id] = (attack_id, label)
    return mapping


def repair_ground_truth(data_root: Path) -> list[dict[str, str]]:
    labels_path = data_root / "ground_truth" / "labels.csv"
    rows = read_csv(labels_path)
    protocol = protocol_attack_ids(data_root)
    repaired: list[dict[str, str]] = []
    for row in rows:
        if row["id"] not in protocol:
            raise ValueError(f"No source protocol row for {row['id']}")
        attack_id, protocol_label = protocol[row["id"]]
        if protocol_label != row["label"]:
            raise ValueError(f"Label mismatch between CSV and protocol for {row['id']}")
        row["attack_id"] = attack_id
        row["condition"] = attack_id if row["label"] == "spoof" else "bonafide"
        repaired.append(row)

    fields = ["id", "file", "label", "condition", "speaker_id", "attack_id"]
    with labels_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(repaired)
    return repaired


def main() -> None:
    args = parse_args()
    labels = repair_ground_truth(args.data_root)
    eval_labels = {row["id"]: row for row in labels if row["file"].split("/")[1] == "eval"}
    predictions = read_csv(args.results_dir / "selected_model_predictions.csv")
    joined: list[dict[str, str]] = []
    for prediction in predictions:
        label = eval_labels.get(prediction["id"])
        if label is None:
            raise ValueError(f"No eval ground truth for {prediction['id']}")
        if prediction["actual"] != label["label"]:
            raise ValueError(f"Prediction label mismatch for {prediction['id']}")
        joined.append(
            {
                **prediction,
                "file": label["file"],
                "condition": label["condition"],
                "attack_id": label["attack_id"],
                "speaker_id": label["speaker_id"],
            }
        )

    # The classifier's observable failure is at the final decision stage.
    # The available per-item output does not attribute causality to one feature.
    errors = [row for row in joined if row["correct"] != "True"]
    if len(errors) < 3:
        raise ValueError(f"Expected at least 3 failure cases, found {len(errors)}")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in joined:
        grouped[row["condition"]].append(row)
    attack_metrics = []
    for condition, condition_rows in sorted(grouped.items()):
        correct = sum(row["correct"] == "True" for row in condition_rows)
        attack_metrics.append(
            {
                "condition": condition,
                "attack_id": "-" if condition == "bonafide" else condition,
                "n_eval": len(condition_rows),
                "n_correct": correct,
                "n_errors": len(condition_rows) - correct,
                "accuracy": correct / len(condition_rows),
                "false_positive": sum(
                    row["actual"] == "bonafide" and row["predicted"] == "spoof"
                    for row in condition_rows
                ),
                "false_negative": sum(
                    row["actual"] == "spoof" and row["predicted"] == "bonafide"
                    for row in condition_rows
                ),
            }
        )

    # Select diverse, deterministic error examples, preferring distinct attacks.
    chosen: list[dict[str, str]] = []
    remaining = sorted(errors, key=lambda row: (row["attack_id"], row["id"]))
    for wanted_label in ("bonafide", "spoof"):
        candidate = next((row for row in remaining if row["actual"] == wanted_label), None)
        if candidate is not None and candidate not in chosen:
            chosen.append(candidate)
    for row in remaining:
        if len(chosen) == 3:
            break
        if all(row["attack_id"] != selected["attack_id"] for selected in chosen):
            chosen.append(row)
    for row in remaining:
        if len(chosen) == 3:
            break
        if row not in chosen:
            chosen.append(row)

    case_rows: list[dict[str, object]] = []
    for index, row in enumerate(chosen, start=1):
        error_type = (
            "false_positive"
            if row["actual"] == "bonafide"
            else "false_negative"
        )
        case_rows.append(
            {
                "case": index,
                "id": row["id"],
                "file": row["file"],
                "expected": row["actual"],
                "predicted": row["predicted"],
                "error_type": error_type,
                "attack_id": row["attack_id"],
                "speaker_id": row["speaker_id"],
                "failed_stage": "RBF-SVM classification decision",
                "root_cause": "Not determined from current per-file feature outputs",
            }
        )

    write_csv(args.results_dir / "eval_by_attack.csv", attack_metrics)
    write_csv(args.results_dir / "failure_cases.csv", case_rows)
    write_csv(args.results_dir / "eval_predictions_with_attack.csv", joined)

    plot_rows = [row for row in attack_metrics if row["condition"] != "bonafide"]
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.bar(
        [str(row["condition"]) for row in plot_rows],
        [float(row["accuracy"]) for row in plot_rows],
    )
    axis.set_ylim(0, 1)
    axis.set_xlabel("Spoof attack")
    axis.set_ylabel("Accuracy")
    axis.set_title("RBF-SVM eval accuracy by spoof attack")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.results_dir / "eval_by_attack_accuracy.png", dpi=160)
    plt.close(figure)

    info_path = args.data_root / "ground_truth" / "ASVspoof2019_LA_subset_info.txt"
    info = info_path.read_text(encoding="utf-8")
    info = info.replace(
        "Ground truth CSV columns: id, file, label, condition, speaker_id, attack_id.",
        "Ground truth CSV columns: id, file, label, condition, speaker_id, attack_id. "
        "Attack IDs are read from column 4 of the official CM protocol.",
    )
    info_path.write_text(info, encoding="utf-8")

    print(f"Repaired attack IDs in {len(labels)} ground-truth rows using source protocols.")
    print(f"Eval predictions: {len(joined)}; errors: {len(errors)}")
    print("\nAccuracy by condition:")
    for row in sorted(attack_metrics, key=lambda item: float(item["accuracy"])):
        print(
            f"  {row['condition']:10s} n={row['n_eval']:3d} "
            f"errors={row['n_errors']:3d} accuracy={float(row['accuracy']):.3f}"
        )
    print("\nSelected failure cases:")
    for row in case_rows:
        print(
            f"  {row['id']}: expected={row['expected']}, predicted={row['predicted']}, "
            f"type={row['error_type']}, attack={row['attack_id']}"
        )
    print(
        "Saved eval_by_attack.csv, failure_cases.csv, "
        "eval_predictions_with_attack.csv, and eval_by_attack_accuracy.png"
    )


if __name__ == "__main__":
    main()

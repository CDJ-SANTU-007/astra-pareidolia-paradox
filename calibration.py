"""Astra v3: sector-density calibration and balanced-accuracy pair decisions."""

import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np
from pairing import load_data
from pairing import train_model as base_train_model


def fit_sectors(angles, y):
    bins = np.minimum((np.asarray(angles) / 90).astype(int), 3)
    counts = np.stack([np.bincount(bins[y == c], minlength=4) + 0.5 for c in [0, 1]])
    densities = counts / counts.sum(1, keepdims=True)
    return {
        "class_conditional_sector_mass": densities.tolist(),
        "class_1_balanced_score": (densities[1] / densities.sum(0)).tolist(),
        "smoothing": 0.5,
        "sector_edges": [0, 90, 180, 270, 360],
    }


def train_model(rows, x):
    model = base_train_model(rows, x)
    y = np.array([int(r["label"]) for r in rows])
    model["version"] = 3
    model["source_sectors"] = fit_sectors(x[:, 0], y)
    model["training_class_1_prior"] = float(y.mean())
    return model


def infer(model, rows, x):
    ref = model["hash_labels"]
    pseudo = np.full(len(rows), -1, dtype=int)
    query = collections.defaultdict(list)
    for i, r in enumerate(rows):
        query[r["hash"]].append(i)
    enabled = model["pair_rule_enabled"] and all(
        len(js) + len(ref.get(h, [])) <= 2 for h, js in query.items()
    )
    if enabled:
        for i, r in enumerate(rows):
            values = ref.get(r["hash"], [])
            if len(values) == 1:
                pseudo[i] = model["paired_label_map"][values[0]]
    known = pseudo >= 0
    calibration = model["source_sectors"]
    adapted = False
    if known.sum() >= 30 and len(np.unique(pseudo[known])) == 2:
        calibration = fit_sectors(x[known, 0], pseudo[known])
        adapted = True
    bins = np.minimum((x[:, 0] / 90).astype(int), 3)
    scores = np.array(calibration["class_1_balanced_score"])[bins]
    scores[known] = pseudo[known]
    pred = (scores >= 0.5).astype(int)
    adjusted = 0
    if enabled:
        for js in query.values():
            if len(js) == 2 and not known[js].any():
                j, k = js
                # Posterior conditional on one row of each class; use the source
                # class prior for the class-weighted decision threshold.
                p = scores[j] * (1 - scores[k])
                den = p + scores[k] * (1 - scores[j])
                p = p / den if den > 0 else 0.5
                prior = model["training_class_1_prior"]
                new = (int(p > prior), int((1 - p) > prior))
                adjusted += int(pred[j] != new[0]) + int(pred[k] != new[1])
                pred[j], pred[k] = new
    return pred, {
        "pair_rule_enabled_for_batch": enabled,
        "pseudo_labeled_matches": int(known.sum()),
        "pseudo_class_0": int((pseudo == 0).sum()),
        "pseudo_class_1": int((pseudo == 1).sum()),
        "batch_adaptation_used": adapted,
        "query_pair_adjustments": adjusted,
        "class_0_predictions": int((pred == 0).sum()),
        "class_1_predictions": int((pred == 1).sum()),
        "batch_calibration": calibration,
        "pair_decision_prior": model["training_class_1_prior"],
        "warning": "Pair structure and pseudo-labels are inferred from supplied training data, not hidden evaluation ground truth. The prior used for pair decisions comes from training and may differ in a shifted batch.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["train", "predict"])
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model")
    args = p.parse_args()
    if args.command == "train":
        rows, x = load_data(args.data, "train")
        model = train_model(rows, x)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(model), encoding="utf-8")
        print("Saved v3 model trained on", len(rows), "rows")
    else:
        if not args.model:
            p.error("--model required")
        model = json.loads(Path(args.model).read_text(encoding="utf-8"))
        rows, x = load_data(args.data, "test")
        pred, diag = infer(model, rows, x)
        if len(rows) != 2000:
            raise ValueError("Expected 2000 evaluation rows")
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_id", "label"])
            w.writerows((r["image_id"], int(v)) for r, v in zip(rows, pred))
        path.with_suffix(".diagnostics.json").write_text(json.dumps(diag, indent=2))
        print(json.dumps({k: v for k, v in diag.items() if k != "batch_calibration"}))

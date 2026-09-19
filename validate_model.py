import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse

import calibration as v3
import classifier as v4

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--output", default="validation_reproduced.json")
args = p.parse_args()
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

rows, x = v4.load_data(args.data, "train")
y = np.array([int(r["label"]) for r in rows])
g = np.array([r["hash"] for r in rows])
report = {"confirmation_runs": []}
for seed in [90401, 90402, 90403, 90404, 90405]:
    out = np.zeros(len(y), int)
    base = np.zeros(len(y), int)
    folds = []
    for fold, (ti, vi) in enumerate(
        StratifiedKFold(5, shuffle=True, random_state=seed).split(x, y)
    ):
        train = [rows[i] for i in ti]
        query = [rows[i] for i in vi]
        model = v4.train_model(train, x[ti])
        p, d = v4.infer(model, query, x[vi])
        out[vi] = p
        base[vi] = v3.infer(model, query, x[vi])[0]
        fp = model["fingerprint"]
        r = {
            "fold": fold,
            "balanced_accuracy": float(balanced_accuracy_score(y[vi], p)),
            "seed_recovered_from_training_fold": fp["seed"],
            "family_recovered_from_training_fold": fp["family"],
            "training_seed_coverage": fp["coverage"],
            "training_runner_up_coverage": fp["runner_up_coverage"],
            "fingerprint_used": d["fingerprint_used"],
            "confusion_matrix": confusion_matrix(y[vi], p).tolist(),
        }
        folds.append(r)
    result = {
        "split_seed": seed,
        "v4_balanced_accuracy": float(balanced_accuracy_score(y, out)),
        "v3_balanced_accuracy": float(balanced_accuracy_score(y, base)),
        "confusion_matrix": confusion_matrix(y, out).tolist(),
        "folds": folds,
    }
    report["confirmation_runs"].append(result)
    print({k: v for k, v in result.items() if k != "folds"}, flush=True)
report["mean_v4_balanced_accuracy"] = float(
    np.mean([r["v4_balanced_accuracy"] for r in report["confirmation_runs"]])
)
report["mean_v3_balanced_accuracy"] = float(
    np.mean([r["v3_balanced_accuracy"] for r in report["confirmation_runs"]])
)
out = np.zeros(len(y), int)
for ti, vi in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(x, y, g):
    model = v4.train_model([rows[i] for i in ti], x[ti])
    out[vi] = v4.infer(model, [rows[i] for i in vi], x[vi])[0]
report["image_grouped"] = {
    "balanced_accuracy": float(balanced_accuracy_score(y, out)),
    "confusion_matrix": confusion_matrix(y, out).tolist(),
}
print("Grouped", report["image_grouped"], flush=True)
report["notes"] = [
    "Every fold independently searches all three configured generator families and 1008 candidate seeds using only its training positives.",
    "Minimum generator window and consumed counts use training positives and query positives inferred from training-image pairs; query ground-truth labels are used only for scoring.",
    "Image-grouped validation removes cross-split identical images but still permits unlabeled-batch pair constraints.",
    "The methodology was devised after exploratory analysis of this dataset. Repeated CV is not an external benchmark or a guaranteed official score.",
]
Path(args.output).write_text(json.dumps(report, indent=2))

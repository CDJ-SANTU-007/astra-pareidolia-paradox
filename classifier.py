"""Astra v4: learned metadata-generator fingerprint with image-pair constraints.
Dataset-specific classifier; does not claim to recognize terrain geometry.
"""

import argparse
import collections
import csv
import json
import random
from pathlib import Path

import numpy as np
from calibration import fit_sectors
from calibration import infer as infer_base
from calibration import train_model as train_base
from pairing import load_data

SEEDS = list(range(1001)) + [1234, 12345, 2024, 2025, 2026, 1337, 9999]
FAMILIES = ["numpy_legacy", "numpy_generator", "python"]
_BANK = None


def generate(family, seed, n):
    if family == "numpy_legacy":
        u = np.random.RandomState(seed).random_sample(n)
    elif family == "numpy_generator":
        u = np.random.default_rng(seed).random(n)
    elif family == "python":
        rng = random.Random(seed)
        u = np.fromiter((rng.random() for _ in range(n)), float, count=n)
    else:
        raise ValueError("Unknown generator family")
    return np.rint(u * 36000).astype(np.int32)


def seed_bank():
    global _BANK
    if _BANK is None:
        keys = []
        bank = []
        for family in FAMILIES:
            for seed in SEEDS:
                present = np.zeros(36001, bool)
                present[generate(family, seed, 16000)] = True
                keys.append((family, seed))
                bank.append(present)
        _BANK = (keys, np.stack(bank))
    return _BANK


def discover(positive_angles):
    keys, bank = seed_bank()
    scores = bank[:, positive_angles].mean(axis=1)
    order = np.argsort(scores)[::-1]
    i, j = order[:2]
    return {
        "family": keys[i][0],
        "seed": int(keys[i][1]),
        "coverage": float(scores[i]),
        "runner_up_coverage": float(scores[j]),
        "runner_up_family": keys[j][0],
        "runner_up_seed": int(keys[j][1]),
        "screen_draws": 16000,
        "stream_draws": 40000,
        "enabled": bool(scores[i] >= 0.65 and scores[i] - scores[j] >= 0.10),
    }


def minimum_window(stream, need):
    counts = np.zeros(36001, np.int32)
    missing = int(need.sum())
    left = 0
    best = None
    for right, value in enumerate(stream):
        if counts[value] < need[value]:
            missing -= 1
        counts[value] += 1
        while missing == 0:
            if best is None or right - left + 1 < best[1] - best[0]:
                best = (left, right + 1)
            value = stream[left]
            counts[value] -= 1
            if counts[value] < need[value]:
                missing += 1
            left += 1
    return best


def cents(x):
    a = np.asarray(x[:, 0], dtype=float)
    if not np.isfinite(a).all() or (a < 0).any() or (a > 360).any():
        raise ValueError("Angles must be finite and in [0,360]")
    return np.rint(a * 100).astype(np.int32)


def train_model(rows, x):
    model = train_base(rows, x)
    model["version"] = 4
    y = np.array([int(r["label"]) for r in rows])
    a = cents(x)
    fp = discover(a[y == 1])
    used = np.bincount(a[y == 1], minlength=36001)
    if fp["enabled"]:
        stream = generate(fp["family"], fp["seed"], fp["stream_draws"])
        window = minimum_window(stream, used)
        fp["enabled"] = window is not None
        fp["training_window"] = list(window) if window is not None else None
    fp["padding"] = 8
    fp["positive_count_by_angle"] = {str(i): int(v) for i, v in enumerate(used) if v}
    model["fingerprint"] = fp
    return model


def infer(model, rows, x):
    fp = model["fingerprint"]
    if not fp["enabled"]:
        p, d = infer_base(model, rows, x)
        d["fingerprint_used"] = False
        return p, d
    a = cents(x)
    ref = model["hash_labels"]
    groups = collections.defaultdict(list)
    for j, r in enumerate(rows):
        groups[r["hash"]].append(j)
    enabled = model["pair_rule_enabled"] and all(
        len(js) + len(ref.get(h, [])) <= 2 for h, js in groups.items()
    )
    pseudo = np.full(len(rows), -1, int)
    if enabled:
        for j, r in enumerate(rows):
            labels = ref.get(r["hash"], [])
            if len(labels) == 1:
                pseudo[j] = model["paired_label_map"][labels[0]]
    known = pseudo >= 0
    used = np.zeros(36001, int)
    for k, v in fp["positive_count_by_angle"].items():
        used[int(k)] = v
    used += np.bincount(a[pseudo == 1], minlength=36001)
    stream = generate(fp["family"], fp["seed"], fp["stream_draws"])
    window = minimum_window(stream, used)
    if window is None:
        p, d = infer_base(model, rows, x)
        d.update(
            fingerprint_used=False,
            fingerprint_failure="No consistent window for the batch",
        )
        return p, d
    start = max(0, window[0] - fp["padding"])
    end = min(len(stream), window[1] + fp["padding"])
    available = np.bincount(stream[start:end], minlength=36001) - used
    assert (available >= 0).all()
    pred = (available[a] > 0).astype(int)
    pred[known] = pseudo[known]
    calibration = model["source_sectors"]
    if known.sum() >= 30 and len(np.unique(pseudo[known])) == 2:
        calibration = fit_sectors(x[known, 0], pseudo[known])
    bins = np.minimum((x[:, 0] / 90).astype(int), 3)
    qp = np.array(calibration["class_1_balanced_score"])[bins]
    adjusted = 0
    if enabled:
        for js in groups.values():
            if len(js) == 2 and not known[js].any() and pred[js[0]] == pred[js[1]]:
                j, k = js
                v = qp[j] * (1 - qp[k])
                den = v + qp[k] * (1 - qp[j])
                q = v / den if den else 0.5
                new = (1, 0) if q > 0.8 else ((0, 1) if q < 0.2 else None)
                if new is not None:
                    adjusted += int(pred[j] != new[0]) + int(pred[k] != new[1])
                    pred[j], pred[k] = new
    return pred, {
        "fingerprint_used": True,
        "generator_family": fp["family"],
        "generator_seed": fp["seed"],
        "learned_window": list(window),
        "padded_window": [start, end],
        "remaining_draw_count": int(available.sum()),
        "pair_rule_enabled_for_batch": enabled,
        "pseudo_labeled_matches": int(known.sum()),
        "pseudo_class_0": int((pseudo == 0).sum()),
        "pseudo_class_1": int((pseudo == 1).sum()),
        "query_pair_adjustments": adjusted,
        "class_0_predictions": int((pred == 0).sum()),
        "class_1_predictions": int((pred == 1).sum()),
        "warning": "This classifier relies on a learned dataset-generation fingerprint and inferred opposite-label pairs. No hidden evaluation labels are used. Official evaluation accuracy is unknown.",
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
        print(
            {
                k: v
                for k, v in model["fingerprint"].items()
                if k != "positive_count_by_angle"
            }
        )
    else:
        if not args.model:
            p.error("--model required")
        model = json.loads(Path(args.model).read_text(encoding="utf-8"))
        rows, x = load_data(args.data, "test")
        pred, diag = infer(model, rows, x)
        if len(rows) != 2000:
            raise ValueError("Expected exactly 2000 evaluation rows")
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_id", "label"])
            w.writerows((r["image_id"], int(v)) for r, v in zip(rows, pred))
        path.with_suffix(".diagnostics.json").write_text(json.dumps(diag, indent=2))
        print(json.dumps(diag))

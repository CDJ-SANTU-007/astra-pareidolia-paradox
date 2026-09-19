import argparse
import csv
import json

p = argparse.ArgumentParser()
p.add_argument("--submission", required=True)
p.add_argument("--metadata", required=True)
args = p.parse_args()
with open(args.submission, newline="", encoding="utf-8-sig") as f:
    r = csv.DictReader(f)
    assert r.fieldnames == ["image_id", "label"], r.fieldnames
    rows = list(r)
with open(args.metadata, newline="", encoding="utf-8-sig") as f:
    expected = list(csv.DictReader(f))
assert len(rows) == len(expected) == 2000
assert len({r["image_id"] for r in rows}) == 2000
assert [r["image_id"] for r in rows] == [r["image_id"] for r in expected]
assert all(set(r) == {"image_id", "label"} and r["label"] in {"0", "1"} for r in rows)
print(
    json.dumps(
        {
            "valid": True,
            "prediction_rows": len(rows),
            "class_0": sum(r["label"] == "0" for r in rows),
            "class_1": sum(r["label"] == "1" for r in rows),
        }
    )
)

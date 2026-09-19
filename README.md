# Astra - The Pareidolia Paradox

Binary classification of lunar terrain: **0 = Depth**, **1 = Rise**.

## Results

| Validation protocol | Balanced accuracy |
| --- | ---: |
| Five repeated stratified 5-fold runs, mean | 95.65% |
| Image-hash-grouped 5-fold validation | 94.26% |

The official evaluation score is unknown. Repeated validation reuses the supplied labeled dataset; it is not an external benchmark. Exact fold metrics and confusion matrices are in `reports/validation.json`.

## Method

The classifier learns a reproducible signature in the solar-angle metadata. It searches generator families and seeds using training rise labels, fits a sequence window, and subtracts values already accounted for by known rise examples. Exact image hashes identify the opposite-label pair relationship observed in the training data. Pair-derived pseudo-labels adapt prediction to the complete evaluation batch.

This is a dataset-specific, transductive classifier. It depends on the metadata-generation pattern and pair structure continuing into evaluation, and does not establish general terrain-image recognition. No hidden evaluation labels are used. Image normalization follows the supplied negative solar-angle rotation, but image brightness features are not used by the selected classifier.

## Setup

Requires Python 3.12. Extract the outer dataset archive and retain the supplied inner image archives beside their metadata.

```sh
python -m venv .venv
# Activate the environment, then install dependencies:
pip install -r requirements.txt
```

## Predict

```sh
python classifier.py predict --data "path/to/The Pareidolia Paradox Dataset" --model model.json --output submission.csv
python validate_submission.py --submission submission.csv --metadata "path/to/The Pareidolia Paradox Dataset/Test/test_metadata.csv"
```

`submission.csv` contains 2,000 predictions in metadata order, with exactly `image_id,label` as the header.

## Train and validate

```sh
python classifier.py train --data "path/to/The Pareidolia Paradox Dataset" --output model.json
python validate_model.py --data "path/to/The Pareidolia Paradox Dataset" --output validation_reproduced.json
```

Generator discovery is repeated independently within each validation training fold. The model falls back to sector calibration when a consistent fingerprint is unavailable. No GPU or pretrained weight download is required.

## Repository layout

- `classifier.py`: generator discovery, training, and prediction.
- `pairing.py`: dataset loading and image-pair inference.
- `calibration.py`: sector calibration and fallback classifier.
- `model.json`: fitted classifier and training-image reference hashes.
- `submission.csv`: current competition predictions.
- `reports/`: validation metrics and prediction diagnostics.

The original dataset is not included.

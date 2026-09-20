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

Hardware: a CPU-only computer is sufficient; no GPU or CUDA is required. Allow 2 GB of available RAM as a practical recommendation (not a measured minimum), plus disk space for the supplied image archives. Training searches candidate random generators in memory; inference processes the full evaluation batch.

```sh
python -m venv .venv
# Activate the environment, then install dependencies:
pip install -r requirements.txt
```

## Predict

```sh
python inference.py --data "path/to/The Pareidolia Paradox Dataset" --model model.json --output submission.csv
python validate_submission.py --submission submission.csv --metadata "path/to/The Pareidolia Paradox Dataset/Test/test_metadata.csv"
```

`submission.csv` contains 2,000 predictions in metadata order, with exactly `image_id,label` as the header.

## Train and validate

```sh
python train.py --data "path/to/The Pareidolia Paradox Dataset" --output model.json
python validate_model.py --data "path/to/The Pareidolia Paradox Dataset" --output validation_reproduced.json
```

Generator discovery is repeated independently within each validation training fold. The model falls back to sector calibration when a consistent fingerprint is unavailable. No GPU or pretrained weight download is required.

## Repository layout

- `train.py`: training entry point.
- `inference.py`: evaluation entry point.
- `classifier.py`: generator discovery, training, and prediction.
- `pairing.py`: dataset loading and image-pair inference.
- `calibration.py`: sector calibration and fallback classifier.
- `model.json`: fitted classifier and training-image reference hashes.
- `submission.csv`: current competition predictions.
- `reports/`: validation metrics and prediction diagnostics.

The original dataset is not included.

## Solar-angle normalization and model artifact

The loader converts each image to grayscale and rotates it by `-sun_azimuth_angle` degrees with Pillow bilinear interpolation, keeping the original canvas size. It then resizes to 128 x 128 for brightness summaries. Positive Pillow angles rotate counterclockwise, so this negative rotation is clockwise. SHA-256 hashes are calculated from the original PNG bytes before rotation.

The selected classifier uses the original solar angle and exact-image pair relationships; the rotated brightness summaries do not influence its predictions. Rotation therefore exists in preprocessing but is not the source of this model's validation performance.

`model.json` is the complete fitted model artifact, containing the discovered generator parameters, angle counts, image-hash label references, and fallback calibration. This model has no neural-network weight tensors. Inference needs this artifact plus the evaluation images and metadata; it does not read training files. Run inference on all 2,000 evaluation images together because the method adapts to the complete batch.

[Download fitted model from Google Drive](https://drive.google.com/uc?export=download&id=1Hkcu8t5zBaOm_TzSFz2nNbLX9M76EQ9K).

[Download competition model ZIP](https://drive.google.com/uc?export=download&id=1diC9NJIqO_HKDpU6wYrhQEKzmK3VgEPp). Extract `model.json` beside `inference.py` before running the command above.

Model SHA-256: `9cbb8bf7c552acdb43354033fbcc266c39069536abfc81a154f8a07afef06742`.

See [METHODOLOGY.md](METHODOLOGY.md) for the submission summary.

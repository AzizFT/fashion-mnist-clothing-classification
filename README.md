# Fashion-MNIST Clothing Classification

Milestone 1 repository for a deep learning project that classifies grayscale clothing images into one of ten categories.

## Project summary

- **Task:** Multiclass image classification
- **Input:** One 28 x 28 grayscale clothing image
- **Output:** One of ten Fashion-MNIST categories
- **Dataset:** 70,000 labeled images (60,000 original training images and 10,000 official test images)
- **Planned split:** 48,000 training, 12,000 validation, and 10,000 test images
- **Validation metric:** Classification accuracy
- **Planned main model:** Convolutional Neural Network (CNN)

## Repository structure

```
fashion-mnist-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── inspect_dataset.py
└── figures/
```

## Setup and execution

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\\Scripts\\activate       # Windows
pip install -r requirements.txt
python src/inspect_dataset.py
```

The script downloads Fashion-MNIST through Keras, performs the predetermined stratified split, prints dataset statistics, and creates:

- `figures/representative_samples.png`
- `figures/class_distribution.png`
- `figures/dataset_summary.csv`
- `figures/split_summary.csv`

Run the script before writing the final PDF, then insert the two PNG figures into the proposal.

## Google Colab option

If TensorFlow installation is difficult locally, upload this repository to Google Drive or clone it in Google Colab and run:

```python
!pip install -r requirements.txt
!python src/inspect_dataset.py
```

Download the generated files from the `figures` directory.

## Reproducibility and leakage prevention

- Random seed: `42`
- The 60,000 original training examples are split into 48,000 training and 12,000 validation examples.
- Splitting is stratified so every class keeps the same proportion.
- The official 10,000-image test set remains untouched during model development.
- No model is trained in Milestone 1.

## Dataset and license

Fashion-MNIST was introduced by Han Xiao, Kashif Rasul, and Roland Vollgraf. The data is publicly available under the MIT license.

- Dataset repository: https://github.com/zalandoresearch/fashion-mnist
- Dataset paper: https://arxiv.org/abs/1708.07747
- Keras loader: https://keras.io/api/datasets/fashion_mnist/

## Authors

- Abdulaziz Altwaijri
- Abdulmalik Al Alshaikh


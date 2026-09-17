"""Inspect Fashion-MNIST and generate all Milestone 1 supporting artifacts.

No model is trained by this script. The official test set remains untouched.
"""

from pathlib import Path
import csv
import random

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import fashion_mnist


SEED = 42
VALIDATION_FRACTION = 0.20
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "figures"


def stratified_train_validation_split(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic, stratified train and validation indices."""
    rng = np.random.default_rng(SEED)
    train_parts = []
    validation_parts = []

    for class_id in range(len(CLASS_NAMES)):
        class_indices = np.flatnonzero(labels == class_id)
        rng.shuffle(class_indices)
        validation_size = int(len(class_indices) * VALIDATION_FRACTION)
        validation_parts.append(class_indices[:validation_size])
        train_parts.append(class_indices[validation_size:])

    train_indices = np.concatenate(train_parts)
    validation_indices = np.concatenate(validation_parts)
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return train_indices, validation_indices


def save_representative_samples(images: np.ndarray, labels: np.ndarray) -> None:
    """Save one actual dataset image from each class."""
    fig, axes = plt.subplots(2, 5, figsize=(12, 6.5))
    for class_id, axis in enumerate(axes.flat):
        sample_index = np.flatnonzero(labels == class_id)[0]
        axis.imshow(images[sample_index], cmap="gray")
        axis.set_title(CLASS_NAMES[class_id], fontsize=10, pad=7)
        axis.axis("off")

    fig.suptitle("Representative Fashion-MNIST Samples", fontsize=15, fontweight="bold")
    fig.subplots_adjust(left=0.03, right=0.99, bottom=0.04, top=0.86, wspace=0.07, hspace=0.34)
    fig.savefig(FIGURES_DIR / "representative_samples.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_class_distribution(train_labels: np.ndarray, test_labels: np.ndarray) -> None:
    """Save class-frequency plot for the complete dataset."""
    all_labels = np.concatenate([train_labels, test_labels])
    counts = np.bincount(all_labels, minlength=len(CLASS_NAMES))

    fig, axis = plt.subplots(figsize=(11, 5.5))
    bars = axis.bar(CLASS_NAMES, counts, color="#31688e", edgecolor="black", linewidth=0.6)
    axis.set_title("Fashion-MNIST Class Distribution", fontsize=15, fontweight="bold")
    axis.set_xlabel("Clothing category")
    axis.set_ylabel("Number of images")
    axis.set_ylim(0, max(counts) * 1.14)
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, count in zip(bars, counts):
        axis.text(bar.get_x() + bar.get_width() / 2, count + 90, f"{count:,}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_csv_files(
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> None:
    """Save exact class counts and split sizes for documentation."""
    with (FIGURES_DIR / "dataset_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["label_id", "class_name", "original_train", "official_test", "total"])
        for class_id, class_name in enumerate(CLASS_NAMES):
            original_train = int(np.sum(train_labels == class_id))
            official_test = int(np.sum(test_labels == class_id))
            writer.writerow([class_id, class_name, original_train, official_test, original_train + official_test])

    split_rows = [
        ("Training", len(train_indices), "Model parameter learning"),
        ("Validation", len(validation_indices), "Model selection and overfitting monitoring"),
        ("Test", len(test_labels), "Final unbiased evaluation only"),
    ]
    with (FIGURES_DIR / "split_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["split", "number_of_images", "purpose"])
        writer.writerows(split_rows)


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_indices, validation_indices = stratified_train_validation_split(train_labels)

    assert train_images.shape == (60_000, 28, 28)
    assert test_images.shape == (10_000, 28, 28)
    assert len(train_indices) == 48_000
    assert len(validation_indices) == 12_000
    assert set(train_indices).isdisjoint(set(validation_indices))
    assert np.all(np.bincount(train_labels[train_indices], minlength=10) == 4_800)
    assert np.all(np.bincount(train_labels[validation_indices], minlength=10) == 1_200)
    assert np.all(np.bincount(test_labels, minlength=10) == 1_000)

    save_representative_samples(train_images, train_labels)
    save_class_distribution(train_labels, test_labels)
    write_csv_files(train_labels, test_labels, train_indices, validation_indices)

    print("Fashion-MNIST inspection completed successfully.")
    print(f"Original training shape: {train_images.shape}")
    print(f"Official test shape:     {test_images.shape}")
    print(f"Planned training size:   {len(train_indices):,}")
    print(f"Planned validation size: {len(validation_indices):,}")
    print(f"Untouched test size:     {len(test_labels):,}")
    print(f"Pixel range:             {train_images.min()} to {train_images.max()}")
    print(f"Outputs saved to:        {FIGURES_DIR}")


if __name__ == "__main__":
    main()

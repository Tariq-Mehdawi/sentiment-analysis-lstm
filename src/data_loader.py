"""
data_loader.py

Downloads the IMDB Movie Reviews dataset (Stanford aclImdb) and loads it
into pandas DataFrames with 'review' and 'sentiment' columns.
"""

import os
import shutil

import pandas as pd
import tensorflow as tf

DATASET_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


def download_dataset(cache_dir="."):
    """Downloads and extracts the aclImdb dataset. Returns the path to the extracted folder."""
    archive_path = tf.keras.utils.get_file(
        fname="aclImdb_v1.tar.gz",
        origin=DATASET_URL,
        extract=True,
        cache_dir=cache_dir,
        cache_subdir="",
    )
    dataset_dir = os.path.join(os.path.dirname(archive_path), "aclImdb")

    # The train folder ships with an extra 'unsup' (unlabeled) folder we don't need
    unsup_dir = os.path.join(dataset_dir, "train", "unsup")
    if os.path.exists(unsup_dir):
        shutil.rmtree(unsup_dir)

    return dataset_dir


def load_dataset(directory):
    """Reads all .txt review files in directory/pos and directory/neg into a DataFrame."""
    reviews, sentiments = [], []
    for label_type, label_value in [("pos", 1), ("neg", 0)]:
        folder = os.path.join(directory, label_type)
        for fname in os.listdir(folder):
            if fname.endswith(".txt"):
                with open(os.path.join(folder, fname), encoding="utf-8") as f:
                    reviews.append(f.read())
                sentiments.append(label_value)

    df = pd.DataFrame({"review": reviews, "sentiment": sentiments})
    # Files are read in pos-then-neg order, so shuffle before using them
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


if __name__ == "__main__":
    dataset_dir = download_dataset()

    train_df = load_dataset(os.path.join(dataset_dir, "train"))
    test_df = load_dataset(os.path.join(dataset_dir, "test"))

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    train_df.to_csv("data/raw/train.csv", index=False)
    test_df.to_csv("data/raw/test.csv", index=False)

"""
evaluate.py

Evaluates the trained model on the held-out test set and writes plots.

Usage:
    python src/evaluate.py
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score
)

from src.preprocess import clean_text, encode, load_tokenizer

RAW_DIR = 'data/raw'
MODEL_DIR = 'models'
PLOT_DIR = 'outputs/plots'


def plot_training_curves(history, out_path):
    epochs = range(1, len(history['accuracy']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, [a * 100 for a in history['accuracy']],
                 'o-', label='Train')
    axes[0].plot(epochs, [a * 100 for a in history['val_accuracy']],
                 's-', label='Validation')
    axes[0].set_title('Model Accuracy', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history['loss'], 'o-', label='Train')
    axes[1].plot(epochs, history['val_loss'], 's-', label='Validation')
    axes[1].set_title('Model Loss', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, out_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix - Test Set', fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    model = load_model(f'{MODEL_DIR}/sentiment_lstm_model.keras')
    tokenizer = load_tokenizer(f'{MODEL_DIR}/tokenizer.pkl')
    with open(f'{MODEL_DIR}/config.json') as f:
        config = json.load(f)

    test_df = pd.read_csv(f'{RAW_DIR}/test.csv')
    cleaned = test_df['review'].apply(clean_text).values
    X_test = encode(cleaned, tokenizer, max_len=config['MAX_LEN'])
    y_test = test_df['sentiment'].values

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    print("=" * 55)
    print("FINAL TEST SET RESULTS")
    print("=" * 55)
    print(f"Test accuracy : {test_acc * 100:.2f}%")
    print(f"Test loss     : {test_loss:.4f}")
    print(f"ROC-AUC score : {roc_auc_score(y_test, y_prob) * 100:.2f}%")
    print("=" * 55)
    print("\nCLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred,
                                target_names=['Negative', 'Positive'],
                                digits=4))

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, f'{PLOT_DIR}/confusion_matrix.png')

    history_path = 'outputs/history.pkl'
    if os.path.exists(history_path):
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
        plot_training_curves(history, f'{PLOT_DIR}/training_curves.png')

    print(f"Plots written to {PLOT_DIR}/")


if __name__ == '__main__':
    main()

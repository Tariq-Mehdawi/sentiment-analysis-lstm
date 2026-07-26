"""
train.py

Trains the sentiment model and saves the model, tokenizer, and config.

Usage:
    python src/train.py
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)

from src.preprocess import prepare_data, save_tokenizer, VOCAB_SIZE, MAX_LEN
from src.model import build_model, EMBEDDING_DIM

RAW_DIR = 'data/raw'
MODEL_DIR = 'models'
OUTPUT_DIR = 'outputs'
EPOCHS = 10
BATCH_SIZE = 64


class PercentLogger(tf.keras.callbacks.Callback):
    """Prints per-epoch metrics with accuracy as a percentage."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"Epoch {epoch + 1:>2} | "
              f"train acc: {logs['accuracy'] * 100:6.2f}% | "
              f"val acc: {logs['val_accuracy'] * 100:6.2f}% | "
              f"train loss: {logs['loss']:.4f} | "
              f"val loss: {logs['val_loss']:.4f}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df = pd.read_csv(f'{RAW_DIR}/train.csv')
    test_df = pd.read_csv(f'{RAW_DIR}/test.csv')

    X_train, y_train, X_val, y_val, X_test, y_test, tokenizer = prepare_data(
        train_df, test_df
    )
    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    model = build_model()
    model.summary()

    # All callbacks monitor val_loss so the restored, checkpointed, and
    # reported model are guaranteed to be the same one.
    callbacks = [
        PercentLogger(),
        EarlyStopping(monitor='val_loss', patience=2,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(f'{MODEL_DIR}/best_model.keras', monitor='val_loss',
                        save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=1, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=0,
    )

    h = history.history
    best = int(np.argmin(h['val_loss']))

    print("\n" + "=" * 55)
    print("TRAINING SUMMARY")
    print("=" * 55)
    print(f"Epochs run          : {len(h['accuracy'])}")
    print(f"Best epoch          : {best + 1}")
    print(f"Best val accuracy   : {h['val_accuracy'][best] * 100:.2f}%")
    print(f"Train acc (best ep) : {h['accuracy'][best] * 100:.2f}%")
    print(f"Overfitting gap     : "
          f"{(h['accuracy'][best] - h['val_accuracy'][best]) * 100:+.2f}%")
    print("=" * 55)

    model.save(f'{MODEL_DIR}/sentiment_lstm_model.keras')
    save_tokenizer(tokenizer, f'{MODEL_DIR}/tokenizer.pkl')

    with open(f'{OUTPUT_DIR}/history.pkl', 'wb') as f:
        pickle.dump(h, f)

    config = {
        'VOCAB_SIZE': VOCAB_SIZE,
        'MAX_LEN': MAX_LEN,
        'EMBEDDING_DIM': EMBEDDING_DIM,
        'architecture': ('Embedding -> Bidirectional LSTM(64) -> Dropout '
                         '-> Dense(32) -> Dense(1)'),
        'best_epoch': best + 1,
        'val_accuracy': float(h['val_accuracy'][best]),
    }
    with open(f'{MODEL_DIR}/config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Model, tokenizer, and config saved to {MODEL_DIR}/")


if __name__ == '__main__':
    main()

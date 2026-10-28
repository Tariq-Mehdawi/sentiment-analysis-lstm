"""
model.py

Bidirectional LSTM architecture for binary sentiment classification.
"""

from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Embedding
from tensorflow.keras.models import Sequential

VOCAB_SIZE = 10000
MAX_LEN = 200
EMBEDDING_DIM = 128
LSTM_UNITS = 64
DROPOUT_RATE = 0.5


def build_model(vocab_size=VOCAB_SIZE,
                max_len=MAX_LEN,
                embedding_dim=EMBEDDING_DIM,
                lstm_units=LSTM_UNITS,
                dropout_rate=DROPOUT_RATE):
    """Embedding -> BiLSTM -> Dropout -> Dense -> Dropout -> Dense(sigmoid).

    The embedding layer holds ~93% of the model's parameters: most of what
    this network learns is word semantics, trained from scratch on the corpus.
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        Bidirectional(LSTM(lstm_units, return_sequences=False)),
        Dropout(dropout_rate),
        Dense(32, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid'),
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'],
    )

    model.build(input_shape=(None, max_len))
    return model

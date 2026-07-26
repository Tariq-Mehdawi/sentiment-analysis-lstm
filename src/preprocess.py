"""
preprocess.py

Text cleaning, tokenization, and padding for the IMDB sentiment dataset.
"""

import re
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE = 10000
MAX_LEN = 200


def clean_text(text):
    """Strip HTML tags, punctuation, and digits; collapse whitespace; lowercase.

    Deliberately does NOT remove stopwords: most stopword lists include
    negation words ("not", "no"), which carry critical sentiment signal
    for a sequential model.
    """
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def build_tokenizer(train_texts, vocab_size=VOCAB_SIZE):
    """Fit a tokenizer on TRAINING TEXT ONLY to avoid data leakage."""
    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts(train_texts)
    return tokenizer


def encode(texts, tokenizer, max_len=MAX_LEN):
    """Convert raw text to padded integer sequences."""
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_len,
                         padding='post', truncating='post')


def prepare_data(train_df, test_df, val_size=0.1, random_state=42):
    """Full pipeline: clean -> split -> fit tokenizer -> encode.

    Returns (X_train, y_train, X_val, y_val, X_test, y_test, tokenizer).
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['clean_review'] = train_df['review'].apply(clean_text)
    test_df['clean_review'] = test_df['review'].apply(clean_text)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_df['clean_review'].values,
        train_df['sentiment'].values,
        test_size=val_size,
        random_state=random_state,
        stratify=train_df['sentiment'].values,
    )

    tokenizer = build_tokenizer(train_texts)

    X_train = encode(train_texts, tokenizer)
    X_val = encode(val_texts, tokenizer)
    X_test = encode(test_df['clean_review'].values, tokenizer)

    y_train = np.asarray(train_labels)
    y_val = np.asarray(val_labels)
    y_test = test_df['sentiment'].values

    return X_train, y_train, X_val, y_val, X_test, y_test, tokenizer


def save_tokenizer(tokenizer, path):
    with open(path, 'wb') as f:
        pickle.dump(tokenizer, f)


def load_tokenizer(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

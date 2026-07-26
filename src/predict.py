"""
predict.py

Runs the trained model on arbitrary text.

Usage:
    python src/predict.py "This movie completely blew me away."
"""

import sys
import json
from tensorflow.keras.models import load_model

from src.preprocess import clean_text, encode, load_tokenizer


class SentimentPredictor:
    """Loads the model, tokenizer, and config as a matched set.

    All three must come from the same training run: a mismatched tokenizer
    or sequence length produces silently wrong predictions rather than
    an error.
    """

    def __init__(self, model_dir='models'):
        self.model = load_model(f'{model_dir}/sentiment_lstm_model.keras')
        self.tokenizer = load_tokenizer(f'{model_dir}/tokenizer.pkl')
        with open(f'{model_dir}/config.json') as f:
            self.config = json.load(f)

    def predict(self, texts):
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        cleaned = [clean_text(t) for t in texts]
        padded = encode(cleaned, self.tokenizer,
                        max_len=self.config['MAX_LEN'])
        probs = self.model.predict(padded, verbose=0).flatten()

        results = []
        for text, prob in zip(texts, probs):
            label = 'POSITIVE' if prob > 0.5 else 'NEGATIVE'
            confidence = float(prob if prob > 0.5 else 1 - prob)
            results.append({
                'text': text,
                'label': label,
                'score': float(prob),
                'confidence': confidence,
            })

        return results[0] if single else results


def main():
    if len(sys.argv) < 2:
        print('Usage: python src/predict.py "your text here"')
        sys.exit(1)

    predictor = SentimentPredictor()
    result = predictor.predict(' '.join(sys.argv[1:]))

    print(f'"{result["text"]}"')
    print(f'   -> {result["label"]} '
          f'(confidence: {result["confidence"] * 100:.2f}%)')
    print(f'   positive score: {result["score"] * 100:.2f}%')


if __name__ == '__main__':
    main()

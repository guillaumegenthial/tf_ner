"""Reload and serve a saved model"""

__author__ = "Guillaume Genthial"

from pathlib import Path
from tensorflow.contrib import predictor


LINE = 'John lives in New York'


def parse_fn(line):
    # Encode in Bytes for TF
    words = [w.encode() for w in line.strip().split()]

    # Chars
    chars = [[c.encode() for c in w] for w in line.strip().split()]
    lengths = [len(c) for c in chars]
    max_len = max(lengths)
    chars = [c + [b'<pad>'] * (max_len - l) for c, l in zip(chars, lengths)]

    return {'words': [words], 'nwords': [len(words)],
            'chars': [chars], 'nchars': [lengths]}


if __name__ == '__main__':
    export_dir = 'saved_model'
    subdirs = [x for x in Path(export_dir).iterdir()
               if x.is_dir() and 'temp' not in str(x)]
    latest = str(sorted(subdirs)[-1])
    predict_fn = predictor.from_saved_model(latest)
    predictions = predict_fn(parse_fn(LINE))
    print(predictions)

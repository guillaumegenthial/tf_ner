"""Reload and serve a saved model"""

__author__ = "Guillaume Genthial"

from pathlib import Path
from tensorflow.contrib import predictor


LINE = 'John lives in New York'

if __name__ == '__main__':
    export_dir = 'saved_model'
    subdirs = [x for x in Path(export_dir).iterdir()
               if x.is_dir() and 'temp' not in str(x)]
    latest = str(sorted(subdirs)[-1])
    predict_fn = predictor.from_saved_model(latest)
    words = [w.encode() for w in LINE.split()]
    nwords = len(words)
    predictions = predict_fn({'words': [words], 'nwords': [nwords]})
    print(predictions)

"""Interact with a model"""

__author__ = "Guillaume Genthial"

from pathlib import Path
import functools
import json

import tensorflow as tf

from main import model_fn

LINE = 'John lives in New York'
DATADIR = '../../data/example'
PARAMS = './results/params.json'
MODELDIR = './results/model'


def pretty_print(line, preds):
    words = line.strip().split()
    lengths = [max(len(w), len(p)) for w, p in zip(words, preds)]
    padded_words = [w + (l - len(w)) * ' ' for w, l in zip(words, lengths)]
    padded_preds = [p.decode() + (l - len(p)) * ' ' for p, l in zip(preds, lengths)]
    print('words: {}'.format(' '.join(padded_words)))
    print('preds: {}'.format(' '.join(padded_preds)))


def predict_input_fn(line):
    # Words
    words = [w.encode() for w in line.strip().split()]
    nwords = len(words)

    # Wrapping in Tensors
    words = tf.constant([words], dtype=tf.string)
    nwords = tf.constant([nwords], dtype=tf.int32)

    return (words, nwords), None


if __name__ == '__main__':
    with Path(PARAMS).open() as f:
        params = json.load(f)

    params['words'] = str(Path(DATADIR, 'vocab.words.txt'))
    params['chars'] = str(Path(DATADIR, 'vocab.chars.txt'))
    params['tags'] = str(Path(DATADIR, 'vocab.tags.txt'))
    params['glove'] = str(Path(DATADIR, 'glove.npz'))

    estimator = tf.estimator.Estimator(model_fn, MODELDIR, params=params)
    predict_inpf = functools.partial(predict_input_fn, LINE)
    for pred in estimator.predict(predict_inpf):
        pretty_print(LINE, pred['tags'])
        break

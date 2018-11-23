"""Export model as a saved_model"""

__author__ = "Guillaume Genthial"

from pathlib import Path
import json

import tensorflow as tf

from main import model_fn

DATADIR = '../../data/example'
PARAMS = './results/params.json'
MODELDIR = './results/model'


def serving_input_receiver_fn():
    """Serving input_fn that builds features from placeholders

    Returns
    -------
    tf.estimator.export.ServingInputReceiver
    """
    words = tf.placeholder(dtype=tf.string, shape=[None, None], name='words')
    nwords = tf.placeholder(dtype=tf.int32, shape=[None], name='nwords')
    chars = tf.placeholder(dtype=tf.string, shape=[None, None, None],
                           name='chars')
    nchars = tf.placeholder(dtype=tf.int32, shape=[None, None],
                            name='nchars')
    receiver_tensors = {'words': words, 'nwords': nwords,
                        'chars': chars, 'nchars': nchars}
    features = {'words': words, 'nwords': nwords,
                'chars': chars, 'nchars': nchars}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


if __name__ == '__main__':
    with Path(PARAMS).open() as f:
        params = json.load(f)

    params['words'] = str(Path(DATADIR, 'vocab.words.txt'))
    params['chars'] = str(Path(DATADIR, 'vocab.chars.txt'))
    params['tags'] = str(Path(DATADIR, 'vocab.tags.txt'))
    params['glove'] = str(Path(DATADIR, 'glove.npz'))

    estimator = tf.estimator.Estimator(model_fn, MODELDIR, params=params)
    estimator.export_saved_model('saved_model', serving_input_receiver_fn)

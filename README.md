# Tensorflow Named Entity Recognition

Each folder contains a standalone, short (<100 lines of Tensorflow), `main.py` that implements a neural-network based model for NER using `tf.estimator`.


## Install

You need to install [`tf_metrics` ](https://github.com/guillaumegenthial/tf_metrics).

## Data

Follow the example dataset in `data/example`:

1. For `name` in `{train, test, ...}`, create files `{name}.words.txt` and `{name}.tags.txt` that contain one sentence per line, each
word separated by space.
2. Create files `vocab.words.txt`, `vocab.tags.txt` and `vocab.chars.txt` that contain one token per line.
3. Create a `glove.npz` matrix of shape `(size_vocab_words, 300)` using [GloVe 840B vectors](https://nlp.stanford.edu/projects/glove/) and [`np.savez_compressed`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.savez_compressed.html).


## Models

Implement these papers

- [Bidirectional LSTM-CRF Models for Sequence Tagging by Huang, Xu and Yu](https://arxiv.org/abs/1508.01991)
- [Neural Architectures for Named Entity Recognition by Lample et al.](https://arxiv.org/abs/1603.01360)
- [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF by Ma et Hovy](https://arxiv.org/abs/1603.01354)

---

### `lstm_crf`

__Paper__: [Bidirectional LSTM-CRF Models for Sequence Tagging by Huang, Xu and Yu](https://arxiv.org/abs/1508.01991)

1. GloVe embeddings
2. Bi-LSTM
3. CRF

---

### `chars_lstm_lstm_crf`

__Paper__: [Neural Architectures for Named Entity Recognition by Lample et al.](https://arxiv.org/abs/1603.01360)

1. GloVe embeddings
2. Chars embeddings
3. Chars bi-LSTM
4. Bi-LSTM
5. CRF

---


### `chars_conv_lstm_crf`

__Paper__: [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF by Ma et Hovy](https://arxiv.org/abs/1603.01354)

1. GloVe embeddings
2. Chars embeddings
3. Chars convolutions and max-pooling
4. Bi-LSTM
5. CRF

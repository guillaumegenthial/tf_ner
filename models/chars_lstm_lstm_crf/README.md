# Chars LSTM + bi-LSTM + CRF

__Architecture__

1. [GloVe 840B vectors](https://nlp.stanford.edu/projects/glove/)
2. Chars embeddings
3. Chars bi-LSTM
4. Bi-LSTM
5. CRF

__Related Paper__ [Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) by Lample et al.

__Training time__ ~ 35 min

|| `train` | `testa` | `testb` | Paper, `testb` |
|---|:---:|:---:|:---:|:---:|
|best| 98.81 | 94.36 | 91.02 | 90.94 |
|best (EMA) |98.73 | 94.50 | __91.14__ | |
|mean ± std | 98.83 ± 0.27| 94.02 ± 0.26| 91.01 ± 0.16 |  |
|mean ± std (EMA) | 98.51 ± 0.25| 94.20 ± 0.28| __91.21__ ± 0.05 |  |
|abs. best |   | |91.22 | |
|abs. best (EMA) | |   | 91.28 |  |

## 1. Format your data to the right format

Follow the [`data/example`](https://github.com/guillaumegenthial/tf_ner/tree/master/data/example) and add your data to `data/your_data` for instance.

1. For `name` in `{train, testa, testb}`, create files `{name}.words.txt` and `{name}.tags.txt` that contain one sentence per line, each
word / tag separated by space. I recommend using the `IOBES` tagging scheme.
2. Create files `vocab.words.txt`, `vocab.tags.txt` and `vocab.chars.txt` that contain one lexeme per line.
3. Create a `glove.npz` file containing one array `embeddings` of shape `(size_vocab_words, 300)` using [GloVe 840B vectors](https://nlp.stanford.edu/projects/glove/) and [`np.savez_compressed`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.savez_compressed.html).

An example of scripts to build the `vocab` and the `glove.npz` files from the  `{name}.words.txt` and `{name}.tags.txt` files is provided in [`data/example`](https://github.com/guillaumegenthial/tf_ner/tree/master/data/example). See

1. [`build_vocab.py`](https://github.com/guillaumegenthial/tf_ner/blob/master/data/example/build_vocab.py)
2. [`build_glove.py`'](https://github.com/guillaumegenthial/tf_ner/blob/master/data/example/build_glove.py)

![Data Format](../../images/data.png)

If you just want to get started, once you have created your `{name}.words.txt` and `{name}.tags.txt` files, simply do

```
cd data/example
make download-glove
make build
```

## 2. Update the `main.py` script

Change the relative path `DATADIR` from the `main.py` script to the directory containing your data.

For example, `DATADIR = '../../data/your_data'`.


## 3. Run `python main.py`

Using python3, it will run training with early stopping and write predictions under `results/score/{name}.preds.txt`.

## 4. Run the `conlleval` script on the predictions

Usage: `../conlleval < results/score/{name}.preds.txt > results/score/score.{name}.metrics.txt`

## 5. Run `python interact.py`

It reloads the trained estimator and computes predicitons on a simple example

## 6. Run `python export.py`

It exports the estimator inference graph under `saved_model`. We will re-use this for serving.

## 7. Run `python serve.py`

It reloads the saved model using `tf.contrib.predictor` and computes prediction on a simple example.

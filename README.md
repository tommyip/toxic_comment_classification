# Toxic comment classification

Jupyter notebooks for my HKUST COMP4901K course final project.

The dataset is from the Kaggle competition [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview).

## Notes

* Due to limited computational resource, the models are only trained on a subset of the whole dataset (training=5000, validation=1000, testing=1000).

* The RNGs are seeded so the result should be reproducible.

## Requirements

Tested with Python 3.8.5.

Dependencies:
* numpy
* pandas
* matplotlib
* spacy
* scikit-learn

Install with

```sh
pip install numpy pandas matplotlib spacy sklearn

# Need the large model for word embeddings
python -m spacy download en_core_web_md
```
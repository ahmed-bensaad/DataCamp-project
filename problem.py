import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np


problem_title = 'Traffic Signs Recognition'



Predictions = rw.prediction_types.make_multiclass(
    label_names=list(np.arange(43)))

workflow = rw.workflows.Classifier()

score_types = [
    rw.score_types.Accuracy(name='acc'),
    rw.score_types.ClassificationError(name='error'),
    rw.score_types.NegativeLogLikelihood(name='nll'),
    rw.score_types.F1Above(name='f1_70', threshold=0.7),
]

def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=57)
    return cv.split(X, y)


def _read_data(path, typ):
    """
    Read and process data and labels.

    Parameters
    ----------
    path : path to directory that has 'data' subdir
    typ : {'Train', 'Test'}

    Returns
    -------
    X, y data

    """
    test = os.getenv('RAMP_TEST_MODE', 0)


    try:
        data_path = os.path.join(path, 'data',
                                 '{}.csv'.format(typ))

        data = pd.read_csv(data_path)
    except IOError:
        raise IOError("'data/{0}.csv' is not "
                      "found. Run annotations_gen.py to get annotations".format(typ))

    X = data['Filename']
    Y = data['ClassId']

    if test:
        # return src, y
        return Y[:100], Y[:100]
    else:
        return X, Y


def get_test_data(path='.'):
    return _read_data(path, 'Test')


def get_train_data(path='.'):
    return _read_data(path, 'Train')

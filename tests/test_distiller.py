import packageloader
import pickle
import gzip

import pandas as pd

from distiller import data_distiller
from unittest import TestCase


class DataDistillerTest(TestCase):
    def setUp(self):
        pass

    def test_dataDistiller(self):
        datafile = 'sample_data.csv'
        filename = data_distiller(datafile)
        with gzip.open(filename, 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f)

        n_rows = pd.read_csv(datafile).shape[0]
        assert train_set[0].shape[0] == int(0.7*n_rows)
        assert train_set[0].shape[1] == 6
        assert train_set[1].shape[0] == int(0.7*n_rows)
        assert len(train_set[1].shape) == 1

        assert valid_set[0].shape[0] == int(0.9*n_rows) - int(0.7*n_rows)
        assert valid_set[0].shape[1] == 6
        assert valid_set[1].shape[0] == int(0.9*n_rows) - int(0.7*n_rows)
        assert len(valid_set[1].shape) == 1

        assert test_set[0].shape[0] == n_rows - int(0.9*n_rows)
        assert test_set[0].shape[1] == 6
        assert test_set[1].shape[0] == n_rows - int(0.9*n_rows)
        assert len(test_set[1].shape) == 1

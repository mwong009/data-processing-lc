import packageloader

import theano
import theano.tensor as T

import pytest

from unittest import TestCase
from train import load_data
from models import MLP, DBN


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class LoadDataTest(TestCase):
    def setUp(self):
        pass

    def test_loadData(self):
        datafile = 'dataset.pkl.gz'
        dataset = load_data(datafile)

    def test_createMLP(self):

        x = T.matrix('x')   # data
        y = T.ivector('y')  # labels
        is_train = T.iscalar('is_train')

        mlp = MLP(
            input=x,
            output=y,
            n_in=6,
            hidden_layers_sizes=[50],
            n_out=3,
            is_train=is_train
        )

        mlp_output = mlp.output

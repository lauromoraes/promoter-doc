import abc
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from promoter_dataset import dataset_manager as dm
from promoter_dataset.dataset_manager import DatasetManager
from sklearn import metrics

from icecream import ic as ic

import logging
import sys



class Experiment(object):
    def __init__(self, name: str, dataset_manager: DatasetManager):
        self.experiment_name = name
        self.data_manager = dataset_manager
        self.model_manager = None


    @abc.abstractmethod
    def setup(self, ):
        pass

    def exec(self):
        _dm = self.data_manager
        i = 0
        for (X_train, X_test), (y_train, y_test) in _dm.get_next_split():
            print(f'Split: {(i := i+1)}')

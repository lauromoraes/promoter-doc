import abc
from src.datamanager.dataset_manager import DatasetManager

import mlflow
import mlflow.sklearn

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
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(description=f'Parent run for {self.experiment_name}.'):
            i = 0
            for (X_train, X_test), (y_train, y_test) in _dm.get_next_split():
                print(f'Split: {(i := i+1)}')

import abc
import importlib
import mlflow
import mlflow.sklearn
from argparse import Namespace

from rich.console import Console

from src.datamanager.dataset_manager import FeaturesManager

class Experiment(object):
    def __init__(self, exp_args: Namespace = None):
        self.exp_args = exp_args
        self.experiment_name = self.exp_args.experiment_name
        self.features_manager = None
        self.models = None
        self.set_features()
        self.set_models()

    def set_models(self):
        models_args = self.exp_args.models_params
        for m_args in models_args:
            m_type = m_args['model_type']
            m_lib = m_args['lib']
            m_params = m_args['params']
            model_class = getattr(importlib.import_module(m_lib), m_type)
            model = model_class(**m_params)
            print(model)

    def set_features(self):
        raw_data_paths = [x['path'] for x in self.exp_args.datasets]
        features_args = self.exp_args.features
        cv = self.exp_args.cv

        self.features_manager = FeaturesManager(fasta_paths = raw_data_paths)
        self.features_manager.transform_raw_dataset(features_args)
        self.features_manager.setup_partitions(n_splits = cv)

    # @abc.abstractmethod
    def train(self, X, y):
        model = None
        return model

    def calculate_metrics(self, y, y_pred):
        metrics = dict()
        return metrics

    # @abc.abstractmethod
    def test(self, model, X, y):
        y_pred = None
        return y_pred

    def exec(self):
        console = Console()
        _dm = self.features_manager
        console.rule(f'Starting experiment: {self.experiment_name}')
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(description=f'Parent run for {self.experiment_name}.') as parent_run:
            # Prepare features
            self.set_features()

            run_idx = 0
            for (X_train, X_test), (y_train, y_test) in _dm.get_next_split():
                with mlflow.start_run(description=f'Child run {self.experiment_name}.') as run:
                    console.print(f'Split: {(run_idx := run_idx+1)}', style='blue')

                    model = self.train(X_train, y_train)
                    self.test(model, X_test, y_test)

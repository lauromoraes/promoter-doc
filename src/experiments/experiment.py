import abc
import importlib
import mlflow
import mlflow.sklearn
from argparse import Namespace

from rich.console import Console

from src.datamanager.dataset_manager import FeaturesManager

console = Console(color_system="windows")

class Experiment(object):
    def __init__(self, exp_args: Namespace = None):
        self.exp_args = exp_args
        self.experiment_name = self.exp_args.experiment_name
        self.features_manager = None
        self.models = None
        self.set_features()
        self.set_models()

    def set_models(self):
        ''' Instantiate each model from definitions on the experiment
        configuration file.

        :return: A list with all instantiated models
        '''

        def get_model(model_name: str, import_module: str, model_params: dict):
            ''' Local function to instantiate models using configuration file data

            :param model_name: Class name of the model
            :param import_module: Module of the class
            :param model_params: Hyperparameters of the model
            :return: A new instantiated model
            '''
            model_class = getattr(importlib.import_module(import_module), model_name)
            model = model_class(**model_params)
            return model

        models = list()
        models_args = self.exp_args.models_params
        for m_args in models_args:
            # Get info from arguments file
            m_name = m_args['model_name']
            m_module = m_args['module']
            m_params = m_args['params']
            # Instantiate new model
            model = get_model(m_name, m_module, m_params)
            models.append(model)

        self.models = models
        return self.models

    def set_features(self):
        ''' Extract features from raw data and stores them into a management
        object.

        :return: The features manager object
        '''
        # Get info from arguments file
        raw_data_paths = [x['path'] for x in self.exp_args.datasets]
        features_args = self.exp_args.features
        cv = self.exp_args.cv
        # Instantiate object to transform raw data into features
        self.features_manager = FeaturesManager(fasta_paths=raw_data_paths)
        # Extract features from raw data
        self.features_manager.transform_raw_dataset(features_args)
        # Split data into Cross validation folds
        self.features_manager.setup_partitions(n_splits=cv)

        return self.features_manager

    # @abc.abstractmethod
    def train(self, X, y):
        for model in self.models:
            console.print(f"Training model:\n\t{model}")
            model_type = str(type(model))
            if 'sklearn' in model_type:
                console.print('Training a Sklearn model')
                model.fit(X[0], y)
            elif 'tensorflow' in model_type:
                console.print('Training a Tensorflow model')
                model.fit(X, y)
        return self.models

    def calculate_metrics(self, y, y_pred):
        metrics = dict()
        return metrics

    # @abc.abstractmethod
    def test(self, model, X, y):
        y_pred = None
        return y_pred

    def exec(self):
        _dm = self.features_manager
        console.rule(f'Starting experiment: {self.experiment_name}')
        mlflow.sklearn.autolog()

        # Create or load experiment by the name in the config file
        try:
            mlflow_experiment = dict(mlflow.get_experiment_by_name(self.experiment_name))
            experiment_id = mlflow_experiment['experiment_id']
            console.print(f'Experiment already exists! Loading ([red]ID {experiment_id}[/red])')
            console.print(mlflow_experiment)
        except Exception as e:
            experiment_id = mlflow.create_experiment(self.experiment_name)
            console.print(e)
            console.print(f'New Experiment created ([red]ID {experiment_id}[/red])')

        # Setup experiment for data splits
        with mlflow.start_run(
                experiment_id=experiment_id,
                tags={'version': self.exp_args.experiment_version},
                description=f'Parent run for {self.experiment_name}.',
        ) as parent_run:
            console.print(f"Starting Experiment", justify="center", style='red')
            # Prepare features
            self.set_features()
            exp_params = {}
            mlflow.log_params(params=exp_params, )

            # Setup splits
            run_idx = 0
            for (X_train, X_test), (y_train, y_test) in _dm.get_next_split():

                # Setup run for one split
                with mlflow.start_run(
                        experiment_id=experiment_id,
                        run_name=f'SPLIT_{run_idx}',
                        nested=True,
                        description=f'Child run {self.experiment_name}.',
                ) as run:
                    console.rule(f"Running split: [red]{(run_idx := run_idx + 1)}[/red]")
                    # Execute TRAINING step
                    model = self.train(X_train, y_train)
                    # Execute TEST step
                    self.test(model, X_test, y_test)


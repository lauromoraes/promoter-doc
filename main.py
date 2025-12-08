# -*- coding: utf-8 -*-

# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import matplotlib
import matplotlib.pyplot as plt
from argparse import Namespace
from icecream import ic as ic
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
from rich.console import Console
from pycaret.classification import *
import time
import icecream as ic

from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_curve, roc_auc_score

import mlflow
import mlflow.sklearn

import importlib

from src.datamanager.dataset_manager import FeaturesManager
from src.utils.arguments import get_args

console = Console(color_system="windows")

SEED = 17

def convert_fasta_to_properties(fasta_paths: list[str] = None, kmer_type: str = 'dinuc'):
    # Convert fasta files to properties files
    organisms_paths = dict()
    for path in fasta_paths:
        organism_name = '_'.join(os.path.basename(path).split('_')[0:-1])
        organism_class = os.path.basename(path).split('_')[-1].split('.')[0]
        print(f'Organism: {organism_name} | Class: {organism_class} | Path: {path}')
        if organisms_paths.get(organism_name) is not None:
            organisms_paths[organism_name].append((organism_class, path))
        else:
            organisms_paths[organism_name] = [(organism_class, path)]

    print(f'Organisms: {organisms_paths.keys()}', end='\n\n')

    for organism_name, classes_tuple in organisms_paths.items():
        paths = [class_path for _, class_path in classes_tuple]
        extract_organism_prop_features(organism_name=organism_name, classes_paths=paths, kmers_type=kmer_type)

    organisms_names = list(organisms_paths.keys())
    return organisms_names


def extract_organism_prop_features(organism_name: str, classes_paths: list[str], params: list[dict] = None, kmers_type: str = 'dinuc'):
    print(f'Extracting features form FASTA files for Organism: {organism_name} ...')

    if params is None:
        # Stub to extract info from organism dataset
        # params_ = [{'k': 2, 'encode': 'prop', 'slice': [59, 20, 20]}]
        params_ = [{'k': 2 if kmers_type=='dinuc' else 3, 'encode': 'prop'}]

    dm = FeaturesManager(fasta_paths=(classes_paths))
    dm.transform_raw_dataset(params=params_)

    # Define number of datasets, classes and properties
    N_FEATURES_TYPES = len(dm.datasets)
    N_CLASSES = len(dm.datasets[0].encoded_classes_datasets)
    N_PROPS = len(dm.datasets[0].encoded_classes_datasets[0])

    print(f'N_FEATURES_TYPES: {N_FEATURES_TYPES} | N_CLASSES: {N_CLASSES} | N_PROPS: {N_PROPS}', end='\n\n')

    # Verify if folder exists
    # organism_path = f'./data/interim/{organism_name}'
    organism_path = os.path.join(os.getcwd(), 'data', 'interim', kmers_type, f'{organism_name}-original')
    if os.path.exists(organism_path) is False:
        os.makedirs(organism_path)


    feature_type_idx = 0

    props = list()
    for prop in range(N_PROPS):
        # verify if property was saved in csv file
        prop_file = os.path.join(organism_path, f'df_data_prop_{prop}.feather')
        classes_dataframes = list()
        for _class in range(N_CLASSES):
            print(f'Creating dataframe for property {prop} and saving to feather file ...')

            # create dataframe for each property and add class column (y) with 1 for positive class and 0 for negative
            prop_data = dm.datasets[feature_type_idx].encoded_classes_datasets[_class][prop]
            print(f'Prop Data : {prop_data}')
            df_data = pd.DataFrame(prop_data)
            df_data['y'] = _class
            classes_dataframes.append(df_data)

        df_data = pd.concat(classes_dataframes) # concatenate dataframes for each class

        print(df_data)


        # df_data.to_csv(prop_file, index=False) # save dataframe to csv file
        df_data.columns = df_data.columns.astype(str)
        print(f'Compressing dataframe to feather file ...')
        df_data.reset_index(drop=True).to_feather(prop_file, compression='zstd', compression_level=9) # save
        # dataframe to feather file

        props.append(df_data) # add dataframe to list of dataframes for each property (prop)


    print(f'Organism {organism_name} properties saved to feather files.', end='\n\n')

    return props

def get_organism_props_dataframes(organism_path: str):
    # verify if folder exists, if not create it
    if os.path.exists(organism_path) is False:
        os.makedirs(organism_path)  # create folder

    # Get all properties files from organism folder
    props_files = [f'{organism_path}/{file}' for file in os.listdir(organism_path) if file.startswith('df_data_prop_')]

    # Get dataframes for each property
    props = list()
    for prop_idx in range(len(props_files)):
        prop_file = f'{organism_path}/df_data_prop_{prop_idx}.feather'
        if os.path.exists(prop_file):
            df_data = pd.read_feather(prop_file)
            print(f'Property {prop_idx} dataframe loaded.', df_data, sep='\n')
            props.append(df_data)
        else:
            print(f'Property {prop_idx} feather file not found.')

    return props


def plot_prop_distribution(pos_data, neg_data, prop):
    # Plot distribution of property along the sequence

    # get dataframes for positive and negative classes
    p_data = pos_data[prop].drop(columns=['y'])
    n_data = neg_data[prop].drop(columns=['y'])

    # configure plot
    fig, ax = plt.subplots(2)
    fig.set_figwidth(15)
    fig.set_figheight(15)
    fig.suptitle(f'Property {prop}', fontsize=15)

    # plot boxplot and mean line for positive class
    ax[0].set_title(f'Positive class | {len(p_data)} sequences')
    ax[0].boxplot(p_data)
    ticks = ax[0].get_xticks()
    ax[0].plot(ticks, p_data.mean())

    # plot boxplot and mean line for negative class
    ax[1].set_title(f'Negative class | {len(n_data)} sequences')
    ax[1].boxplot(n_data)
    ticks = ax[1].get_xticks()
    ax[1].plot(ticks, n_data.mean())

    # show plot
    plt.show()

def plot_profile(org_name: str, org_df: pd.DataFrame, prop: int = None):

    # Setup folder to save profiles plots if not exists yet
    profiles_folder = os.path.join(os.getcwd(), 'reports', 'properties_profiles')
    if os.path.exists(profiles_folder) is False:
        os.mkdir(profiles_folder)

    n_classes = org_df['y'].unique().shape[0]

    # configure plot
    fig, ax = plt.subplots(n_classes)
    fig.set_figwidth(15)
    fig.set_figheight(15)
    fig.suptitle(f'Property {prop} of organism {org_name}', fontsize=15)

    for _class in range(n_classes):
        class_df = org_df[org_df['y'] == _class]
        class_df = class_df.drop(columns=['y'])

        ax[_class].set_title(f'Class {_class} | {len(class_df)} sequences')
        ax[_class].boxplot(class_df)
        ticks = ax[0].get_xticks()
        ax[_class].plot(ticks, class_df.mean())
        ax[_class].plot(ticks, class_df.median())

    # Save plot
    plt.savefig(f'{profiles_folder}/{org_name}_prop_{prop}.pdf', dpi=300, bbox_inches='tight')

    # show plot
    plt.show()

def main(args: Namespace) -> None:
    print(f'START MAIN FUNCTION')
    # exp = Experiment(exp_args=args)
    # exp.exec()

    # ---- START DATA PREPARATION ----

    # Define paths to fasta files
    pos_fasta = './data/raw-data/fasta/Bacillus_pos.fa'
    neg_fasta = './data/raw-data/fasta/Bacillus_neg.fa'

    # Prepare dataset object and transform raw dataset
    dm = FeaturesManager(fasta_paths=(pos_fasta, neg_fasta))
    dm.transform_raw_dataset(params=args.features)

    # Define number of datasets, classes and properties
    N_DATASETS = len(dm.datasets)
    N_CLASSES = len(dm.datasets[0].encoded_classes_datasets)
    N_PROPS = len(dm.datasets[0].encoded_classes_datasets[0])

    print(f'N_DATASETS: {N_DATASETS} | N_CLASSES: {N_CLASSES} | N_PROPS: {N_PROPS}', end='\n\n')

    # Setup indexes to select dataset, class and property
    _dataset = 0
    _class_pos = 0
    _class_neg = 1
    _prop = 12

    # Get dataframes for positive and negative classes
    pos_data = list()
    neg_data = list()
    props = list()
    for prop in range(N_PROPS):
        # verify if property was saved in feather file
        prop_file = f'./data/interim/df_data_prop_{prop}.feather'
        if not os.path.exists(prop_file):
            print(f'Property {prop} not found in feather file. Creating dataframe...')

            # create dataframe for each property and add class column (y) with 1 for positive class and 0 for negative
            pos = pd.DataFrame(dm.datasets[_dataset].encoded_classes_datasets[_class_pos][prop])
            neg = pd.DataFrame(dm.datasets[_dataset].encoded_classes_datasets[_class_neg][prop])
            pos['y'] = 1
            neg['y'] = 0
            # add dataframes to lists
            pos_data.append(pos)
            neg_data.append(neg)

            # concatenate dataframes for positive and negative classes
            df_data = pd.concat([pos_data[prop], neg_data[prop]], ignore_index=True)
            print(f'Property {prop} | df_data shape: {df_data.shape}')

            # add dataframe to list of dataframes for each property (prop)
            props.append(df_data)


            # df_data.to_csv(prop_file, index=False)  # save dataframe to csv file
            df_data.columns = df_data.columns.astype(str)
            df_data.reset_index(drop=True).to_feather(prop_file, compression='zstd', compression_level=9)  # save
            # dataframe to feather file
        else:
            # read dataframe from csv file and add to list of dataframes for each property (prop)
            df_data = pd.read_feather(prop_file, use_threads=True)
            props.append(df_data)

            print(f'Property {prop} feather file found. Reading dataframe...')
            print(f'Property {prop} | df_data shape: {df_data.shape}')

            # add dataframe to list of dataframes for each property (prop)
            pos = df_data[df_data['y'] == 1]
            neg = df_data[df_data['y'] == 0]

            # add dataframes to lists
            pos_data.append(pos)
            neg_data.append(neg)


    # ---- VIZUALIZE DATA ----

    # plot_prop_distribution(pos_data, neg_data, _prop)

    # ---- END DATA PREPARATION ----

    # ---- START MODEL TRAINING ----
    # Setup PyCaret experiment object with data and target column name (y) and other parameters to configure the
    # experiment (verbose, html, session_id, log_experiment, log_plots, log_profile, log_data, experiment_name,
    # use_gpu)
    df_prop = props[_prop]
    s = setup(data=df_prop, target='y', verbose=True,
              html=True, session_id=123, log_experiment=False, log_plots=True, log_profile=True, log_data=True,
              experiment_name='experiment', use_gpu=True)

    # compare all models
    best = compare_models()
    print(f'best model: {best}')

    # check the final params of best model
    best.get_params()
    print(f'best model params: {best.get_params()}')

    # ---- END MODEL TRAINING ----

    # ---- START MODEL EVALUATION ----

    evaluate_model(best)

    # ---- END MODEL EVALUATION ----

    plot_model(best, plot='confusion_matrix')

    print(f'END MAIN FUNCTION')

    # df_data = None
    #
    # # Setup PyCaret experiment object with data and target column name (y) and other parameters to configure the
    # # experiment (verbose, html, session_id, log_experiment, log_plots, log_profile, log_data, experiment_name,
    # # use_gpu)
    # s = ClassificationExperiment()
    # s.setup(df_data, target='y', verbose=True, html=True, session_id=123, log_experiment=False,
    #         log_plots=True, log_profile=True, log_data=True, experiment_name='experiment', use_gpu=True)
    #
    # # model training and selection
    # best = s.compare_models()
    #
    # console.print(f'Best model:\n\t{best}')
    #
    # s.plot_model(best, plot='residuals_interactive')
    #
    # # evaluate model
    # s.evaluate_model(best)
    #
    # # predict on holdout set
    # pred_holdout = s.predict_model(best)
    #
    # # predict on new dataset
    # new_df_data = df_data.copy().drop('y', axis=1)
    # predictions = s.predict_model(best, data=new_df_data)
    #
    # # save transformation pipeline and model
    # s.save_model(best, 'best-model')



def auto_extract_features(fasta_folder: str, kmer_type: str = 'dinuc'):
    # Get all FAST files paths in a folder
    print(f'Fasta folder: {fasta_folder}')
    organisms_names = set()
    fasta_paths = list()
    for fasta_file in os.listdir(fasta_folder):
        if fasta_file.endswith('.fa'):
            organism_name = fasta_file.split('_')[0]
            organisms_names.add(organism_name)
            print(f'Oganism name: {organism_name} | Fasta file: {fasta_file}')
            fasta_path = os.path.join(fasta_folder, fasta_file)
            fasta_paths.append(fasta_path)

    print(f'Fasta files paths: {fasta_paths}')
    organisms_names = convert_fasta_to_properties(fasta_paths, kmer_type)

def get_props_names(kmer_type: str = 'dinuc', original_values=True):
    fname = 'original' if original_values else 'normalized'
    kmers_values_folder = os.path.join(os.getcwd(), 'data', 'raw-data',
                                       'physicochemical-properties-reference',
                                           f'{fname}-{kmer_type}.tsv')
    kmer_df = pd.read_csv(kmers_values_folder, sep='\t', index_col=0)
    props_names = kmer_df.index.tolist()
    return props_names


if __name__ == '__main__':
    args: Namespace = get_args()
    print(f'Arguments:\n{args}', end='\n\n')
    # main(args)

    fasta_folder = os.path.join(os.getcwd(), 'data', 'raw-data', 'fasta')
    interim_folder = os.path.join(os.getcwd(), 'data', 'interim')

    # Set kmer type, get properties names and number of properties for that kmer type
    kmer_types = ('dinuc', 'trinuc')
    kmer_type = kmer_types[1]  # Select kmer type
    props_names = get_props_names(kmer_type=kmer_type)
    n_props = len(props_names)
    print(f'Kmer: {kmer_type} | Number of properties: {n_props}')
    print(f'Properties: {props_names}')

    extract_features = False
    if extract_features:
        # Extract features from fasta files
        for kmer_type in kmer_types:
            print(f'EXTRACTING FEATURES FOR KMER TYPE: {kmer_type}')
            auto_extract_features(fasta_folder, kmer_type)

    # Get all features dataframes for each organism
    organisms_names = (
        'Arabidopsis_non_tata',
        'Arabidopsis_tata',
        'Bacillus',
        'Ecoli',
        'Human_non_tata',
        'Mouse_non_tata',
        'Mouse_tata',
    )
    organisms_properties_features = dict()
    for organism_name in organisms_names:
        # organism_feature_folder = os.path.join(interim_folder, organism_name)
        organism_feature_folder = os.path.join(interim_folder, kmer_type, f'{organism_name}-original')
        props_dataframes = get_organism_props_dataframes(organism_path=organism_feature_folder)
        organisms_properties_features[organism_name] = props_dataframes
        print('=' * 100)
        print(props_dataframes)
        print(type(props_dataframes))
        print(f'Organism: {organism_name} | Properties dataframes: {len(props_dataframes)}')

    print()




    # # Prepare dataset object
    # pos_fasta = './data/raw-data/fasta/Bacillus_pos.fa'
    # neg_fasta = './data/raw-data/fasta/Bacillus_neg.fa'
    #
    # features_manager = dm.FeaturesManager(fasta_paths=(pos_fasta, neg_fasta))
    # features_manager.transform_raw_dataset(args.data)
    # features_manager.setup_partitions(n_splits=10)
    #
    # all_scores = list()
    # i = 0
    # # mlflow.sklearn.autolog(registered_model_name='GradientBoostingClassifier')
    # for (X_train, X_test), (y_train, y_test) in features_manager.get_next_split():
    #     ic(f'Split: {(i := i + 1)}')
    #
    #     # # Log number of samples per class
    #     # ic(np.unique(y_train, return_counts=True),
    #     #    np.unique(y_test, return_counts=True))
    #
    #     # Instantiate and fit the Classifier
    #     clf = RandomForestClassifier(max_depth=2,
    #                                  random_state=0)
    #     _params = {
    #         'n_estimators': 100,
    #         'learning_rate': 1.0,
    #         'max_depth': 1,
    #         'random_state': 0
    #     }
    #     clf = GradientBoostingClassifier(**_params)
    #     clf.fit(X_train[0], y_train)
    #
    #     # Make predictions for the test set
    #     y_pred_test = clf.predict(X_test[0])
    #     pred_probs = clf.predict_proba(X_test[0])[:, 1]
    #
    #     fp_rate, tp_rate, threshold1 = roc_curve(y_test, pred_probs)
    #     print('roc_auc_score: ', roc_auc_score(y_test, pred_probs))
    #
    #     # Calculate accuracy score
    #     _score = accuracy_score(y_test, y_pred_test)
    #     all_scores.append(_score)
    #
    #     # View stats
    #     ic(_score)
    #     # View the classification report for test data and predictions
    #     report = classification_report(y_test, y_pred_test)
    #     ic(report)
    #
    #     # # View ROC curve
    #     # plot_roc(fp_rate, tp_rate)
    #     # # View confusion matrix for test data and predictions
    #     # plot_confusion(y_test, y_pred_test)
    #
    # # ic(np.mean(all_scores), np.std(all_scores))
    #
    # cv_scores_df = pd.Series(all_scores)
    # ic(cv_scores_df, cv_scores_df.mean(), cv_scores_df.std())
    # ic(cv_scores_df.describe())
    #
    # EXP_NAME = "xboost-exp"
    # mlflow.set_experiment(EXP_NAME)
    #
    # with mlflow.start_run():
    #
    #     # Log parameters to remote MLFlow
    #     mlflow.log_params(
    #         {"model_name": "GradientBoostingClassifier"} | _params
    #     )
    #
    #     # Log metrics to remote MLFlow
    #     mlflow.log_metrics({"accuracy_mean": cv_scores_df.mean(), })
    #
    #     # Save models if it not exists yet
    #     model_path = os.path.join(os.getcwd(), 'models', 'models-GradientBoostingClassifier')
    #     if not os.path.isdir(model_path):
    #         mlflow.sklearn.save_model(clf, model_path)
    #
    #     # Plot boxplot - CV Acc scores
    #     plt.Figure()
    #     sns.boxplot(cv_scores_df, color=sns.color_palette("Set2")[0])
    #     # plt.show()
    #     fig_path = os.path.join(os.getcwd(), 'reports', f'{EXP_NAME}-cv-boxplot.svg') # Define local artifact path
    #     plt.savefig(fig_path)   # Save local artifact
    #     mlflow.log_artifact(fig_path) # Log artifact to remote MLFlow
    #
    # # joined: se.Dataset = se.MergedEncodedDataset(features_manager.datasets)
    # # print(joined)
    # # for i in joined.encoded_datasets:
    # #     print(i.shape)

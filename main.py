# -*- coding: utf-8 -*-

# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import matplotlib
import matplotlib.pyplot as plt
from argparse import Namespace
from src.utils.arguments import get_args
from icecream import ic as ic
import seaborn as sns
import pandas as pd
import joblib

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.ensemble import RandomForestClassifier

from src.datamanager import dataset_manager as dm

import mlflow
import mlflow.sklearn

matplotlib.use('TkAgg')
matplotlib.use('Qt5Agg')


def main() -> None:
    print(f'START MAIN FUNCTION')
    args: Namespace = get_args()

    # Prepare dataset object
    pos_fasta = './data/raw-data/fasta/Bacillus_pos.fa'
    neg_fasta = './data/raw-data/fasta/Bacillus_neg.fa'
    data_manager = dm.DatasetManager(fasta_paths=(pos_fasta, neg_fasta))
    data_manager.transform_raw_dataset(args.data)
    data_manager.setup_partitions(n_splits=10)

    all_scores = list()
    i = 0
    # mlflow.sklearn.autolog(registered_model_name='GradientBoostingClassifier')
    for (X_train, X_test), (y_train, y_test) in data_manager.get_next_split():
        ic(f'Split: {(i := i + 1)}')

        # # Log number of samples per class
        # ic(np.unique(y_train, return_counts=True),
        #    np.unique(y_test, return_counts=True))

        # Instantiate and fit the Classifier
        clf = RandomForestClassifier(max_depth=2,
                                     random_state=0)
        _params = {
            'n_estimators': 100,
            'learning_rate': 1.0,
            'max_depth': 1,
            'random_state': 0
        }
        clf = GradientBoostingClassifier(**_params)
        clf.fit(X_train[0], y_train)

        # Make predictions for the test set
        y_pred_test = clf.predict(X_test[0])
        pred_probs = clf.predict_proba(X_test[0])[:, 1]

        fp_rate, tp_rate, threshold1 = roc_curve(y_test, pred_probs)
        print('roc_auc_score: ', roc_auc_score(y_test, pred_probs))

        # Calculate accuracy score
        _score = accuracy_score(y_test, y_pred_test)
        all_scores.append(_score)

        # View stats
        ic(_score)
        # View the classification report for test data and predictions
        report = classification_report(y_test, y_pred_test)
        ic(report)

        # # View ROC curve
        # plot_roc(fp_rate, tp_rate)
        # # View confusion matrix for test data and predictions
        # plot_confusion(y_test, y_pred_test)

    # ic(np.mean(all_scores), np.std(all_scores))

    cv_scores_df = pd.Series(all_scores)
    ic(cv_scores_df, cv_scores_df.mean(), cv_scores_df.std())
    ic(cv_scores_df.describe())

    EXP_NAME = "xboost-exp"
    mlflow.set_experiment(EXP_NAME)

    with mlflow.start_run():

        # Log parameters to remote MLFlow
        mlflow.log_params(
            {"model_name": "GradientBoostingClassifier"} | _params
        )

        # Log metrics to remote MLFlow
        mlflow.log_metrics({"accuracy_mean": cv_scores_df.mean(), })

        # Save model if it not exists yet
        model_path = os.path.join(os.getcwd(), 'models', 'model-GradientBoostingClassifier')
        if not os.path.isdir(model_path):
            mlflow.sklearn.save_model(clf, model_path)

        # Plot boxplot - CV Acc scores
        plt.Figure()
        sns.boxplot(cv_scores_df, color=sns.color_palette("Set2")[0])
        # plt.show()
        fig_path = os.path.join(os.getcwd(), 'reports', f'{EXP_NAME}-cv-boxplot.svg') # Define local artifact path
        plt.savefig(fig_path)   # Save local artifact
        mlflow.log_artifact(fig_path) # Log artifact to remote MLFlow

    # joined: se.Dataset = se.MergedEncodedDataset(data_manager.datasets)
    # print(joined)
    # for i in joined.encoded_datasets:
    #     print(i.shape)


if __name__ == '__main__':
    main()

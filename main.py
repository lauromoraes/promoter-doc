# -*- coding: utf-8 -*-

# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib
import matplotlib.pyplot as plt
from argparse import Namespace
from utils.arguments import get_args
from icecream import ic as ic
from sklearn.manifold import TSNE
from numpy import reshape
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_curve, roc_auc_score


from sklearn.ensemble import RandomForestClassifier

import statsmodels.api as sm


from promoter_dataset import dataset_manager as dm
from promoter_dataset import seq_encoders as se

# import mlflow
import dvc

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
    for (X_train, X_test), (y_train, y_test) in data_manager.get_next_split():
        ic(f'Split: {(i := i+1)}')

        # # Log number of samples per class
        # ic(np.unique(y_train, return_counts=True),
        #    np.unique(y_test, return_counts=True))

        # Instantiate and fit the Classifier
        clf = RandomForestClassifier(max_depth=2,
                                     random_state=0)
        clf = GradientBoostingClassifier(n_estimators=100,
                                         learning_rate=1.0,
                                         max_depth=1,
                                         random_state=0)
        clf.fit(X_train[0], y_train)

        # Make predictions for the test set
        y_pred_test = clf.predict(X_test[0])
        pred_probs = clf.predict_proba(X_test[0])[:,1]

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

    df = pd.Series(all_scores)
    ic(df, df.mean(), df.std())
    ic(df.describe())

    plt.Figure()
    sns.boxplot(df, color=sns.color_palette("Set2")[0])
    plt.show()


    # joined: se.Dataset = se.MergedEncodedDataset(data_manager.datasets)
    # print(joined)
    # for i in joined.encoded_datasets:
    #     print(i.shape)





if __name__ == '__main__':
    main()

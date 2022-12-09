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

import statsmodels.api as sm


from promoter_dataset import dataset_manager as dm
from promoter_dataset import seq_encoders as se

matplotlib.use('TkAgg')
matplotlib.use('Qt5Agg')
# ic.disable()

def main() -> None:

    ic(f'START MAIN FUNCTION')
    args: Namespace = get_args()

    # Prepare dataset object
    pos_fasta = './data/raw-fasta/Bacillus_pos.fa'
    neg_fasta = './data/raw-fasta/Bacillus_neg.fa'
    data_manager = dm.DatasetManager(fasta_paths=(pos_fasta, neg_fasta))
    data_manager.transform_raw_dataset(args.data)
    data_manager.setup_partitions(n_splits=5)
    for X, y in data_manager.get_next_split():
        ic(X)
        ic(y)

    # joined: se.Dataset = se.MergedEncodedDataset(data_manager.datasets)
    # print(joined)
    # for i in joined.encoded_datasets:
    #     print(i.shape)





if __name__ == '__main__':
    main()

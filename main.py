# -*- coding: utf-8 -*-

# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib
import matplotlib.pyplot as plt
from argparse import Namespace
from utils.arguments import get_args
from promoter_dataset import dataset_manager as dm
from icecream import ic as ic
from sklearn.manifold import TSNE
from numpy import reshape
import seaborn as sns
import pandas as pd
import numpy as np


from sklearn.ensemble import RandomForestClassifier

import statsmodels.api as sm

matplotlib.use('TkAgg')
matplotlib.use('Qt5Agg')
# ic.disable()

def main() -> None:

    print(f'START MAIN FUNCTION')
    args: Namespace = get_args()
    print(args)

    # Prepare dataset object
    pos_fasta = './data/raw-fasta/Bacillus_pos.fa'
    neg_fasta = './data/raw-fasta/Bacillus_neg.fa'
    data_manager = dm.DatasetManager(fasta_paths=(pos_fasta, neg_fasta))
    data_manager.setup_datasets(args.data)






if __name__ == '__main__':
    main()

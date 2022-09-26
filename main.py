# -*- coding: utf-8 -*-

# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from argparse import Namespace
from utils.arguments import get_args

from promoter_dataset import dataset_manager as dm

def main() -> None:
    print(f'START MAIN FUNCTION')
    args: Namespace = get_args()
    print(args)

    pos_fasta = './data/raw-fasta/Bacillus_pos.fa'
    neg_fasta = './data/raw-fasta/Bacillus_neg.fa'
    slice = (61,5,5)
    data_manager = dm.DatasetManager(fasta_paths=(pos_fasta, neg_fasta))
    data_manager.prepare_fasta_sequences()
    data_manager.encode_datasets(encoder_type='label', slice=(61,5,5), k=1, step=1)


if __name__ == '__main__':
    main()

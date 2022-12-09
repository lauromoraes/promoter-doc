import re
import numpy as np
from Bio import SeqIO
from sklearn.model_selection import StratifiedKFold

from promoter_dataset.seq_encoders import *

from icecream import ic as ic

# ic.disable()

SEED = 17


class DatasetManager(object):

    def __init__(self, fasta_paths: tuple[str] = None):
        self.fasta_paths = fasta_paths
        self.raw_dataset_manager: RawDatasetManager = RawDatasetManager(fasta_paths=fasta_paths)
        self.raw_dataset_manager.prepare_fasta_sequences()
        self.datasets: list[EncodedDataset] = None
        self.partitions = None
        self.X = None
        self.y = None

    def transform_raw_dataset(self, params: list[dict]) -> list[EncodedDataset]:
        """ Iterate over each parameter set and process raw data, transforming it into new encoded and
        sliced feature dataset.

        :param params: List of parameters to configure the data transforming.
        """

        def transform_dataset(p: dict) -> EncodedDataset:
            # Extract parameters from dict
            k = p['k']
            encode_type = p['encode']
            slice_positions = p['slice']
            # Encode single dataset
            d = self.raw_dataset_manager.encode_datasets(
                encoder_type=encode_type,
                slice=slice_positions,
                k=k,
                step=1,
                verbose=True)

            return d

        # Apply each encoding function to corresponding datasets
        datasets: list[EncodedDataset] = [transform_dataset(p) for p in params]
        # Set attribute
        self.datasets = datasets

        return datasets

    def setup_partitions(self, n_splits, verbose: bool = True) -> StratifiedKFold:
        """ Setup object partitions index and iterators.

        :param n_splits: The number of partitions to set.
        :param verbose:
        :return: sklearn partition object
        """
        ic.enable() if verbose else ic.disable()
        # Setup partition object from sklearn
        self.partitions = StratifiedKFold(n_splits=n_splits, random_state=SEED, shuffle=True)
        # Setup X with all datasets samples
        X = [np.concatenate(d.encoded_classes_datasets, axis=0) for d in self.datasets]
        # Setup y with class labels
        _tmp = self.datasets[0].encoded_classes_datasets
        y = np.concatenate([np.full(c.shape[0], fill_value=i) for i, c in enumerate(_tmp)], axis=0)

        # Set attributes values
        self.X = X
        self.y = y

        # Create partitions index and iterators
        self.partitions.get_n_splits(self.X[0], self.y)

        ic.disable() if verbose else None
        return self.partitions


    def get_next_split(self, verbose: bool = True):
        ic.enable() if verbose else ic.disable()
        for i, (train_i, test_i) in enumerate(self.partitions.split(self.X[0], self.y)):
            ic('Split', i)
            # Setup X
            _X_train = [x[train_i] for x in self.X]
            _X_test = [x[test_i] for x in self.X]
            _X = (_X_train, _X_test)

            # Setup y
            _y_train = self.y[train_i]
            _y_test = self.y[test_i]
            _y = (_y_train, _y_test)
            ic(np.unique(_y[0], return_counts=True), np.unique(_y[1], return_counts=True))
            yield _X, _y

class RawDatasetManager(object):
    """ DatasetManager is responsible for manager a single nucleotide dataset.
    It applies a specific encoding function on each of the problem class datasets.
    """

    def __init__(self, fasta_paths: tuple[str] = None) -> None:
        self.fasta_paths = fasta_paths
        self.raw_datasets = list()

    def prepare_fasta_sequences(self, discard_invalids: bool = True) -> list[list[str]]:
        """ Read all FASTA files in a list of paths and extract characters' sequences for each.

        :param discard_invalids: Inform whether to discard sequences with invalid nucleotide character. Not in
        (A, T, G, C).
        :return: A list of datasets (problem class) compounded of a list of strings (each nucleotide sequence).
        """
        fasta_sequences = list()
        for fasta in self.fasta_paths:
            sequences = self.get_fasta_sequences(fasta, discard_invalids)
            fasta_sequences.append(sequences)
        self.raw_datasets = fasta_sequences
        return fasta_sequences

    def get_fasta_sequences(self, fasta_path: str, discard_invalids: bool = True) -> list[
        str]:
        """ Read all nucleotide sequences in a FASTA file and transform them into a list of strings.

        :param fasta_path: A FASTA file path.
        :param discard_invalids: Inform whether to discard sequences with invalid nucleotide character. Not in
        (A, T, G, C).
        :return: A list of strings as sequences of nucleotides.
        """
        sequences = list()
        for record in SeqIO.parse(fasta_path, 'fasta'):
            seq = str(record.seq.upper())
            if discard_invalids and re.search(r'[^ATGC]', seq):
                continue
            sequences.append(seq)
        return sequences

    def encode_datasets(self, encoder_type: str, slice: tuple[int], k: int = 1, step: int = 1, verbose=False) -> \
            EncodedDataset:
        """ Transform the sequences of all raw datasets using a specific encoding type.

        :param encoder_type: Defines the encoding type.
        :param slice:
        :param k:
        :param step:
        :param verbose:
        :return: EncodedDataset
        """

        ic.enable() if verbose else ic.disable()

        args = {
            'raw_datasets': self.raw_datasets,
            'k': k,
            'step': step,
            'encode_type': encoder_type,
            'slice': slice
        }
        # ic(args)

        if encoder_type == 'label':
            encoded = IntegerSeqEncoder(**args)
        elif encoder_type == 'onehot':
            encoded = OneHotSeqEncoder(**args)
        elif encoder_type == 'prop':
            encoded = PropertyEncoder(**args)
        else:
            print(f'Invalid encode type: {encoder_type}.')
        # ic(encoded)

        ic.disable() if verbose else None
        return encoded

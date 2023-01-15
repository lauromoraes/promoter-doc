import os
import re
import abc
import pandas as pd
import numpy as np
from numpy import array
from itertools import product
from Bio import SeqIO
from collections.abc import Iterable

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from icecream import ic as ic


# ic.disable()

class Dataset(object):
    def __init__(self):
        pass


class EncodedDataset(Dataset):
    """This class is responsible to process nucleotide datasets and transform them into encoded datasets.

    Attributes:
        raw_datasets (tuple[list[str]]): A tuple nucleotides datasets. Each one will be a list of sequences as
            strings.
        k (int, optional): This argument defines the size of processed K-mers.
        encode_type (str, optional): A string defining the encoding type for this sequence dataset.
        slice (tuple[int], optional): A tuple as (tss_position, upstream, downstream) defining a
            subsequence surrounding the TSS position.
        discard_invalids (bool, optional): Defines if only standard sequences will be accept (only contains A, T, G, C).

    """

    def __init__(self, raw_datasets: tuple[list[str]], k: int = 1, step: int = 1, encode_type: str = 'label',
                 slice: tuple[int] = None, discard_invalids: bool = True) -> None:
        """__init__ method
            Args:
        raw_datasets (tuple[list[str]]): A tuple nucleotides datasets. Each one will be a list of sequences as
            strings.
        k (int, optional): This argument defines the size of processed K-mers. Default is k=1.
        encode_type (str, optional): A string defining the encoding type for this sequence dataset. Default is
            the `label` encoding.
        slice (tuple[int], optional): A tuple as (tss_position, upstream, downstream) defining a
            subsequence surrounding the TSS position. If not provided, take all sequence.
        discard_invalids (bool, optional): Defines if only standard sequences will be accept (only contains A, T, G, C).
        """
        super(EncodedDataset, self).__init__()
        self.raw_datasets = raw_datasets
        self.k = k
        self.step = step
        self.encode_type = encode_type
        self.slice = slice
        self.discard_invalids = discard_invalids
        self.encoded_classes_datasets = self.fit_transform()

    def get_k_mers(self, sequence: str, k: int, step: int):
        n = len(sequence) - k + 1
        k_mers = [sequence[i:i + k] for i in range(0, n, step)]
        return k_mers

    def get_k_mers_encoder(self, k: int):
        nucleotides = ['G', 'A', 'C', 'T']
        tups = list(product(nucleotides, repeat=k))
        data = array([''.join(x) for x in tups])
        label_encoder = LabelEncoder()
        label_encoder.fit(data)
        return label_encoder

    @abc.abstractmethod
    def transform_sequences(self, sequences: list[str], k: int = 1, step: int = 1):
        pass

    def fit_transform(self):
        datasets = list()
        if self.slice:
            upstream = self.slice[0] - self.slice[1] - 1
            downstream = self.slice[0] + self.slice[2]
        for dataset in self.raw_datasets:
            if self.slice:
                dataset = [s[upstream:downstream] for s in dataset]
                dataset = self.transform_sequences(sequences=dataset, k=self.k, step=self.step)
            datasets.append(dataset)
        return datasets


class MergedEncodedDataset(Dataset):
    def __init__(self, original_datasets: list[EncodedDataset], type: str = 'horizontal'):
        super(MergedEncodedDataset, self).__init__()
        print('Inside MergedEncodedDataset')
        self.original_datasets: list[EncodedDataset] = original_datasets
        self.type = type
        self.encoded_datasets = None
        self.combine_datasets()

    def combine_datasets(self, _datasets: list[EncodedDataset] = None):
        datasets = self.original_datasets if not _datasets else _datasets
        n_encode_types = len(datasets)
        if n_encode_types == 1:
            return datasets[0]
        elif n_encode_types <= 1:
            raise Exception
        n_class = len(datasets[0].encoded_classes_datasets)
        _axis = 0 if self.type == 'vertical' else 1
        new_encoded_datasets = list()
        for c in range(n_class):
            _datasets = [d.encoded_classes_datasets[c] for d in datasets]
            joined = np.concatenate(_datasets, axis=_axis)
            new_encoded_datasets.append(joined)
        self.encoded_datasets = new_encoded_datasets
        return new_encoded_datasets


class IntegerSeqEncoder(EncodedDataset):
    def __init__(self, raw_datasets: tuple[list[str]], k: int = 1, step: int = 1, encode_type: str = 'label',
                 slice: tuple[int] = None, discard_invalids: bool = True) -> None:
        super(IntegerSeqEncoder, self).__init__(raw_datasets, k, step, encode_type, slice, discard_invalids)

    def transform_sequences(self, sequences: list[str], k: int = 1, step: int = 1):
        sequences_mers = [self.get_k_mers(s, k, step) for s in sequences]
        label_encoder = self.get_k_mers_encoder(k)
        labeled_sequences = np.array([label_encoder.transform(s) for s in sequences_mers])
        return labeled_sequences


class OneHotSeqEncoder(EncodedDataset):
    def __init__(self, raw_datasets: tuple[list[str]], k: int = 1, step: int = 1, encode_type: str = 'label',
                 slice: tuple[int] = None, discard_invalids: bool = True) -> None:
        """

        :param raw_datasets:
        :param k:
        :param step:
        :param encode_type:
        :param slice:
        :param discard_invalids:
        """
        super(OneHotSeqEncoder, self).__init__(raw_datasets, k, step, encode_type, slice, discard_invalids)

    def transform_sequences(self, sequences: list[str], k: int = 1, step: int = 1):
        from tensorflow.keras.utils import to_categorical

        def encode(seq):
            return to_categorical(seq, num_classes=4 ** k, dtype='int32').T

        sequences_mers = [self.get_k_mers(s, k, step) for s in sequences]
        label_encoder = self.get_k_mers_encoder(k)
        encoded_sequences = np.array([label_encoder.transform(s) for s in sequences_mers])
        encoded_sequences = np.array([encode(x) for x in encoded_sequences])
        return encoded_sequences


class PropertyEncoder(EncodedDataset):
    # TODO: implement PropertyEncoder Class
    def __init__(self, raw_datasets: tuple[list[str]], k: int = 1, step: int = 1, encode_type: str = 'label',
                 slice: tuple[int] = None, discard_invalids: bool = True) -> None:
        """

        :param raw_datasets:
        :param k:
        :param step:
        :param encode_type:
        :param slice:
        :param discard_invalids:
        """
        super(PropertyEncoder, self).__init__(raw_datasets, k, step, encode_type, slice, discard_invalids)

    def get_prop_data(self, k: int = None):
        if k == 2:
            file_path = 'dinuc'
        elif k == 3:
            file_path = 'trinuc'
        else:
            raise ValueError(f'k value must be 2 or 3, received {k}')
        file_path = os.path.join(os.getcwd(), '..', 'data', 'raw-data', 'physicochemical-properties-reference',
                                 f'{file_path}.tsv')
        prop_data = pd.read_csv(file_path, sep='\t', index_col=0)
        self.prop_index = prop_data.index
        return prop_data

    def transform_sequences(self, sequences: list[str], k: int = 1, step: int = 1):
        """Receive a single dataset of nucelotides sequences and transform it into
        a dataset of sequences of

        :param sequences:
        :param k:
        :param step:
        :return:
        """
        sequences_mers = [self.get_k_mers(s, k, step) for s in sequences]
        prop_data = self.get_prop_data(k)
        prop_idx = prop_data.index
        encoded = np.array([np.array([prop_data[mer] for mer in sequence]) for sequence in sequences_mers])
        encoded = encoded.swapaxes(0, 2)
        encoded = encoded.swapaxes(1, 2)
        return encoded

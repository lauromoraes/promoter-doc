import re
import abc
import numpy as np
from numpy import array
from itertools import product
from Bio import SeqIO

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


class EncodedDataset(object):
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
        self.raw_datasets = raw_datasets
        self.k = k
        self.step = step
        self.encode_type = encode_type
        self.slice = slice
        self.discard_invalids = discard_invalids
        self.encoded_datasets = self.fit_transform()

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


class IntegerSeqEncoder(EncodedDataset):
    def __init__(self):
        super(IntegerSeqEncoder, self).__init__()

    def transform_sequences(self, sequences: list[str], k: int = 1, step: int = 1):
        sequences_mers = [self.get_k_mers(s, k, step) for s in sequences]
        label_encoder = self.get_k_mers_encoder(k)
        labeled_sequences = [label_encoder.transform(s) for s in sequences_mers]
        return labeled_sequences


class OneHotSeqEncoder(object):
    #TODO: criar nova classe de encoder
    def __init__(self):
        super(OneHotSeqEncoder, self).__init__()

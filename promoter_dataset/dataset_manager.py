import re
import numpy as np
from numpy import array
from itertools import product
from Bio import SeqIO

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


class IntegerEncoder(object):
    def __init__(self):
        pass

    def get_k_mers(self, sequence: str, k:int, step: int):
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

    def fit_transform(self, sequences: list[str], k: int = 1, step: int = 1):
        sequences_mers = [self.get_k_mers(s, k, step) for s in sequences]
        label_encoder = self.get_k_mers_encoder(k)
        labeled_sequences = [label_encoder.transform(s) for s in sequences_mers]
        return labeled_sequences


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
        self.label_encoder = IntegerEncoder()
        self.label_datasets = self.process_raw_datasets()

    def process_raw_datasets(self):
        datasets = list()
        if self.slice:
            upstream = self.slice[0] - self.slice[1] - 1
            downstream = self.slice[0] + self.slice[2]
        for dataset in self.raw_datasets:
            if self.slice:
                dataset = [s[upstream:downstream] for s in dataset]
                dataset = self.label_encoder.fit_transform(sequences=dataset, k=self.k, step=self.step)
            datasets.append(dataset)
        return datasets


class DatasetManager(object):
    '''DatasetManager

    '''

    def __init__(self, fasta_paths: tuple[str] = None) -> None:
        self.fasta_paths = fasta_paths
        self.raw_datasets = list()

    def prepare_fasta_sequences(self, discard_invalids: bool = True):
        fasta_sequences = list()
        for fasta in self.fasta_paths:
            sequences = self.get_fasta_sequences(fasta, discard_invalids)
            fasta_sequences.append(sequences)
        self.raw_datasets = fasta_sequences
        return fasta_sequences

    def get_fasta_sequences(self, fasta_path: str, discard_invalids: bool = True) -> list[
        str]:
        """Read all sequences in a FASTA file, slice it all and make a list with them.

        :param fasta_path: FASTA file path.
        :param discard_invalids: To discard sequences with non-valid nucleotide characters (A, T, G, C).
        :return: List with all sliced sequences as strings.
        """
        sequences = list()
        for record in SeqIO.parse(fasta_path, 'fasta'):
            seq = str(record.seq.upper())
            if discard_invalids and re.search(r'[^ATGC]', seq):
                continue
            sequences.append(seq)
        return sequences

    def encode_datasets(self, encoder_type: str, slice: tuple[int], k: int = 1, step: int = 1):
        label_encoded = EncodedDataset(raw_datasets=self.raw_datasets, k=k, step=step, encode_type=encoder_type,
                                       slice=slice)
        for d in label_encoded.label_datasets:
            print(d)

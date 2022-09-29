import re
import numpy as np
from Bio import SeqIO

from promoter_dataset.seq_encoders import *

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
        """Transform the sequences of all raw datasets using a specific encoding type.

        :param encoder_type: Defines the encoding type.
        :param slice:
        :param k:
        :param step:
        :return:
        """
        if encoder_type == 'label':
            encoded = EncodedDataset(raw_datasets=self.raw_datasets, k=k, step=step, encode_type=encoder_type,
                                           slice=slice)
        elif encoder_type == 'onehot':
            encoded = None
        else:
            print(f'Invalid encode type: {encoder_type}.')
        for d in encoded.encoded_datasets:
            print(d)

        return encoded

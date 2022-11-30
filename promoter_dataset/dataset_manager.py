import re
import numpy as np
from Bio import SeqIO

from promoter_dataset.seq_encoders import *

from icecream import ic as ic
# ic.disable()

class DatasetManager(object):

    def __init__(self, fasta_paths: tuple[str] = None):
        self.fasta_paths = fasta_paths
        self.raw_dataset_manager: RawDatasetManager = RawDatasetManager(fasta_paths=fasta_paths)
        self.raw_dataset_manager.prepare_fasta_sequences()
        self.datasets: list[RawDatasetManager] = list()

    def setup_datasets(self, params: list[dict]):
        for dataset_param in params:
            k = dataset_param['k']
            encode_type = dataset_param['encode']
            slice_positions = dataset_param['slice']
            # step = dataset_param['step']
            self.raw_dataset_manager.encode_datasets(encoder_type=encode_type, slice=slice_positions, k=k, step=1,
                                                     verbose=True)


class RawDatasetManager(object):
    ''' DatasetManager is responsible for manager a single nucleotide dataset.
    It applies a specific encoding function on each of the problem class datasets.
    '''

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
        ic(args)

        if encoder_type == 'label':
            encoded = IntegerSeqEncoder(**args)
        elif encoder_type == 'onehot':
            encoded = OneHotSeqEncoder(**args)
        elif encoder_type == 'prop':
            encoded = PropertyEncoder(**args)
        else:
            print(f'Invalid encode type: {encoder_type}.')
        ic(encoded)

        ic.disable() if verbose else None
        return encoded

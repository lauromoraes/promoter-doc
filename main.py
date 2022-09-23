# -*- coding: utf-8 -*-

# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from argparse import Namespace
from utils.arguments import get_args

# ==========================
import re
from Bio import SeqIO
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

class SequenceEncoder(object):
    def __init__(self):
        pass

    def onehot_encode(self, k: int = 1):
        pass

class PromoterDataset(object):
    def __init__(self, fasta_paths: tuple[str], k: int, encode_type: str, slice: tuple[int] = None, discard_invalids:
    bool = True) -> None:
        self.fasta_paths = fasta_paths
        self.k = k
        self.encode_type = encode_type
        self.slice = slice
        self.discard_invalids = discard_invalids

    def get_fasta_sequences(self, fasta_path: str)-> list[str]:
        """Read all sequences in a FASTA file, filter and slice them.

        :param fasta_path: FASTA file path.
        :return: List with all sliced sequences as strings
        """
        sequences = list()
        if self.slice:
            downstream = self.slice[0]-self.slice[1]-1
            upstream = self.slice[0]-self.slice[2]
        for record in SeqIO.parse(fasta_path, 'fasta'):
            seq = str(record.seq).upper()
            if self.discard_invalids and re.search(r'[^ATGC]', seq):
                continue
            if self.slice:
                seq = seq[downstream:upstream]
            sequences.append(seq)
        return sequences

class DatasetManager(object):
    def __init__(self, fasta_paths: tuple[str] = None) -> None:
        self.fasta_paths = fasta_paths

    def prepare_fasta_sequences(self, slice: tuple[int] = None, discard_invalids: bool = True):
        fasta_sequences = list()
        for fasta in self.fasta_paths:
            sequences = self.get_fasta_sequences(fasta, slice, discard_invalids)
            fasta_sequences.append(sequences)
        return fasta_sequences

    def get_fasta_sequences(self, fasta_path: str, slice: tuple[int] = None, discard_invalids: bool = True) -> list[
        str]:
        """Read all sequences in a FASTA file, slice it all and make a list with them.

        :param fasta_path: FASTA file path.
        :param slice: Tuple with slice references (TSS position, Upstream bases, Downstream bases) - index positions in
        a sequence, not in a array.
        :param discard_invalids: To discard sequences with non-valid nucleotide characters (A, T, G, C).
        :return: List with all sliced sequences as strings.
        """
        sequences = list()
        for record in SeqIO.parse(fasta_path, 'fasta'):
            seq = str(record.seq.upper())
            if discard_invalids and re.search(r'[^ATGC]', seq):
                continue
            if slice:
                # slice = (tss_position, upstream, downstream)
                seq = seq[(slice[0]-slice[1]-1):(slice[0]+slice[2])]
            sequences.append(seq)
        return sequences
# ==========================

def main() -> None:

    print(f'START MAIN FUNCTION')
    args: Namespace = get_args()
    print(args)

    pos_fasta = './data/raw-fasta/Bacillus_pos.fa'
    neg_fasta = './data/raw-fasta/Bacillus_neg.fa'

    prom_data = DatasetManager(fasta_paths=(pos_fasta, neg_fasta))
    seqs = prom_data.get_fasta_sequences(fasta_path=prom_data.fasta_paths[0], slice=(61, 60, 20))
    print(f'{len(seqs)}')
    print(f'{len(seqs[0])} - {seqs[0]}')


if __name__ == '__main__':
    main()

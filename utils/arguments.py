# -*- coding: utf-8 -*-

import os
import yaml
from argparse import ArgumentParser
from argparse import Namespace
import datetime


def get_args_yaml(yaml_path: str) -> dict:
    """ Parse the YAML file to a Dictionary

    :param yaml_path: Path of YAML file.
    :return: YAML parsed to a Python Dictionary
    """
    with open(yaml_path) as f:
        doc = yaml.load(f, Loader=yaml.FullLoader)
        return doc


def get_args_parser() -> ArgumentParser:
    """Configure and get the command line argument parser

    :return: Argument Parser
    """
    parser = ArgumentParser()
    parser.add_argument('-y', '--yaml', help='The YAML file with experiment configuration.', required=True)
    return parser


def get_samples_seeds(n_seeds: int) -> list[int]:
    """Get prime seeds

    :param n_seeds: The number of seeds to return
    :return: A list with prime numbers
    """
    prime_numbers = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
                     101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
                     199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313,
                     317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439,
                     443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571,
                     577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691,
                     701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829,
                     839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977,
                     983, 991, 997, 1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091,
                     1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213,
                     1217, 1223]
    return prime_numbers[:n_seeds]


def get_args() -> Namespace:
    """Configure all experiment arguments

    :return: Namespace object with all experiment arguments and values
    """
    args = get_args_parser().parse_args()
    yaml_args = get_args_yaml(args.yaml)

    args.experiment = os.path.splitext(os.path.basename(args.yaml))[0]  # Set experiment name as configuration file
    args.ts = datetime.datetime.now().strftime("%Y_%m_%d=%H_%M_%S")
    for k, v in yaml_args.items():
        setattr(args, k, v)
    args.samples_seeds = get_samples_seeds(args.n_samples)

    return args

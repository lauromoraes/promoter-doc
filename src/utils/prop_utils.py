import pandas as pd
import os
import numpy as np

# Utility functions
def get_organism_props_dataframes(organism_path: str, extension: str = 'feather'):
    # Get all properties files from organism folder
    props_files = [f'{organism_path}/{file}' for file in os.listdir(organism_path) if file.startswith('df_data_prop_')]

    # Get dataframes for each property
    props = list()
    for prop_idx in range(len(props_files)):
        prop_file = f'{organism_path}/df_data_prop_{prop_idx}.{extension}'  # Get property file path
        if os.path.exists(prop_file): # Check if property file exists
            if extension == 'feather': # Feather is faster than csv but not compatible with other libraries
                df_data = pd.read_feather(prop_file)
            elif extension == 'csv': # CSV is more compatible with other libraries
                df_data = pd.read_csv(prop_file, index_col=False)
            else:
                raise ValueError('Extension not supported.')
            props.append(df_data) # Append dataframe to list
        else:
            print(f'Property {prop_idx} {extension} file not found.')

    return props

def get_bp_positions(seq_len: int, step: int = 10):
    # Set range of bp positions (upstream, TSS, downstream)
    if seq_len == 80:
        _range = np.arange(-60, 20, step)
    elif seq_len == 79:
        _range = np.arange(-60, 19, step)
    elif seq_len == 250:
        _range = np.arange(-200, 50, step)
    elif seq_len == 249:
        _range = np.arange(-200, 49, step)
    else:
        raise ValueError('Sequence length not supported.')
    return _range

def get_props_names(kmer_type: str = 'dinuc'):
    kmers_values_folder = os.path.join(os.getcwd(), os.pardir, 'data', 'raw-data',
                                       'physicochemical-properties-reference',
                                           f'original-{kmer_type}.tsv')
    kmer_df = pd.read_csv(kmers_values_folder, sep='\t', index_col=0)
    props_names = kmer_df.index.tolist()
    return props_names
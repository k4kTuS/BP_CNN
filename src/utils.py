import pandas as pd
import tensorflow as tf

DATASET_PATHS = '../MURA-v1.1/detailed_paths.csv'


def get_dataframe(body_part, split):
    if body_part == 'ALL':
        body_part = '.*'
    if split == 'ALL':
        split = '.*'

    df = pd.read_csv(DATASET_PATHS)
    filtered_df = df[(df.body_part.str.match(body_part, case=False)) & (df.split.str.match(split, case=False))]
    return filtered_df


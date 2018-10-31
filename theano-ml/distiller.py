import os
import sys
import pickle
import gzip

import numpy as np
import pandas as pd

from sklearn import preprocessing


def data_distiller(sample_data):
    # read csv file and shuffle
    df = pd.read_csv(sample_data)
    df = df.sample(frac=1.)
    df = df.reset_index(drop=True)

    n_rows = df.shape[0]

    # column names
    df.columns = [
        'instance_id', 'chunk_id', 'brake', 'gas', 'velocity',
        'steering', 'lon_acc', 'lat_acc', 'image', 'category'
    ]

    # normalize data
    # df['brake'] = (df['brake']-0.046)/0.168807
    # df['gas'] = (df['gas']-3.62)/4.9
    # df['velocity'] = (df['velocity']/3.6/10-2.46)/0.208169  # m/s; 10Hz
    # df['steering'] = (df['steering']-1.057)/2.46  # degrees to rads
    # df['lon_acc'] = (df['lon_acc']-0.120877)/0.029484
    # df['lat_acc'] = (df['lat_acc']+0.028278)/0.02285
    df['category'] = df['category'] + 1  # [0, 1 ,2]
    scalar = preprocessing.StandardScaler().fit(df.loc[:, 'brake':'lat_acc'])
    scaled_data = scalar.fit_transform(df.loc[:, 'brake':'lat_acc'])
    df.loc[:, 'brake':'lat_acc'] = scaled_data

    # distill dataset
    df_new = df.loc[:, 'brake':'lat_acc'].merge(
        right=df.loc[:, ['category']],
        left_index=True,
        right_index=True
    )

    array_slices = [0, int(0.7*n_rows), int(0.9*n_rows), n_rows]

    train_data = df_new[array_slices[0]:array_slices[1]]
    valid_data = df_new[array_slices[1]:array_slices[2]]
    test_data = df_new[array_slices[2]:array_slices[3]]

    train_data = (train_data.iloc[:, :-1], train_data.iloc[:, -1])
    valid_data = (valid_data.iloc[:, :-1], valid_data.iloc[:, -1])
    test_data = (test_data.iloc[:, :-1], test_data.iloc[:, -1])

    dataset = [train_data, valid_data, test_data]

    out_file = 'dataset.pkl.gz'
    with gzip.open(out_file, 'wb') as f:
        pickle.dump(dataset, f)

    return out_file

if __name__ == "__main__":
    data_distiller('sample_data.csv')

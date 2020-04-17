import numpy as np
import pandas as pd

def removeEmptyRows(data):
    n_samples, n_days = data.shape[0], data.shape[1]
    
    rows = []
    for n in range(n_samples):
        # if first or second elements are nans remove
        if np.isnan(data[n,0]) or np.isnan(data[n,1]):
            continue
        else: rows.append(data[n].reshape(1,-1))

    data = np.vstack(rows)

    return data

import argparse
parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--data_file", type=str, default='')
parser.add_argument("--tr_out_file", type=str, default='')
parser.add_argument("--val_out_file", type=str, default='')
parser.add_argument("--test_out_file", type=str, default='')
parser.add_argument("--seed", type=int, default='')

args = parser.parse_args()

if __name__ == "__main__":

    df_data = pd.read_csv(args.data_file)
    np_data = np.asarray(df_data)
    np_data = removeEmptyRows(np_data)
    
    # shuffle
    ids = list(range(np_data.shape[0]))
    np.random.seed(args.seed)
    np.random.shuffle(ids)

    np_data = np_data[ids]
    
    # split into train,valid,test
    train_ratio = 0.8
    valid_ratio = 0.1
    test_ratio = 1 - train_ratio - valid_ratio

    n_samples = np_data.shape[0]
    train_split = int(np.ceil(n_samples * train_ratio))
    valid_split = int(np.ceil(n_samples * (train_ratio + valid_ratio)))

    train_data = np_data[:train_split]
    valid_data = np_data[train_split:valid_split]
    test_data = np_data[valid_split:]

    with open(args.tr_out_file, "wt") as fp:
        np.savetxt(fp, train_data, delimiter=',')
    with open(args.val_out_file, "wt") as fp:
        np.savetxt(fp, valid_data, delimiter=',')
    with open(args.test_out_file, "wt") as fp:
        np.savetxt(fp, test_data, delimiter=',')

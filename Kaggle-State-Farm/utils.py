import numpy as np
import pandas as pd
from config import PATH, TRAINING, TESTING, RESULT
import os.path


def train_set():
    pkl = pd.read_pickle(TRAINING)
    features = np.array(list(pkl['conv']))
    flatten = features.reshape(features.shape[0], -1)
    df = pkl[['classname', 'subject', 'path', 'img']]
    del pkl, features
    return df, flatten


def test_set():
    pkl = pd.read_pickle(TESTING)
    features = np.array(list(pkl['conv']))
    flatten = features.reshape(features.shape[0], -1)
    df = pkl[['img', 'path']]
    del pkl, features
    return df, flatten


def submit(df, filename='submission.csv', include_path=False, include_class=False):
    indexed_df = df.set_index(df['img'])
    cols = ["c" + str(i) for i in range(10)]
    ret = pd.DataFrame()
    ret[cols] = indexed_df[cols]
    if include_path:
        try:
            ret['path'] = indexed_df['path']
        except:
            print('No "path" column!')
    if include_class:
        try:
            ret['classname'] = indexed_df['classname']
        except:
            print('No "classname" column!')
    if filename is not None:
        ret.to_csv(os.path.join(RESULT, filename))
    return ret

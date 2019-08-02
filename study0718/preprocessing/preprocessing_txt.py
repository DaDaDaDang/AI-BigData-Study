import numpy as np
import sklearn
import scipy
import os
from os.path import join
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def importData() :
    abalone_path = join('./data', 'abalone.txt')
    column_path = join('./data', 'abalone_attributes.txt')

    abalone_columns = list()

    for I in open(column_path) :
        abalone_columns.append((I.strip()))

    data = pd.read_csv(abalone_path, header = None, names = abalone_columns)

    label = data['Sex']
    del data["Sex"]

    return data

def mmscaler(data) :
    mscaler = MinMaxScaler()
    mscaler.fit(data)
    mMscaled_data = mscaler.transform(data)
    mMscaled_data_f = pd.DataFrame(mMscaled_data, columns=data.columns)
    print("min_values=>", mMscaled_data.min())
    print("max_values=>", mMscaled_data.max())
    print("data.mMscaled_data_f()=>", mMscaled_data_f)
    print("data()=>")

if __name__ == '__main__' :
    data = importData()
    mmscaler(data)
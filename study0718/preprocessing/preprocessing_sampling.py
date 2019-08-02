from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import numpy
import pandas as pd
import sklearn
import scipy
from os.path import join

def importData() :
    abalone_path = join('../data', 'abalone.txt')
    column_path = join('../data', 'abalone_attributes.txt')

    abalone_columns = list()

    for I in open(column_path) :
        abalone_columns.append((I.strip()))

    data = pd.read_csv(abalone_path, header = None, names = abalone_columns)

    label = data['Sex']
    del data["Sex"]

    return data


def psampling(data) :
    label = data['Sex']
    ros = RandomOverSampler(random_state = 2019)
    rus = RandomUnderSampler(random_state = 2019)

    oversampled_data, oversampled_label = ros.fit_resample(data, label)
    oversampled_data = pd.DataFrame(oversampled_data, columns = data.columns)

    undersampled_data, undersampled_label = ros.fit_resample(data, label)
    undersampled_data = pd.DataFrame(undersampled_data, columns = data.columns)

    print('원본 데이터의 클래스 비율 \n{}'.format(pd.get_dummies(label).sum()))
    print('\nRandom Over 샘플링 결과 \n{}'.format(pd.get_dummies(oversampled_label).sum()))
    print('\nRandom Under 샘플링 결과 \n{}'.format(pd.get_dummies(undersampled_label).sum()))

if __name__ == '__main__' :
    psampling(importData())

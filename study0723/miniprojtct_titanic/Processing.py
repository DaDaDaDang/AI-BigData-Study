import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import  seaborn as sns
from miniprojtct_titanic import dataset

def processing () :
#def sexFature () :
    dataset_ = [dataset.train, dataset.test]

    for ds in dataset_ :
        ds['Sex'] = ds['Sex'].astype(str)

#def embarkedFeature () :
    dataset_ = [dataset.train, dataset.test]

    print("train.isnull().sum()", dataset.train.isnull().sum())

    dataset.train['Embarked'].value_counts(dropna=False)

    for ds in dataset_:
        ds['Embarked'] = ds['Embarked'].fillna('S')
        ds['Embarked'] = ds['Embarked'].astype(str)
    print("train.isnull().sum()", dataset.train.isnull().sum())

# def age() :
    for ds in dataset_:
        ds['Age'].fillna(ds['Age'].mean(), inplace=True)
        ds['Age'] = ds['Age'].astype(int)
        dataset.train['AgeBand'] = pd.cut(dataset.train['Age'], 5)
    print(dataset.train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean())

    for ds in dataset_:
        ds.loc[ds['Age'] <= 16, 'Age'] =0
        ds.loc[(ds['Age'] > 16)& (ds['Age']<=32), 'Age'] = 1
        ds.loc[(ds['Age'] > 32)& (ds['Age']<=48), 'Age'] = 2
        ds.loc[(ds['Age'] > 48)& (ds['Age']<=64), 'Age'] = 3
        ds.loc[ds['Age'] > 64, 'Age'] = 4
        ds['Age'] = ds['Age'].map({0: 'Child', 1: 'Young', 2: 'Middle', 3: 'Prime', 4: 'Old'}).astype(str)

        #sibsp&Parch Reature
        for ds in dataset_:
            ds['Family'] = ds['Parch'] + ds['SibSp']
            ds['Family'] = ds['Family'].astype(int)

def dataCheck() :
    features_drop = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
    dataset.train = dataset.train.drop(features_drop, axis =1)
    dataset.test = dataset.test.drop(features_drop, axis=1)

    print(dataset.train.head())
    print(dataset.test.head())
    print("train.isnull().sum()", dataset.train.isnull().sum())

    #One - hot - encoding for categorical variables
    dataset.train = pd.get_dummies(dataset.train)
    dataset.test = pd.get_dummies(dataset.test)

    train_label = dataset.train['Survived']
    train_data = dataset.train.drop('Survived', axis=1)
    test_data = dataset.test.drop('PassengerId', axis=1).copy()

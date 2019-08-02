import os
from os.path import join
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# from sklearn.datasets import load_boston
boston = load_boston()
# print(boston.DESCR) #주택가격에 대한 설명
data = boston.data
label = boston.target
# print(label)
columns = boston.feature_names

data = pd.DataFrame(data, columns=columns)
# print(boston.DESCR)
# print(data.head())
# print(data.shape)
# print(data.describe())
# print(data.info())

# from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=2019)
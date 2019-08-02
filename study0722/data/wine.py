from sklearn.datasets import load_wine
# from  util.logfile import logger
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

wine = load_wine()
data = wine.data
label = wine.target
columns = wine.feature_names
classname = wine.target_names
# logger.debug(wine.DESCR)
print(wine.DESCR)
data = pd.DataFrame(data, columns=columns)
# logger.debug(data.info())
print(data.info())

x_train, x_test, y_train, y_test = train_test_split(data, label, stratify=label, random_state=0)
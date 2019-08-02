from sklearn.linear_model import LinearRegression
from data import boston as bd
import os
from os.path import join
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

sim_lr = LinearRegression()
sim_lr.fit(bd.x_train['RM'].values.reshape((-1,1)),bd.y_train)
y_pred = sim_lr.predict(bd.x_test['RM'].values.reshape((-1,1)))

plt.scatter(bd.x_test['RM'], bd.y_test, s=10, c='black')
plt.plot(bd.x_test['RM'], y_pred, c='red')
plt.legend(['Regression line', 'x_test'], loc='upper left')
print('단순 선형 회귀, R2 : {:.4f}'.format(r2_score(bd.y_test, y_pred)))
# plt.show()

mul_lr = LinearRegression()
mul_lr.fit(bd.x_train.values, bd.y_train)
y_pred = mul_lr.predict(bd.x_test.values)
print('다중 선형 회귀, R2 : {:.4f}'.format(r2_score(bd.y_test, y_pred)))


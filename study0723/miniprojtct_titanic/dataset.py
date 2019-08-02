import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import  seaborn as sns

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

#데이터분석
#survivied는 생존 여부 (0은 사망, 1은 생존; train 데이터에서만 제공),
#pclass는 사회경제적 지위
#SipSp는 배우자나 형제 자매 명 수의 총 합,
#Parch는 부모 자식 명 수의 총 합을 나타낸다.

print('train data shape: ', train.shape)
print('test data shape: ', test.shape)
print('--------------[train infomation]-----------------')
print(train.info())
print(train.head())
print('--------------[test infomation]-----------------')
print(test.info())
print(test.head())

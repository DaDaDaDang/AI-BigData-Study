#kaggel 제공
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

#데이터 로드
data = pd.read_csv('../data/creditcard.csv')
print(data.head())
print(data.columns) #컬럼명 확인
#데이터 빈도수 확인
print(pd.value_counts(data['Class']))

pd.value_counts(data['Class']).plot.bar()
plt.title('Fraud class histogram')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()

#amount stanardscaler 전처리
sdscaler = StandardScaler()
data['normAmount'] = sdscaler.fit_transform(data['Amount'].values.reshape(-1,1))
data = data.drop(['Time', 'Amount'], axis = 1) #불필요한 컬럼 삭제


#데이터 train, test나누기
x=np.array(data.ix[:, data.columns != 'Class'])
y=np.array(data.ix[:, data.columns == 'Class'])


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

print("Number transactions x_train dataset : ", x_train.shape)
print("Number transactions x_train dataset : ", y_train.shape)
print("Number transactions x_train dataset : ", x_test.shape)
print("Number transactions x_train dataset : ", y_test.shape)

print("Before OverSampling, counts of label '1' : {} ".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0' : {} ".format(sum(y_train==0)))
print("y_train : ", y_train)
print("y_train.ravel : ", y_train.ravel())

sm = SMOTE(random_state=2)
x_train_res, y_train_res = sm.fit_sample(x_train, y_train.ravel())

print("After OverSampling, the shape of train_x : {} ".format(x_train_res.shape))
print("After OverSampling, the shape of train_y : {} ".format(y_train_res.shape))

print("After OverSampling, counts of y_train_res '1' : {} ".format(sum(y_train_res==1)))
print("After OverSampling, counts of y_train_res '0' : {} ".format(sum(y_train_res==0)))

print("After OverSampling, the shape of train_x : {} ".format(x_test.shape))
print("After OverSampling, the shape of train_y : {} ".format(y_test.shape))

#실제 정확도를 알암보기 위한 새로운 데이터 갯수
print("Before OverSampling, counts of label '1' : {} ".format(sum(y_test==1)))
print("Before OverSampling, counts of label '0' : {} ".format(sum(y_test==0)))
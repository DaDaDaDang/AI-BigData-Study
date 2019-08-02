from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from plotnine import *
import pandas as pd
import numpy as np


#titanic data에 적용
#data load
data = pd.read_csv('./data/titanic_proc.csv')
data['Survived'] = data['Survived'].astype(str)
label = data['Survived']
del data['Survived']


#scaling
sdscaler = StandardScaler()
sdscaler.fit(data)
sdscaler_data = sdscaler.transform(data)
sdscaler_pd = pd.DataFrame(sdscaler_data, columns = data.columns)

#성능 비교를 위한 test set 설정
X_train, X_test, Y_train, Y_test = train_test_split(sdscaler_pd, label, test_size=0.1, shuffle=True, random_state=5)

#2. smote
smote = SMOTE(k_neighbors=5, random_state=2019)

smoted_data, smoted_label = smote.fit_resample(X_train, Y_train)
smoted_data = pd.DataFrame(smoted_data, columns=data.columns)
smoted_label = pd.DataFrame({'Survived' : smoted_label})

print('원본 데이터의 클래스 비율 \n{}'.format(pd.get_dummies(Y_train).sum()))
print('\nsmoted_label 클래스 비율 \n{}'.format(pd.get_dummies(smoted_label).sum()))


#성능 비교
def train_and_test(model, X_train, Y_train, X_test, Y_test) :
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    accuracy = round(accuracy_score(Y_test, pred)*100, 2)
    print("Accuracy : ", accuracy, "%")

print("original data ")
train_and_test(SVC(gamma='scale'), X_train, Y_train, X_test, Y_test)

print("oversample data ")
train_and_test(SVC(gamma='scale'),smoted_data, np.ravel(smoted_label,order='C'), X_test, Y_test)

#시각화 확인
def ggplot_point(X_train, Y_train, x, y) :
    data = pd.concat([X_train, Y_train], axis =1)
    plot = (ggplot(data)+aes(x=x, y=y, fill='factor(Survived)')+geom_point())
    print(plot)

ggplot_point(X_train, Y_train, X_train.columns[2], X_train.columns[3])
ggplot_point(smoted_data, smoted_label, smoted_data.columns[2], smoted_data.columns[3])

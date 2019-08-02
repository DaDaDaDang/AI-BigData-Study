import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

#데이터 불러오기
data = pd.read_csv('./data/abalone.csv')
label = data['Sex']
del data['Sex']

#scaling
sdscaler = StandardScaler()
sdscaler.fit(data)
sdscaler_data = sdscaler.transform(data)
sdscaler_pd = pd.DataFrame(sdscaler_data, columns = data.columns)

#성능 비교를 위한 test set 설정
X_train, X_test, Y_train, Y_test = train_test_split(sdscaler_pd, label, test_size=0.1, shuffle=True, random_state=5)

## 1. Random Sampling
ros = RandomOverSampler(random_state = 2019)
rus = RandomUnderSampler(random_state = 2019)

oversampled_data, oversampled_label = ros.fit_resample(data, label)
oversampled_data = pd.DataFrame(oversampled_data, columns = data.columns)

undersampled_data, undersampled_label = ros.fit_resample(data, label)
undersampled_data = pd.DataFrame(undersampled_data, columns = data.columns)

print('원본 데이터의 클래스 비율 \n{}'.format(pd.get_dummies(label).sum()))
print('\nRandom Over 샘플링 결과 \n{}'.format(pd.get_dummies(oversampled_label).sum()))
print('\nRandom Under 샘플링 결과 \n{}'.format(pd.get_dummies(undersampled_label).sum()))

#성능 비교
def train_and_test(model, X_train, Y_train, X_test, Y_test) :
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    accuracy = round(accuracy_score(Y_test, pred)*100, 2)
    print("Accuracy : ", accuracy, "%")

print("original data ")
train_and_test(SVC(), X_train, Y_train, X_test, Y_test)

print("oversample data ")
train_and_test(SVC(), oversampled_data, oversampled_label, X_test, Y_test)

print("undersample data ")
train_and_test(SVC(), undersampled_data, undersampled_label, X_test, Y_test)
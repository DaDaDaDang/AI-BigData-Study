import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

#데이터 위치 path
PATH = '../data/'

#타이타닉 데이터 불러오기
train_data = pd.read_csv(PATH + 'train.csv')
test_data = pd.read_csv(PATH + 'test.csv')
gender_submission = pd.read_csv(PATH + 'gender_submission.csv')

#데이터 지정
PassengerId = test_data['PassengerId']

#데이터 head확인
print(train_data.head())
print(test_data.head())

#descrive
print(train_data.describe())
# print(test_data.describe())

#columns
print(train_data.columns)

#dtype
print(train_data.dtypes)

#number of missing value
column_names = train_data.columns
for column in column_names:
    print(column + ' - ' + str(train_data[column].isnull().sum())) #The columns 'Age' and 'Cabin' contains more null values.

# Insights
# # 'Survived' is the target column/variable.
# # 'PassengerId', 'Name' and 'Ticket' doesn't contribute to the target variable 'Survived'. So, we can remove it from the data.
# # 'Age' and 'Embarked' has less number of missing value. We have to impute them using different techniques.
# # As there are a lot of missing values in the column 'Cabin', we can remove it from the training data.
# # 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare' doesn't have any missing values.
# # We can also create new variable like 'total size of the family' from the columns 'SibSp' and 'Parch'.

#  Visualization of 'Survived' (Target column)
# # As we know, majority of passengers couldn't survive.
# # Data is imbalanced.


##########################
#########시각화###########
##########################

#survived의 0과 1에 대한 갯수를 셈. 0 : dead, 1 : survived
print(train_data.Survived.value_counts())

#생존에 대한 plot차트 그리기
plt.figure()
sns.set()
suv = train_data.Survived.value_counts().plot(kind = 'bar', color = ['blue', 'red'])
suv.set_xlabel('Survived or not')
suv.set_ylabel('Passenger Count')
plt.xticks(np.arange(2), ['DEAD', 'SURVIVED'])
plt.show()

#승객등급에 따른 plot차트 그리기
pcl = train_data.Pclass.value_counts().sort_index().plot('bar', title='', color = ['red', 'orange', 'yellow'])
pcl.set_xlabel('Pclass')
pcl.set_ylabel('Survival Probability')
plt.show()

#승객등급별 사람 수
print(train_data[['Pclass', 'Survived']].groupby('Pclass').count())

#승객등급별 생존자 수
print(train_data[['Pclass', 'Survived']].groupby('Pclass').sum())

#승객등급별 생존자 비율 <= 1
pcPerSuv = train_data[['Pclass', 'Survived']].groupby('Pclass').mean().Survived.plot('bar')
pcPerSuv.set_xlabel('Pclass')
pcPerSuv.set_ylabel('Survival Probability')
plt.show()

#전체승객의 성별 수
sex = train_data.Sex.value_counts().sort_index().plot('bar')
sex.set_xlabel('Sex')
sex.set_ylabel('Passenger count')
plt.show()

#성별에 따른 생존자 수
sexSuv = train_data[['Sex', 'Survived']].groupby('Sex').mean().Survived.plot('bar')
sexSuv.set_xlabel('Sex')
sexSuv.set_ylabel('Survival Probability')
plt.show()

#항구별 승객 수
emb = train_data.Embarked.value_counts().sort_index().plot('bar')
emb.set_xlabel('Embarked')
emb.set_ylabel('Passenger count')
plt.show()

#항구별 생존자 수
embSuv = train_data[['Embarked', 'Survived']].groupby('Embarked').mean().Survived.plot('bar')
embSuv.set_xlabel('Embarked')
embSuv.set_ylabel('Passenger count')
plt.show()

#동승자 수와 생존자 수의 비율
ssp = train_data[['SibSp', 'Survived']].groupby('SibSp').mean().Survived.plot('bar')
ssp.set_xlabel('SibSp')
ssp.set_ylabel('Survival Probability')
plt.show()

#부모자식수와 생존자수의 비율
ps = train_data[['Parch', 'Survived']].groupby('Parch').mean().Survived.plot('bar')
ps.set_xlabel('Parch')
ps.set_ylabel('Survival Probability')
plt.show()

#항구마다 승객등급 비교
sns.factorplot('Pclass', col = 'Embarked', data = train_data, kind = 'count')
plt.show()


#승객등급마다 성별 비교
sns.factorplot('Sex', col = 'Pclass', data = train_data, kind = 'count')
plt.show()

#항구마다 성별 비교
sns.factorplot('Sex', col = 'Embarked', data = train_data, kind = 'count')
plt.show()


#########################################################################
#                  train_data, test_data 전처리                         #
#########################################################################
#Create a new feature 'Family size' from the features 'SibSp' and 'Parch'
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1

train_data['Sex'] = train_data['Sex'].map({'male':0, 'female':1})
train_data['Embarked'] = train_data['Embarked'].map({'C':0, 'Q':1, 'S':2})
#Remove unnecessary columns
train_data = train_data.drop(columns=['Ticket', 'PassengerId', 'Cabin'])

train_data['Title'] = train_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
train_data = train_data.drop(columns='Name')

train_data['Title'] = train_data['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Countess', 'Sir', 'Jonkheer', 'Lady', 'Capt', 'Don'], 'Others')
train_data['Title'] = train_data['Title'].replace('Ms', 'Miss')
train_data['Title'] = train_data['Title'].replace('Mme', 'Mrs')
train_data['Title'] = train_data['Title'].replace('Mlle', 'Miss')
train_data['Title'] = train_data['Title'].map({'Master':0, 'Miss':1, 'Mr':2, 'Mrs':3, 'Others':4})

ts = train_data[['Title', 'Survived']].groupby('Title').mean().Survived.plot('bar')
ts.set_xlabel('Title')
ts.set_ylabel('Survival Probability')
plt.show()

#handling missing value
print(train_data.isnull().sum())
#Emb에서 null의 갯수
print(train_data['Embarked'].isnull().sum())
#2로 채운다.
train_data['Embarked'] = train_data['Embarked'].fillna(2)
print(train_data.head())

#Age 의 널값 처리
NaN_indexes = train_data['Age'][train_data['Age'].isnull()].index

for i in NaN_indexes:
    pred_age = train_data['Age'][((train_data.SibSp == train_data.iloc[i]["SibSp"]) & (train_data.Parch == train_data.iloc[i]["Parch"])
                                  & (train_data.Pclass == train_data.iloc[i]["Pclass"]))].median()
    if not np.isnan(pred_age):
        train_data['Age'].iloc[i] = pred_age
    else:
        train_data['Age'].iloc[i] = train_data['Age'].median()

#null값 처리 확인
print(train_data.isnull().sum())

#testdata 전처리
test_data = test_data.drop(columns=['Ticket', 'PassengerId', 'Cabin'])
test_data['Sex'] = test_data['Sex'].map({'male':0, 'female':1})
test_data['Embarked'] = test_data['Embarked'].map({'C':0, 'Q':1, 'S':2})
test_data['Title'] = test_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test_data = test_data.drop(columns='Name')

test_data['Title'] = test_data['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Countess', 'Sir', 'Jonkheer', 'Lady', 'Capt', 'Don'], 'Others')
test_data['Title'] = test_data['Title'].replace('Ms', 'Miss')
test_data['Title'] = test_data['Title'].replace('Mme', 'Mrs')
test_data['Title'] = test_data['Title'].replace('Mlle', 'Miss')

test_data['Title'] = test_data['Title'].map({'Master':0, 'Miss':1, 'Mr':2, 'Mrs':3, 'Others':4})
print(test_data.isnull().sum())

NaN_indexes = test_data['Age'][test_data['Age'].isnull()].index

for i in NaN_indexes:
    pred_age = train_data['Age'][((train_data.SibSp == test_data.iloc[i]["SibSp"]) & (train_data.Parch == test_data.iloc[i]["Parch"]) & (test_data.Pclass == train_data.iloc[i]["Pclass"]))].median()
    if not np.isnan(pred_age):
        test_data['Age'].iloc[i] = pred_age
    else:
        test_data['Age'].iloc[i] = train_data['Age'].median()


title_mode = train_data.Title.mode()[0]
test_data.Title = test_data.Title.fillna(title_mode)

fare_mean = train_data.Fare.mean()
test_data.Fare = test_data.Fare.fillna(fare_mean)

test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1

from sklearn.utils import shuffle
train_data = shuffle(train_data)

# training_data, valid_data = train_test_split(train_data, test_size=0.2)

X_train = train_data.drop(columns='Survived')
y_train = train_data.Survived
y_train = pd.DataFrame({'Survived':y_train.values})

# X_valid = valid_data.drop(columns='Survived')
# y_valid = valid_data.Survived

X_test = test_data
y_test = gender_submission.drop(columns='PassengerId')

###################################################################
#                           모델, 예측                            #
###################################################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("rf accuracy:{:.2f}%".format(accuracy_score(y_test, rf_pred)*100))

# rfd = rf_pred.data
# rfc = rf_pred.columns
# sf = pd.DataFrame(rfd, columns=rfc)
# submission = sf.to_csv(rf_pred, "submission.csv")

submission = pd.DataFrame({
        "PassengerId": PassengerId,
        "Survived": rf_pred
    })

submission.to_csv('submission.csv', index=False)

submission = pd.read_csv('submission.csv')
print(submission.head())
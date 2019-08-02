from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd
import matplotlib.pyplot as plt
from miniprojtct_titanic import dataset

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

#Sex,Pclass, Embarked 생존율 확인 (숙제)
def pie_chart(feature) :
    feature_ratio = dataset.train[feature].value_counts(sort=False)
    feature_size = feature_ratio.size
    feature_index = feature_ratio.index
    survived = dataset.train[dataset.train['Survived'] == 1][feature].value_counts()
    dead = dataset.train[dataset.train['Survived']==0][feature].value_counts()

    print("feature_size\n", feature_size)
    print("feature_index\n", feature_index)
    print("survived count\n", survived)
    print("dead count\n", dead)

    plt.plot(aspect = 'auto')
    plt.pie(feature_ratio, labels=feature_index, autopct='%1.1f%%')
    plt.title(feature + '\'s ratio in total')
    plt.show()

    for i, index in enumerate(feature_index) :
        plt.subplot(1, feature_size + 1, i + 1, aspect = 'equal')
        plt.pie([survived[index], dead[index]], labels = ['Survivied', 'Dead'], autopct = '%1.1f%%')
        plt.title(str(index) + '\'s ratio')

    plt.show()

def bar_chart(feature) :
    survived = dataset.train[dataset.train['Survived']==1][feature].value_counts()
    dead = dataset.train[dataset.train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived, dead])

    print("survived", survived)
    print("dead", dead)
    df.index = ['Survived', 'Dead']
    df.plot(kind = 'bar', stacked = True)
    plt.show()


def processing():
    # def sexFature () :
    dataset_ = [dataset.train, dataset.test]

    for ds in dataset_:
        ds['Sex'] = ds['Sex'].astype(str)

    # def embarkedFeature () :
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
        ds.loc[ds['Age'] <= 16, 'Age'] = 0
        ds.loc[(ds['Age'] > 16) & (ds['Age'] <= 32), 'Age'] = 1
        ds.loc[(ds['Age'] > 32) & (ds['Age'] <= 48), 'Age'] = 2
        ds.loc[(ds['Age'] > 48) & (ds['Age'] <= 64), 'Age'] = 3
        ds.loc[ds['Age'] > 64, 'Age'] = 4
        ds['Age'] = ds['Age'].map({0: 'Child', 1: 'Young', 2: 'Middle', 3: 'Prime', 4: 'Old'}).astype(str)

        # sibsp&Parch Reature
        for ds in dataset_:
            ds['Family'] = ds['Parch'] + ds['SibSp']
            ds['Family'] = ds['Family'].astype(int)


def dataCheck():
    features_drop = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
    dataset.train = dataset.train.drop(features_drop, axis=1)
    dataset.test = dataset.test.drop(features_drop, axis=1)

    print(dataset.train.head())
    print(dataset.test.head())
    print("train.isnull().sum()", dataset.train.isnull().sum())

    # One - hot - encoding for categorical variables
    dataset.train = pd.get_dummies(dataset.train)
    dataset.test = pd.get_dummies(dataset.test)

    train_label = dataset.train['Survived']
    train_data = dataset.train.drop('Survived', axis=1)
    test_data = dataset.test.drop('PassengerId', axis=1).copy()
    test_data.fillna(test_data.mean())
    print(train_data.columns)
    print(test_data.columns)
    test_data_co = test_data.fillna(test_data.mean())

    log_pred = train_and_test(LogisticRegression(), train_data, train_label)

    svm_pred = train_and_test(SVC(), train_data, train_label)

    rf_pred = train_and_test(RandomForestClassifier(n_estimators=100), train_data, train_label)

    log_pred = train_and_test_cv(LogisticRegression(), train_data, train_label)


def train_and_test(model, train_data, train_label):
    X_train, X_test, Y_train, Y_test = train_test_split(train_data, train_label, test_size=0.2, shuffle=True,
                                                        random_state=5)
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    accuracy = round(accuracy_score(Y_test, prediction) * 100, 2)
    print("Accuracy : ", accuracy, "%")

def train_and_test_cv(model, train_data, train_label, test, disp):
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, train_data, train_label, cv = 10)
    print(scores)
    print(disp+ 'rf k-fold CV score:{:2f}%'.format(scores.mean()))
    model.fit(train_data, train_label)
    prediction = model.predict(test)
    return prediction





if __name__ == '__main__' :
    pie_chart('Sex')
    pie_chart('Pclass')
    pie_chart('Embarked')

    bar_chart('SibSp')
    bar_chart('Parch')
    bar_chart('Embarked')

    processing()
    dataCheck()

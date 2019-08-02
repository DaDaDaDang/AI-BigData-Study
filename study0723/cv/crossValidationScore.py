from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from IPython.display import display
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs #인위적인 데이터셋 만듬

def set__data() :
    iris = load_iris()
    kf_data = iris.data
    kf_label = iris.target
    kf_columns = iris.feature_names
    return iris, kf_data, kf_label, kf_columns

def k_fold() :
    data = set__data()

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=2019)
    scores = cross_val_score(rf, data[1], data[2], cv=10)

    print(scores)
    print('rf k-fold CV score : {:.2f}%'.format(scores.mean()))

def k_fold_validate() :
    data = set__data() # 0 : iris, 1 : kf_data, 2 : kf_label, 3 : kf_columns

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=2019)
    scores = cross_validate(rf,data[1], data[2], cv=10, return_train_score=True)

    print("<<score>>")
    display(scores)
    res_df = pd.DataFrame(scores)
    print("<<res_df>>")
    display(res_df)
    print("평균 시간과 점수 : \n", res_df.mean())


def Stratified_KFold_ex():
    kfold = KFold(n_splits=5)
    stratifiedKfold = StratifiedKFold(n_splits=5)

    iris = load_iris()
    x, y = make_blobs(random_state=0)

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)

    print("교차 검증 점수(kfold) : \n", cross_val_score(logreg, iris.data, iris.target, cv=kfold))
    print("교차 검증 점수(stratifiedKfold) : \n", cross_val_score(logreg, iris.data, iris.target, cv=stratifiedKfold))


if __name__ == "__main__" :
    k_fold()
    k_fold_validate()
    Stratified_KFold_ex()
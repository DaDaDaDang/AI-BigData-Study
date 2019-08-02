from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def irisData() : #iris 데이터 변수에 저장
    iris = load_iris()
    kf_data = iris.data
    kf_label = iris.target
    kf_columns = iris.feature_names

    kf_data = pd.DataFrame(kf_data, columns=kf_columns)

    return kf_data, kf_label

def kFold() : #KFold 모델
    #KFold 모델 변수 저장, 5개로 나누어
    kf = KFold(n_splits=5, random_state=0)
    #iris 데이터 변수에 저장
    iData = irisData()
    # kf_data = iData[0]
    # kf_label = iData[1]
    # dataIdxPrint(kf, kf_data, kf_label)
    #데이터 인덱스 정보 출력
    dataIdxPrint(kf,iData[0],iData[1])
    #RandomForestClassifier 모델로 모델학습을 하여 Accuracy 출력
    # modelLearning(kf,iData[0],iData[1])

def stratified_kFold() : #StratifiedKFold 모델
    # StratifiedKFold 모델 변수 저장, 5개로 나누어
    kf = StratifiedKFold(n_splits=5, random_state=0)
    # iris 데이터 변수에 저장
    iData = irisData()
    # 데이터 인덱스 정보 출력
    dataIdxPrint(kf, iData[0], iData[1])
    # RandomForestClassifier 모델로 모델학습을 하여 Accuracy 출력
    # modelLearning(kf, iData[0], iData[1])


def dataIdxPrint(model, data, label) : #데이터 정보 출력
    # split()는 학습용과 검증용의 데이터 인덱스 출력
    #enumerate is useful for obtaining an indexed list:
    #    (0, seq[0]), (1, seq[1]), (2, seq[2]), ...

    val_scores = list() # 모델 학습을 위한 리스트

    for i, (train_idx, valid_idx) in enumerate(model.split(data.values, label)):
        train_data, train_label = data.values[train_idx, :], label[train_idx]
        valid_data, valid_label = data.values[valid_idx, :], label[valid_idx]

        # print("{} Fold train label\n{}".format(i, train_label))
        # print("{} Fold valid label\n{}".format(i, valid_label))

        #모델학습
        # RandomForestClassifier 모델로 모델학습을 하여 Accuracy 출력
        valid_acc = modelLearning(i, train_data, train_label, valid_data, valid_label)
        val_scores.append(valid_acc)

        print("Cross Validation Score:{:.2f}%".format(np.mean(val_scores)))

def modelLearning(i, train_data, train_label, valid_data, valid_label) : #RandomForestClassifier 모델로 모델학습을 하여 Accuracy 출력
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=2019)

    #모델학습
    clf.fit(train_data, train_label)

    train_acc = clf.score(train_data, train_label)*100
    valid_acc = clf.score(valid_data, valid_label)*100

    # val_scores.append(valid_acc)
    print("{}fold, train Accuracy : {:.2f}%, validation Accuracy : {:.2f}%".format(i, train_acc, valid_acc))

    return valid_acc



if __name__ == "__main__" :
    print("<<<<<<<<<<<<<<< KFold >>>>>>>>>>>>>>>>>>")
    kFold()
    print("<<<<<<<<<<<<<<< StratifiedKFold >>>>>>>>>>>>>>>>>>")
    stratified_kFold()

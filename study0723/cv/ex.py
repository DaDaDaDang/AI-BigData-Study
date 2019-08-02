from sklearn.datasets import make_blobs #인위적인 데이터셋 만듬
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

def train_test_split_() :
    #인위적인 데이터셋을 만듭니다.
    x, y = make_blobs(random_state=0)

    #데이터와 타켓레이블을 훈련세트와 테스트 세트로 나눕니다.
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    # x_test = x_test.T
    print(x_test)
    # y_test = y_test.T
    print(y_test)
    # #모델 학습
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)

    #테스트 세트 평가
    print("테스트 세트 점수 : {:.2f}".format(logreg.score(x_test, y_test)))

if __name__ == "__main__" :
    train_test_split_()
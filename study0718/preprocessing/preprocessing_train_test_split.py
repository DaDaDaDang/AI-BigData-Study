import numpy as np
from sklearn.model_selection import train_test_split

X = [[0,1], [2, 3], [4, 5], [6, 7], [8, 9]]
Y = [0, 1, 2, 3, 4]

#데이터(X)만 넣었을 경우
X_train, X_test = train_test_split(X, test_size=0.2, random_state=123)
print('X_train : ', X_train)
print('X_test : ', X_test)

#데이터(X)와 레이블(Y)을 넣었을 경우
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=321)
print('X_train : ', X_train)
print('X_test : ', X_test)
print('Y_train : ', Y_train)
print('Y_test : ', Y_test)
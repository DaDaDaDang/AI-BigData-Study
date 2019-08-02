#1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])

x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])

x3 = np.array([101,102,103,104,105])
# y3 = np.array([101,102,103,104,105])
# x4 = np.array([range(30,50)]) #dim을 20개의 열로 인식
x4 = np.array(range(30,50)) #[]가 없으면 하나의 열로 인식, range => [30,31,32,...,50]이 되기때문에 [range(30,50)] => [[30,31,32,...,50]]이 된다. [[1,2,3]]은 3열 1행이된다. [1,2,3]은 1열 3행

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
#model은 순차적인 변수
model = Sequential()
#model.add : layer 1계층, 하나당 한계층이다(ex. model.add가 3세개이면 3개의 계층을 가진다)
#Dense(n) : n은 노드의 개수 input_dim=1
model.add(Dense(20, input_dim = 1, activation = 'relu'))
model.add(Dense(30))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))

#3. 훈련
#model.compitle : 컴파일 시킨다.
#model.fit : x_train, y_train은 훈련 데이터, epochs은 훈련 시킬 횟수, batch_size = 데이터 단위
#batch_size : 지정 batch_size하지 않으면 기본값은 32입니다.
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 30 ,batch_size = 1)
# model.fit(x_train, y_train, epochs = 30)

#4. 평가 예측
#평가하기
loss, acc = model.evaluate(x_test, y_test, batch_size = 1) 
print("acc : ", acc)
#실제 예측값 확인
# y_predict = model.predict(x_test)
# y_predict = model.predict(x_train)
# y_predict = model.predict(x3)
y_predict = model.predict(x4)
print(y_predict)
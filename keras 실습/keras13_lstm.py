from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

#1. 데이터
X = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
Y = array([4,5,6,7])
# Y1 = array([[4,5,6,7]])

print("X.shape : ", X.shape) #(4,3)
print("Y.shape : ", Y.shape) #(4,)
# print("Y1.shape : ", Y1.shape) #(4,)

#reshape from [samles, timesteps] into [samples, timesteps, teatures]
X = X.reshape((X.shape[0], X.shape[1], 1)) 

print("X.shape : ", X.shape) #(4,3,1) 4행 3열에 1개씩 데이터가 돌아간다.
print("Y.shape : ", Y.shape) #(4,)

#2. 모델 구성
model = Sequential()
model.add(LSTM(30, activation='relu', input_shape=(3, 1))) #input_shape = (4, 3, 1) <- 행 무시 ; input_shape = (3, 1)
# model.add(LSTM(30))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(30))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mse')


#3. 실행
model.fit(X, Y, epochs = 200, verbose = 2)
#deomonstrate prdiction
x_input = array([70, 80, 90])      # 70, 80, 90 => ?
x_input = x_input.reshape((1, 3, 1))
yhat = model.predict(x_input, verbose = 0)
print(yhat)

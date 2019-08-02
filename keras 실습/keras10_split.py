#1. 데이터
import numpy as np
#y=wx+b x와 y의 값을 정제하여 넣어주는 것이 중요하다. 여기서는 임의의 값을 넣어준다. x=y 
#split :  np.split(x, [3, 5, 6, 10]) : x-데이터 arr, [3, 5, 6, 10]-3번째까지, 5번째까지, 6번째까지, 10번째까지
XXX=(range(100))
YYY=(range(100))

#train_test_split : arr의 내용을 train과 test로 종속, 독립이 같은 값이 되도록 나누어 준다.
from sklearn.model_selection import train_test_split

x_train, x_test, y_train,y_test = train_test_split(XXX, YYY, test_size=0.4, train_size=0.6, shuffle=True)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, train_size=0.5, shuffle=True)

print("x_train : ", x_train)
print("y_train : ", y_train)
print("x_test : ", x_test)
print("y_test : ", y_test)
print("x_val : ", x_val)
print("y_val : ", y_val)



#2. 모델구성
#최적의 weight값을 구함. 'w'
from keras.models import Sequential
from keras.layers import Dense
#model은 순차적인 변수
model = Sequential()
#model.add : layer 1계층, 하나당 한계층이다(ex. model.add가 3세개이면 3개의 계층을 가진다)
#Dense(n) : n은 output노드의 개수, input_dim=1 input노드의 개수5
model.add(Dense(1024, input_shape = (1,), activation = 'relu')) #input_shape = () <- (,1)이 1열이라는 뜻이지만 행을 무시했다는 의미로 (1,)로 쓴다. input_shape=(1,) == input_dim=1
model.add(Dense(512))
model.add(Dense(1024))
model.add(Dense(512))
model.add(Dense(128))
model.add(Dense(8))
model.add(Dense(1))

#3. 훈련
#model.compitle : 컴파일 시킨다.
#model.fit : x_train, y_train은 훈련 데이터, epochs은 훈련 시킬 횟수, batch_size = 데이터 단위
#batch_size : 지정 batch_size하지 않으면 기본값은 32입니다.
#lose : mse, mean_squared_error, mae, mean_absolute_error
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy']) #mse 오차범위 (평균 제곱 오차) 수가 적을수록 좋다.
# model.compile(loss = 'mse', optimizer = 'adam') #mse 오차범위 (평균 제곱 오차) 수가 적을수록 좋다.
# ------------------------------------------------------------------
#|      |  model.add   |              model.compile                 |
#|      |-----------------------------------------------------------
#|      |  activation  |           loss            |    optimizer   |
#|------------------------------------------------------------------|
#|      |              |            mse            |                |
#|  회  |     relu     |     mean_squared_error    |                |
#|  귀  |              |            mae            |                |
#|      |    linear    |     mean_absolute_error   |      adam      |
#|-------------------------------------------------|                |
#|      |    softmax   | categofical_crossentropy  |     rmsprop    |
#| 분류 |------------------------------------------|                |
#|      |    sigmoid   |    binary_crossentropy    |       sgd      |
# ------------------------------------------------------------------
model.fit(x_train, y_train, epochs = 100 ,batch_size = 1, validation_data=(x_val, y_val)) #validation_data : 훈련할 때 머신이 직접 검증할 수 있도록 넣어주는 데이터셋
# model.fit(x_train, y_train, epochs = 30) #batch_size를 없애면 기본값이 32로 설정된다.

#4. 평가 예측
#평가하기
loss, acc = model.evaluate(x_test, y_test, batch_size = 1) 
print("acc : ", acc)
print("loss : ", loss)

#실제 예측값 확인
# y_predict = model.predict(x_test)
# y_predict = model.predict(x_train)
# y_predict = model.predict(x3)
y_predict = model.predict(x_test)
print(y_predict)

#RMSE 구하기 : 평균 제곱근 오차
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) : #x_test 값으로 구한 y_predict 갑과 원래의 정답의 값인 y_test값을 비교하여 RMSE를 구함. 내재된 함수가 없기때문에 만든 함수. MSE에 sqrt를 씌워서 RMSE를 만듬.
    return np.sqrt(mean_squared_error(y_test,y_predict))

print("RMSE : ", RMSE(y_test, y_predict))

#R2 : 결졍계수
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)

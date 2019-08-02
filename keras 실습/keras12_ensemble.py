#data_set이 여러개인 경우의 여러모델을 하나의 모델로 머지하여 앙상블하기.
#1. 데이터
import numpy as np
#y=wx+b x와 y의 값을 정제하여 넣어주는 것이 중요하다. 여기서는 임의의 값을 넣어준다. x=y 
#split :  np.split(x, [3, 5, 6, 10]) : x-데이터 arr, [3, 5, 6, 10]-3번째까지, 5번째까지, 6번째까지, 10번째까지
XXX1=np.array([range(100),range(311,411),range(1000,1100)])
XXX2=np.array([range(100,200),range(411,511),range(1100,1200)])
YYY1=np.array([range(501,601), range(622,522,-1),range(800,900)])
YYY2=np.array([range(601,701), range(722,622,-1),range(900,1000)])

XXX1 = XXX1.T
XXX2 = XXX2.T
YYY1 = YYY1.T
YYY2 = YYY2.T

#train_test_split : arr의 내용을 train과 test로 종속, 독립이 같은 값이 되도록 나누어 준다.
from sklearn.model_selection import train_test_split

x1_train, x1_test, y1_train, y1_test = train_test_split(XXX1, YYY1, test_size=0.4, train_size=0.6, shuffle=True)
x2_train, x2_test, y2_train, y2_test = train_test_split(XXX2, YYY2, test_size=0.4, train_size=0.6, shuffle=True)
x1_test, x1_val, y1_test, y1_val = train_test_split(x1_test, y1_test, train_size=0.5, shuffle=True)
x2_test, x2_val, y2_test, y2_val = train_test_split(x2_test, y2_test, train_size=0.5, shuffle=True)

# print(y1_test)


#2. 모델구성
#최적의 weight값을 구함. 'w'
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.layers.merge import concatenate

input1 = Input(shape=(3,))
input2 = Input(shape=(3,))

# model 1
x1 = Dense(100, activation='relu')(input1)

# model 2 
x2 = Dense(100, activation='linear')(input2)

# merging models
x3 = concatenate([x1, x2]) #두개의 model을 merge하여 하나의 model로 만듬.

# output layer // 두개의 output을 만듬
output1 = Dense(50, activation = 'relu')(x3) #Dense(output노드수,)(input데이터?노드?)
output1 = Dense(40, activation='relu')(output1)
output1 = Dense(30, activation='relu')(output1)
predictions1 = Dense(3, activation='relu')(output1) #최종출력물 3개

output2 = Dense(50, activation = 'relu')(x3) #Dense(output노드수,)(input데이터?노드?)
output2 = Dense(40, activation='relu')(output2)
output2 = Dense(30, activation='relu')(output2)
output2 = Dense(20, activation='relu')(output2)
predictions2 = Dense(3, activation='relu')(output2) #최종출력물 3개

# generate a model from the layers above
model = Model(inputs=[input1,input2], outputs=[predictions1,predictions2])


# Always a good idea to verify it looks as you expect it to 
model.summary()

# model.fit([x1_train, x2_train], [y1_train,y2_train], epochs = 60 ,batch_size = 1, validation_data=([x1_val,x2_val], [y1_val,y2_val])) #validation_data : 훈련할 때 머신이 직접 검증할 수 있도록 넣어주는 데이터셋

#3. 훈련
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

model.fit([x1_train, x2_train], [y1_train,y2_train],
             epochs = 100 ,batch_size = 1, 
             validation_data=([x1_val,x2_val], [y1_val,y2_val])) #validation_data : 훈련할 때 머신이 직접 검증할 수 있도록 넣어주는 데이터셋


#4. 평가 예측
#평가하기
# loss, acc = model.evaluate([x1_test, x2_test],[y1_test, y2_test], batch_size = 1) 
acc = model.evaluate([x1_test, x2_test],[y1_test, y2_test], batch_size = 1) 
print("acc : ", acc)
# print("loss : ", loss)
'''
y1_predict, y2_predict = model.predict([x1_test, x2_test])

from sklearn.metrics import mean_squared_error
def RMSE( y_test, y_predict) : #x_test 값으로 구한 y_predict 갑과 원래의 정답의 값인 y_test값을 비교하여 RMSE를 구함. 내재된 함수가 없기때문에 만든 함수. MSE에 sqrt를 씌워서 RMSE를 만듬.
   return np.sqrt(mean_squared_error(y_test, y_predict))

RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)

print("RMSE1 : ", RMSE1)
print("RMSE2 : ", RMSE2)

from sklearn.metrics import r2_score

print("R2_1 : ", r2_score(y1_test, y1_predict))
print("R2_2 : ", r2_score(y2_test, y2_predict))

'''


#실제 예측값 확인
y_predict = model.predict([x1_test, x2_test])

#RMSE 구하기 : 평균 제곱근 오차
from sklearn.metrics import mean_squared_error
def RMSE( y_predict, y_test) : #x_test 값으로 구한 y_predict 갑과 원래의 정답의 값인 y_test값을 비교하여 RMSE를 구함. 내재된 함수가 없기때문에 만든 함수. MSE에 sqrt를 씌워서 RMSE를 만듬.
    y=len(y_test)
    for i in range(y):
        print("RMSE : ",np.sqrt(mean_squared_error(y_test[i],y_predict[i],multioutput='raw_values')))

RMSE( y_predict, [y1_test,y2_test])

#R2 : 결졍계수
from sklearn.metrics import r2_score
def r2__predict(y_test, y_predict) :
    y = len(y_test)
    for i in range(y):
        print("R2 : ", r2_score(y_test[i], y_predict[i]))

r2__predict([y1_test,y2_test],y_predict)
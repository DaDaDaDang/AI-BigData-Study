#1. 데이터
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
#model은 순차적인 변수
model = Sequential()
#model.add : layer 1계층, 하나당 한계층이다(ex. model.add가 3세개이면 3개의 계층을 가진다)
#Dense(n) : n은 output노드의 개수 input_dim=1 : input 노드의 개수는 1개이다. activation : ?
model.add(Dense(2, input_dim = 1, activation = 'relu'))
model.add(Dense(253))
model.add(Dense(4556))
model.add(Dense(3455))
model.add(Dense(456))
model.add(Dense(1))

model.summary() #Param(파라미터 계산 ) : (input node 수 + bias(바이어스, 계층마다 하나씩 붙는다))*node 수
'''
#3. 훈련 : 컴파일 시키고 fit시킨다.
#model.compile :
#model.fit :
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x, y, epochs = 90, batch_size = 1)

#4. 평가 예측
loss, acc = model.evaluate(x, y, batch_size = 1)

print("acc : ", acc)
'''
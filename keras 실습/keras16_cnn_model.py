from keras.models import Sequential
from keras.layers import Conv2D, Flatten

# filter_size = 32 #output
# kernel_size = (3, 3)
model = Sequential()

model.add(Conv2D(32, (2, 2), padding = 'same',
                input_shape = (7, 7, 1))) #padding => 'same'이면 주변에 0으로 이루어진 프레임을 한번 더 씌워서 인풋된 사이즈와 같은 사이즈의 아웃풋을 만들어 낸다.

model.add(Conv2D(16, (3, 3)))
model.add(Conv2D(100, (3, 3)))

from keras.layers import MaxPooling2D
model.add(MaxPooling2D(pool_size=2)) #MaxPooling은 이미지를 pool_size로 나누어 가장 MAX값을 빼와서 사이즈를 줄인다. 이때 MAX값은 특성값으로 볼 수 있다.

model.add(Flatten())
# from keras.layers import MaxPooling2D
# pool_size = (2, 2)
# model.add(MaxPooling2D(pool_size))

model.summary()
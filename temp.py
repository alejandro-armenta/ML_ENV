from keras.datasets import cifar10
from keras import utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout

import matplotlib.pyplot as plt

import numpy as np


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

fig = plt.figure(figsize=(20,5))

for i in range(36):
    
    ax = fig.add_subplot(3, 12, i+1, xticks=[], yticks=[])

    ax.imshow(x_train[i])

plt.tight_layout()
plt.savefig('images.png')

x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

num_classes = len(np.unique(y_train))

y_train = utils.to_categorical(x=y_train, num_classes=num_classes)
y_test = utils.to_categorical(x=y_test, num_classes=num_classes)

x_train, x_valid = x_train[5000:], x_train[:5000]
y_train, y_valid = y_train[5000:], y_train[:5000]


print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)
print(x_test.shape, y_test.shape)

model = Sequential()

model.add(
    Conv2D(
        filters=16,
        kernel_size=2,
        activation='relu',
        padding='same',
        input_shape=(32,32,3)
    )
)

model.add(    
    MaxPool2D(pool_size=2)
)

#16*2*2*3 + 16 = 208

model.add(
    Conv2D(
        filters=32,
        kernel_size=2,
        activation='relu',
        padding='same',
    )
)

model.add(    
    MaxPool2D(pool_size=2)
)

#32 * (2*2*16) + 32 = 2080

model.add(
    Conv2D(
        filters=64,
        kernel_size=2,
        activation='relu',
        padding='same',
    )
)

#64 * (2*2*32) + 64 = 8256

model.add(    
    MaxPool2D(pool_size=2)
)

# apaga el 30 por ciento de mi data 
# hay una dependencia de los pesos 

model.add(Dropout(rate=0.3))

model.add(Flatten())

model.add(
    Dense(
        units=500,
        activation='relu'
    )
)

model.add(Dropout(rate=0.4))

model.add(
    Dense(
        units=10,
        activation='softmax'
    )
)

model.summary()





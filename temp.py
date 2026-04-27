
from keras.datasets import mnist

from keras import utils

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


(x_train, y_train), (x_test, y_test) = mnist.load_data()

fig = plt.figure(figsize=(20,20))
#print(y_train[0])

for i in range(6):
    ax = fig.add_subplot(1,6, i+1, xticks=[], yticks=[])
    ax.imshow(x_train[i], cmap='gray')
    ax.set_title(str(y_train[i]))

plt.tight_layout()
plt.savefig('images.png')


x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

y_train = utils.to_categorical(x=y_train, num_classes=10)
y_test = utils.to_categorical(x=y_test, num_classes=10)

#print(x_train.shape)

img_rows, img_cols = 28, 28

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

print(x_train.shape)
print(x_test.shape)











import keras
from keras.models import Sequential

from keras.datasets import cifar10


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype(dtype='float32')
x_test = x_test.astype(dtype='float32')


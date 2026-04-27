from keras.datasets import mnist

from model import model
from utils import prepare_data

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test, y_train, y_test = prepare_data(x_train, x_test, y_train, y_test)

model.load_weights('model_best.keras')

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

score = model.evaluate(x=x_test, y=y_test)

print(score)
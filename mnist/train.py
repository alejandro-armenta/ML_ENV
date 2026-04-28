

import keras 

from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint

from model import model

from utils import prepare_data

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


(x_train, y_train), (x_test, y_test) = mnist.load_data()

def visualize():

    fig = plt.figure(figsize=(20,20))
    #print(y_train[0])

    for i in range(6):
        ax = fig.add_subplot(1,6, i+1, xticks=[], yticks=[])
        ax.imshow(x_train[i], cmap='gray')
        ax.set_title(str(y_train[i]))

    plt.tight_layout()
    plt.savefig('images.png')

x_train, x_test, y_train, y_test = prepare_data(x_train, x_test, y_train, y_test)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

ch = ModelCheckpoint(filepath='model_best.keras', verbose=1, save_best_only=True)

model.fit(
    x=x_train, 
    y=y_train, 
    batch_size=32, 
    epochs=12, 
    validation_data=(x_test, y_test), 
    verbose=1, 
    shuffle=True,
    callbacks=[ch]
)

#loaded_model = keras.saving.load_model('best_model.keras')





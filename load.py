
from keras.datasets import cifar10
from keras import utils
import keras

import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

num_classes = len(np.unique(y_train))

y_train = utils.to_categorical(x=y_train, num_classes=num_classes)
y_test = utils.to_categorical(x=y_test, num_classes=num_classes)

x_train, x_valid = x_train[5000:], x_train[:5000]
y_train, y_valid = y_train[5000:], y_train[:5000]

model = keras.saving.load_model("model_best.keras")

model.summary()

model.evaluate(x_test, y_test)

y_hat =  model.predict(x_test)

cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# plot a random sample of test images, their predicted labels, and ground truth
fig = plt.figure(figsize=(20, 8))
for i, idx in enumerate(np.random.choice(x_test.shape[0], size=32, replace=False)):

    ax = fig.add_subplot(4, 8, i + 1, xticks=[], yticks=[])

    ax.imshow(np.squeeze(x_test[idx]))

    #estos son indices
    pred_idx = np.argmax(y_hat[idx])

    true_idx = np.argmax(y_test[idx])

    ax.set_title("{} ({})".format(cifar10_labels[pred_idx], cifar10_labels[true_idx]),
                 color=("green" if pred_idx == true_idx else "red"))

plt.savefig('predictions.png')

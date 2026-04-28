from keras import utils

def prepare_data(x_train, x_test, y_train, y_test):

    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255

    y_train = utils.to_categorical(x=y_train, num_classes=10)
    y_test = utils.to_categorical(x=y_test, num_classes=10)

    img_rows, img_cols = 28, 28

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    return x_train, x_test, y_train, y_test
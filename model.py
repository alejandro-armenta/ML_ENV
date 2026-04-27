from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

#class my_model():

    #def ini
model = Sequential()

model.add(    
    Conv2D(
        filters=32, 
        kernel_size=(3,3), 
        padding='same', 
        strides=(1,1), 
        activation='relu', 
        input_shape=(28,28,1),
        use_bias=True
    )
)

model.add(    
    MaxPooling2D(pool_size=(2,2))
)

model.add(    
    Conv2D(
        filters=64, 
        kernel_size=(3,3), 
        padding='same', 
        activation='relu',
    )
)

model.add(    
    MaxPooling2D(pool_size=(2,2))
)

model.add(Flatten())

model.add(Dense(units=64, activation='relu'))

model.add(Dense(units=10, activation='softmax'))

model.summary()
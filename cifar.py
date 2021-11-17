from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, ZeroPadding2D
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def make_model():
    model = Sequential()

    # convolution layer
    model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2), activation='relu',
                     input_shape=(32, 32, 3), padding='same'))

    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), activation='relu',
                     padding='same'))

    model.add(BatchNormalization())
    model.add(Dropout(0.3))



    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), activation='relu',
                     padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), activation='relu',
                     padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    #full connected layer
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))

    #output layer
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model




#하이퍼 파라메타
MY_EPOCH = 250
MY_BATCHSIZE = 300
filename = f"cnn_cifa{MY_EPOCH}.h5"

def train(model, x, y):
    x = x / 255.

    y = tf.keras.utils.to_categorical(y, 10)
    history = model.fit(x, y, epochs=MY_EPOCH, batch_size=MY_BATCHSIZE)
    model.save(filename)

    return history

def evaluate_test(x, y):

    model = load_model(filename)
    x = x / 255.
    y = tf.keras.utils.to_categorical(y, 10)
    test_loss, test_acc = model.evaluate(x, y)
    return test_loss, test_acc

def predic_test(x):
    model = load_model(filename)

    x = x.reshape(-1, 32, 32, 3)
    x = x / 255.
    y = model.predict(x)
    index = np.argmax(y)
    print("our index: ", class_names[index])

if __name__ == '__main__':
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)

    # 학습
    cnn = make_model()
    # train(cnn, x_train, y_train)

    evaluate_test(x_test, y_test)

    predic_test(x_test[0])
    print("ans: ", class_names[y_train[0][0]])






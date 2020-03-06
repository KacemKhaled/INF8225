import time
import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adamax
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import backend as K

# import the dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# shuffle the dataset
X_train, y_train = shuffle(X_train, y_train)

# reshape data to fit model
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

# pre-process: divide the "data" by 255 to shrink its size and keep it between [0-1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# convert class vectors to binary class matrices
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# split test data into test and validation
X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# first model to compare
model_0 = Sequential()
model_0.add(Conv2D(32, kernel_size=(3, 3),
                   activation='relu',
                   input_shape=(28, 28, 1)))
model_0.add(MaxPooling2D(pool_size=(4, 4)))
model_0.add(Conv2D(64, (3, 3), activation='relu'))
model_0.add(MaxPooling2D(pool_size=(2, 2)))
model_0.add(Flatten())
model_0.add(Dense(128, activation='relu'))
model_0.add(Dense(10, activation='softmax'))
model_0.compile(loss='categorical_crossentropy', optimizer=Adadelta(lr=1.0), metrics=['accuracy'])

# same as above but with less conv layers
model_1 = Sequential()
model_1.add(Conv2D(64, kernel_size=(3, 3),
                   activation='relu',
                   input_shape=(28, 28, 1)))
model_1.add(MaxPooling2D(pool_size=(4, 4)))
model_1.add(Dropout(0.5))
model_1.add(Flatten())
model_1.add(Dense(128, activation='relu'))
model_1.add(Dropout(0.25))
model_1.add(Dense(10, activation='softmax'))
model_1.compile(loss='categorical_crossentropy', optimizer=Adadelta(lr=1.0), metrics=['accuracy'])

# same as above but with less output channels for the conv layer
model_2 = Sequential()
model_2.add(Conv2D(32, kernel_size=(3, 3),
                   activation='relu',
                   input_shape=(28, 28, 1)))
model_2.add(MaxPooling2D(pool_size=(4, 4)))
model_2.add(Dropout(0.5))
model_2.add(Flatten())
model_2.add(Dense(128, activation='relu'))
model_2.add(Dropout(0.25))
model_2.add(Dense(10, activation='softmax'))
model_2.compile(loss='categorical_crossentropy', optimizer=Adadelta(lr=1.0), metrics=['accuracy'])

# same as above but with different activation functions for the hidden layers
model_3 = Sequential()
model_3.add(Conv2D(32, kernel_size=(3, 3),
                   activation='sigmoid',
                   input_shape=(28, 28, 1)))
model_3.add(MaxPooling2D(pool_size=(4, 4)))
model_3.add(Dropout(0.5))
model_3.add(Flatten())
model_3.add(Dense(128, activation='sigmoid'))
model_3.add(Dropout(0.25))
model_3.add(Dense(10, activation='softmax'))
model_3.compile(loss='categorical_crossentropy', optimizer=Adadelta(lr=1.0), metrics=['accuracy'])

# same as model_2 but with more dense layers after conv
model_4 = Sequential()
model_4.add(Conv2D(32, kernel_size=(3, 3),
                   activation='relu',
                   input_shape=(28, 28, 1)))
model_4.add(MaxPooling2D(pool_size=(4, 4)))
model_4.add(Dropout(0.5))
model_4.add(Flatten())
model_4.add(Dense(128, activation='relu'))
model_4.add(Dense(128, activation='relu'))
model_4.add(Dense(128, activation='relu'))
model_4.add(Dropout(0.25))
model_4.add(Dense(10, activation='softmax'))
model_4.compile(loss='categorical_crossentropy', optimizer=Adadelta(lr=1.0), metrics=['accuracy'])

# same as above but with more neurons in the dense layers
model_5 = Sequential()
model_5.add(Conv2D(32, kernel_size=(3, 3),
                   activation='relu',
                   input_shape=(28, 28, 1)))
model_5.add(MaxPooling2D(pool_size=(4, 4)))
model_5.add(Dropout(0.5))
model_5.add(Flatten())
model_5.add(Dense(256, activation='relu'))
model_5.add(Dense(256, activation='relu'))
model_5.add(Dense(256, activation='relu'))
model_5.add(Dropout(0.25))
model_5.add(Dense(10, activation='softmax'))
model_5.compile(loss='categorical_crossentropy', optimizer=Adadelta(lr=1.0), metrics=['accuracy'])

# CNN Example from Keras Documentation
model_6 = Sequential()
model_6.add(Conv2D(32, kernel_size=(3, 3),
                   activation='relu',
                   input_shape=(28, 28, 1)))
model_6.add(Conv2D(64, (3, 3), activation='relu'))
model_6.add(MaxPooling2D(pool_size=(2, 2)))
model_6.add(Dropout(0.25))
model_6.add(Flatten())
model_6.add(Dense(128, activation='relu'))
model_6.add(Dropout(0.5))
model_6.add(Dense(10, activation='softmax'))

model_6.compile(loss='categorical_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])

# Fully Connected Model
model_7 = Sequential()
model_7.add(Flatten(input_shape=(28, 28, 1)))
model_7.add(Dense(100, activation='sigmoid', bias_initializer='ones', use_bias=True))
model_7.add(Dense(100, activation='sigmoid', bias_initializer='ones', use_bias=True))
model_7.add(Dense(100, activation='sigmoid', bias_initializer='ones', use_bias=True))
model_7.add(Dense(10, activation='softmax', bias_initializer='ones', use_bias=True))
model_7.compile(loss='categorical_crossentropy', optimizer=Adamax(lr=0.001), metrics=['accuracy'])

models = [model_0, model_1, model_2, model_3, model_4, model_5, model_6, model_7]

best_validation = 0
best_model = -1
for i in range(len(models)):
    print("_" * 50)
    print("\nModel_%d" % (i))
    model = models[i]

    history = model.fit(X_train, y_train,
                        batch_size=128,
                        epochs=12,
                        verbose=0,  # silent mode to reduce the amount of prints in the report
                        validation_data=(X_validation, y_validation))
    score = model.evaluate(X_validation, y_validation, verbose=0)
    print('Validation loss:', score[0])
    print('Validation accuracy:', score[1])
    if score[1] > best_validation:
        best_validation = score[1]
        best_model = i

    # collect and plot the losses for the training and validation sets
    accuracy = history.history['acc']
    loss = history.history['loss']
    val_accuracy = history.history['val_acc']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(14, 6))
    plt.plot(losses_train, label="train {}".format(i))
    plt.plot(losses_val, label="validation {}".format(i))
    plt.title('Courbes d\'apprentissage Loss / Epochs')
    plt.ylabel('Log de vraisemblance négative')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.show()

    plt.figure()
    plt.plot(accuracy, label="train {}".format(i))
    plt.plot(val_accuracy, label="validation {}".format(i))
    plt.title('Courbes d\'apprentissage Accuracy / Epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.show()

print("Best model is : model_%d" % (best_model))
score = models[best_model].evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
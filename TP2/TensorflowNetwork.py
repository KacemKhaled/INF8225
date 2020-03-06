import time
from tensorflow import keras
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.datasets import fashion_mnist
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# import the dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# shuffle the dataset
X_train, y_train = shuffle(X_train, y_train)

# pre-process: divide the "data" by 255 to shrink its size and keep it between [0-1]
X_train = X_train/255.0
X_test = X_test/255.0

# convert class vectors to binary class matrices
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# split test data into test and validation
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# set up the DNN architecture
model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(300, activation='relu', bias_initializer='ones' , use_bias=True))
model.add(Dense(300, activation='relu', bias_initializer='ones' , use_bias=True))
model.add(Dense(300, activation='relu', bias_initializer='ones' , use_bias=True))
model.add(Dense(10, activation='softmax', bias_initializer='ones' , use_bias=True))

# print the model summary
model.summary()

# set up parameters for the model
model.compile(optimizer=SGD(lr=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

# train the model, while collecting the loss calculated for each epoch
start_time = time.time()
history = model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_validation, y_validation))
end_time = time.time()
# test the model and store the final loss and accuracy results
val_loss, val_acc = model.evaluate(X_validation, y_validation)
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Loss: {}".format(test_loss))
print("Test Accuracy: {}".format(test_acc))

training_time_keras = end_time - start_time
print("Training Time = %.3f seconds" % (training_time_keras))

# collect and plot the losses for the training and validation sets
accuracy = history.history['acc']
loss = history.history[ 'loss']
val_accuracy = history.history['val_acc']
val_loss = history.history['val_loss']

plt.figure(figsize=(14, 6))
plt.plot(loss, label="train")
plt.plot(val_loss, label="validation")
plt.title('Courbes d\'apprentissage Loss / Epochs')
plt.ylabel('Log de vraisemblance négative')
plt.xlabel('Epoch')
plt.legend(loc='best')

plt.figure()
plt.plot(accuracy, label="train")
plt.plot(val_accuracy, label="validation")
plt.title('Courbes d\'apprentissage Accuracy / Epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='best')

plt.show()
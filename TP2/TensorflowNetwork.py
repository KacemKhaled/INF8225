import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# import the dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# pre-process: divide the "data" by 255 to shrink its size and keep it between [0-1]
train_images = train_images/255.0
test_images = test_images/255.0

# split test data into test and validation
test_images, val_images, test_labels, val_labels = train_test_split(test_images, test_labels, test_size=0.5, random_state=42)

# set up the DNN architecture
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

# print the model summary
model.summary()

# set up parameters for the model
adam = keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=adam, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# train the model, while collecting the loss calculated for each epoch
history = model.fit(train_images, train_labels, batch_size=32, epochs=20, validation_data=(val_images, val_labels))

# test the model and store the final loss and accuracy results
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Evaluation Loss: {}".format(test_loss))
print("Evaluation Accuracy: {}".format(test_acc))

# collect and plot the losses for the training and validation sets
losses_train = history.history['loss']
losses_val = history.history['val_loss']

plt.figure(figsize=(14, 6))
plt.plot(losses_train, label="train")
plt.plot(losses_val, label="validation")
plt.title('Courbes d\'apprentissage')
plt.ylabel('Log de vraisemblance négative')
plt.xlabel('Epoch')
plt.legend(loc='best')

plt.show()
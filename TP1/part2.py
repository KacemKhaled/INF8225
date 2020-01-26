import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

X = digits.data

y = digits.target
y_one_hot = np.zeros((y.shape[0], len(np.unique(y))))
y_one_hot[np.arange(y.shape[0]), y] = 1  # one hot target or shape NxK

X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.3, random_state=42)

X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

W = np.random.normal(0, 0.01, (len(np.unique(y)), X.shape[1]))  # weights of shape KxL

best_W = None
best_accuracy = 0
lr = 0.001
nb_epochs = 50
minibatch_size = len(y) // 20

losses = []
accuracies = []


def softmax(x):
    # assurez vous que la fonction est numeriquement stable
    # e.g. softmax(np.array([1000, 10000, 100000], ndim=2))
    pass


def get_accuracy(X, y, W):
    pass


def get_grads(y, y_pred, X):
    pass


def get_loss(y, y_pred):
    pass


for epoch in range(nb_epochs):
    loss = 0
    accuracy = 0
    for i in range(0, X_train.shape[0], minibatch_size):
        pass  # TODO
    losses.append(loss)  # compute the loss on the train set
    accuracy = None  # TODO
    accuracies.append(accuracy)  # compute the accuracy on the validation set
    if accuracy > best_accuracy:
        pass  # select the best parameters based on the validation accuracy

accuracy_on_unseen_data = get_accuracy(X_test, y_test, best_W)
print(accuracy_on_unseen_data)  # 0.897506925208

plt.plot(losses)

plt.imshow(best_W[4, :].reshape(8, 8))

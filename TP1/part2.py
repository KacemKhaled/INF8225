import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn import datasets
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

X = digits.data
X_bias = np.concatenate((X,np.ones((X.shape[0], 1))), axis=1)

y = digits.target
y_one_hot = np.zeros((y.shape[0], len(np.unique(y))))
y_one_hot[np.arange(y.shape[0]), y] = 1  # one hot target or shape NxK

X_train, X_test, y_train, y_test = train_test_split(X_bias, y_one_hot, test_size=0.3, random_state=42)

X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

def softmax(x):
    expZ = np.exp(x-np.max(x, axis=1, keepdims=True))
    return np.divide(expZ, np.sum(expZ, axis=1, keepdims=True))

def get_accuracy(X, y, W):
    return np.sum(np.argmax(softmax(X @ W.T), axis=1) == np.argmax(y, axis=1)) / len(y)


def get_grads(y, y_pred, X):
    #return ( y.T - y_pred.T ) @ X # y.T @ X - y_pred.T @ X
    #return np.matmul(y, X.T) - np.matmul(y_pred, X.T)
    return (1/len(y)) * np.dot( (y- y_pred).T, X)


def get_loss(y, y_pred):
    eps=1.0e-5
    y_pred = np.clip(y_pred, eps, 1 - eps) # to prevent dividing by zero
    return (-1 / len(y)) * np.sum(y * np.log(y_pred))

def fit(X_train,y_train,lr,minibatch_size):
    # generate matrix of weights
    W = np.random.normal(0, 0.01, (len(np.unique(y)), X.shape[1]))  # weights of shape KxL
    Theta = np.random.normal(0, 0.01, (len(np.unique(y)), X.shape[1]+1))  # weights of shape KxL+1

    best_W = None
    best_theta = None
    best_accuracy = 0
    nb_epochs = 50
    losses_train = []
    losses_val = []
    accuracies = []
    for epoch in range(nb_epochs):
        loss = 0
        accuracy = 0
        grads = []
        #X_prime=np.append(X_train,[1])
        #Theta=np.append(W, np.ones((10,1)), axis=1)#Theta=[W b] de dimension (10,65)
        #prob_act=softmax(np.dot(Theta,X_prime),axis=0)#y_pred
        for i in range(0, X_train.shape[0], minibatch_size):
            X_train_mini = X_train[i:i + minibatch_size] if (i + minibatch_size < X_train.shape[0]) \
                else X_train[i:X_train.shape[0]]
            y_train_mini = y_train[i:i + minibatch_size] if (i + minibatch_size < X_train.shape[0]) \
                else y_train[i:X_train.shape[0]]
            # feedforward
            y_pred = softmax(np.dot(X_train_mini, Theta.T)) # (y_train_mini * Theta * X_train_mini.T)
            Theta = Theta + lr * get_grads(y_train_mini, y_pred, X_train_mini)
            #W = W + lr * get_grads(y_train_mini, y_pred, X_train_mini)
        # compute the loss on the train set
        loss = get_loss(y_train, softmax(np.dot(X_train, Theta.T)))
        losses_train.append(loss)
        # compute the loss on the validation set
        loss = get_loss(y_validation,softmax(np.dot(X_validation, Theta.T)))
        losses_val.append(loss)
        # compute the accuracy on the validation set
        accuracy = get_accuracy(X_validation,y_validation,Theta)
        accuracies.append(accuracy)
        if accuracy > best_accuracy:
            # select the best parameters based on the validation accuracy
            best_accuracy = accuracy
            best_W = W
            best_theta = Theta
    accuracy_on_unseen_data = get_accuracy(X_test, y_test, best_theta) # get_accuracy(X_test, y_test, best_W)
    print("Accuracy on validation data:{:.10f}\n".format(best_accuracy))
    print("Accuracy on unseen data:{:.10f}\n".format(accuracy_on_unseen_data)) # 0.897506925208
    return losses_train,losses_val, best_theta, best_accuracy

#lr = 0.001
#minibatch_size = len(y) // 20
test_nb = 0

chosen_accuracy = 0

lrs = [0.1, 0.01, 0.001]
minibatch_sizes = [len(y) // 20, 1, 20, 200, 1000]
for lr in lrs:
  for minibatch_size in minibatch_sizes:
    test_nb += 1
    print("Test Number #{},\t lr: {},\t minibatch_size: {}\n".format(test_nb,lr,minibatch_size))
    plt.figure(lrs.index(lr),figsize = (14,6))
    gridspec.GridSpec(4,8)
    losses_train, losses_val, best_theta, best_accuracy = fit(X_train,y_train,lr,minibatch_size)

    if best_accuracy > chosen_accuracy:
      chosen_accuracy = best_accuracy
      chosen_test = test_nb
      chosen_lr = lr
      chosen_minibatch_size = minibatch_size

    best_W = best_theta[:,:64]
    best_b = best_theta[:,64:]

    ax1 = plt.subplot2grid((4,8), (0,0), colspan=6, rowspan=4)
    ax1.plot(losses_train, label="train")
    ax1.plot(losses_val, label="validation")
    ax1.set_title('Courbes d\'apprentissage, lr ='+str(lr)+', minibatch_size ='+str(minibatch_size))
    ax1.set_ylabel('Log de vraisemblance négative')
    ax1.set_xlabel('Epoch')
    ax1.legend(loc='best')

    ax2 = plt.subplot2grid((4,8), (0,6),colspan=2, rowspan=3)
    ax2.imshow(best_W[4, :].reshape(8, 8))
    ax2.set_title('Poids W appris pour le chiffre 4')

    ax3 = plt.subplot2grid((4,8), (3,6),colspan=2, rowspan=1)
    ax3.imshow(best_b[4, :].reshape(1,1))
    ax3.set_title('Poids b appris pour le chiffre 4')

    plt.show()
print('\n\nChosen Test #{}\n\
Chosen Accuracy = {:.10f}\n\
Chosen learning rate = {}\n\
Chosen minibatch size = {}\n'.format(chosen_test,chosen_accuracy,chosen_lr,chosen_minibatch_size))

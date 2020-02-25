import numpy as np
import sys

class NeuralNetwork:

    def __init__(self,
                 no_of_in_nodes,
                 no_of_out_nodes,
                 no_of_hidden_nodes,
                 no_of_hidden_layers):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.no_of_hidden_layers = no_of_hidden_layers
        self.Teta = self.create_weight_matrices()

    # do not forget to change W to Teta
    def create_weight_matrices(self):
        # generate matrix of weights
        Teta = []
        print("self.no_of_hidden_layers " + str(self.no_of_hidden_layers))
        for i in range(0, self.no_of_hidden_layers + 1):
            if i == 0:
                input_nodes = self.no_of_in_nodes
                output_nodes = self.no_of_hidden_nodes
            elif i == self.no_of_hidden_layers:
                input_nodes = self.no_of_hidden_nodes
                output_nodes = self.no_of_out_nodes
            else:
                input_nodes = self.no_of_hidden_nodes
                output_nodes = self.no_of_hidden_nodes
            print("layer #" + str(i))
            Teta_layer = np.random.normal(0, 0.01,
                                          (output_nodes, input_nodes + 1))  # we add +1 to take b into consideration
            print(Teta_layer.shape)
            Teta.append(Teta_layer)
        return Teta  # list of np arrays

    def softmax(self, X):
        x = np.copy(X)
        expZ = np.exp(x-np.max(x, axis=1, keepdims=True))
        return np.divide(expZ, np.sum(expZ, axis=1, keepdims=True))
        #expZ = np.exp(x - np.max(x))
        #return np.divide(expZ, np.sum(expZ))

    def get_grads_softmax(self, y, y_pred, X):
        y = np.atleast_2d(y)
        y_pred = np.atleast_2d(y_pred)
        X = np.atleast_2d(X)
        print(y_pred.T.shape)
        print(y.T.shape)
        print(X.shape)
        return (y - y_pred).T @ X # there was - here

    def softmax_backward(self, y, y_pred):
        return (y - y_pred) # there was - here

    def relu(self, X):
        x = np.copy(X)
        return np.maximum(x,0)

    def relu_backward(self, X):
        x = np.copy(X)
        x[x <= 0] = 0
        x[x > 0] = 1
        # a = np.diagflat(x)
        return x

    def get_loss(self, Y, Y_pred):
        y = Y
        y_pred = Y_pred
        eps = 1.0e-5
        y_pred = np.clip(y_pred, eps, 1 - eps)  # to prevent dividing by zero
        return (-1 / len(y)) * np.sum(y * np.log(y_pred))

    def get_accuracy(self, Y_pred, Y):
        y_pred = np.copy(Y_pred)
        y = np.copy(Y)

        return np.sum(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)) / len(y)


    def predict(self, X):
        # feedforward for all the exaples and predict the output
        L = self.no_of_hidden_layers
        Teta = self.Teta

        # forward pass for each example
        node_a_prev = np.copy(X)  # item j from the input vector of the example
        for l in range(1, L + 2):
            node_in = np.dot(node_a_prev, Teta[l - 1].T)  # in_i <- sum_j{W_j_i * a_j}
            if l < L + 1:
                # the hidden layers
                node_a_cur = self.relu(node_in)  # relu
                node_a_cur = np.concatenate((node_a_cur, np.ones((node_a_cur.shape[0], 1))), axis=1)  # add the bias b
            else:
                # the output layer
                node_a_cur = self.softmax(node_in)
            node_a_prev = node_a_cur
        return node_a_cur

    def progress(self,i,n):
        sys.stdout.write('\r')
        # just a fancy progress bar:
        sys.stdout.write("[%-100s] %.1f%%" % ('=' * int(i*100/n), (100/n)*i))
        sys.stdout.flush()

    def train(self, X_train, y_train, X_validation, y_validation, X_test, y_test, lr, nb_epochs,minibatch_size):
        # neural network with L hidden layers

        L = self.no_of_hidden_layers
        Teta = self.Teta
        best_teta = None
        best_accuracy = 0
        losses_train = []
        losses_val = []
        accuracies = []

        for epoch in range(nb_epochs):

            loss = 0
            accuracy = 0
            print("\n\nepoch : "+str(epoch+1)+" / "+str(nb_epochs))
            #y_train_pred = []
            y_validation_pred = []

            # for i in range(len(X_train[:100])): # range(len(X_train))
            for i in range(0, X_train.shape[0], minibatch_size):
                X_train_mini = X_train[i:i + minibatch_size] if (i + minibatch_size < X_train.shape[0]) \
                    else X_train[i:X_train.shape[0]]
                y_train_mini = y_train[i:i + minibatch_size] if (i + minibatch_size < X_train.shape[0]) \
                    else y_train[i:X_train.shape[0]]

                self.progress(i,len(X_train))
                # forward pass for each example
                node_in = [None] * (L + 2) # [[] for i in range(L + 2)]
                node_a = [None] * (L + 2)
                node_delta = [None] * (L + 2)
                node_a[0] = X_train_mini # item j from the input vector of the example
                for l in range(1, L + 2):
                    node_in[l] = node_a[l - 1] @ Teta[l - 1].T  # in_i <- sum_j{W_j_i * a_j}
                    if l < L + 1:
                        # the hidden layers
                        node_a[l] = self.relu(node_in[l]) # relu
                        node_a[l] = np.concatenate((node_a[l], np.ones((node_a[l].shape[0], 1))),
                                                    axis=1)  # add the bias b to use it in the next layer
                    else:
                        # the output layer
                        node_a[l] = self.softmax(node_in[l])

                # backpropagation
                y = y_train_mini
                node_delta[L + 1] = - self.softmax_backward(y, node_a[L + 1])
                for l in range(L, -1, -1):
                    node_delta[l] = - self.relu_backward(node_a[l][:,:-1]) * (node_delta[l + 1] @ Teta[l][:, :-1] )  # W.T @ node_delta[l + 1]
                    # update parameters
                    Teta[l] = Teta[l] - lr * (node_delta[l + 1].T @ node_a[l] )
                #y_train_pred.append(node_a[L+1])
            # compute the loss on the train set
            #y_train_pred = np.asarray(y_train_pred).squeeze()
            self.Teta = Teta

            y_train_pred = self.predict(X_train)
            loss = self.get_loss(y_train, y_train_pred)
            losses_train.append(loss)

            # compute the loss on the validation set
            y_validation_pred = self.predict(X_validation)
            loss = self.get_loss(y_validation, y_validation_pred)
            losses_val.append(loss)

            # compute the accuracy on the validation set
            accuracy = self.get_accuracy(y_validation_pred, y_validation)
            accuracies.append(accuracy)
            print("\nAccuracy on validation data:{:.10f}\n".format(accuracy))

            if accuracy > best_accuracy:
                # select the best parameters based on the validation accuracy
                best_accuracy = accuracy
                best_teta = Teta

        # accuracy_on_unseen_data = get_accuracy(X_test, y_test, best_teta) # get_accuracy(X_test, y_test, best_W)
        print("\Best Accuracy on validation data:{:.10f}\n".format(best_accuracy))
        # print("Accuracy on unseen data:{:.10f}\n".format(accuracy_on_unseen_data)) # 0.897506925208
        return losses_train, losses_val, best_teta, best_accuracy

    def run(self):
        pass


from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def one_hot(y, K):
    y_one_hot = np.zeros((y.shape[0], K))  # we have K classes, the general form is len(np.unique(y)
    y_one_hot[np.arange(y.shape[0]), y] = 1  # one hot target or shape NxK
    return y_one_hot


((X_train, y_train), (X_test, y_test)) = fashion_mnist.load_data()

X_train = X_train.reshape((X_train.shape[0], 28 ** 2))
X_test = X_test.reshape((X_test.shape[0], 28 ** 2))

# scale data to the range of [0, 1]
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
# append bias to X
X_train = np.concatenate((X_train,np.ones((X_train.shape[0], 1))), axis=1)
X_test = np.concatenate((X_test,np.ones((X_test.shape[0], 1))), axis=1)
# one-hot encode the training and testing labels
y_train = one_hot(y_train, 10)
y_test = one_hot(y_test, 10)

"""
X_train = train_data.train_data
y_train = one_hot(train_data.train_labels)

X_validation = valid_data.train_data
y_validation = one_hot(valid_data.train_labels)

X_test = valid_data.test_data
y_test = one_hot(valid_data.test_labels)
"""
X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

NN = NeuralNetwork(no_of_in_nodes=784, no_of_out_nodes=10, no_of_hidden_nodes=300, no_of_hidden_layers=1)

losses_train, losses_val, best_teta, best_accuracy = NN.train(X_train[:100], y_train[:100],
                                                              X_validation, y_validation,
                                                              X_test, y_test,
                                                              lr=0.1, nb_epochs=50, minibatch_size=32)

plt.figure(figsize=(14, 6))

plt.plot(losses_train, label="train")
plt.plot(losses_val, label="validation")
plt.title('Courbes d\'apprentissage')
plt.ylabel('Log de vraisemblance négative')
plt.xlabel('Epoch')
plt.legend(loc='best')

plt.show()

# ADAM (the same function used in the Jupyter Notebook, test and validation data are already there)
def fit_ADAM(X_train,y_train):
  # initialiser les paramètres de l'algorithme avec celles mentionnées dans le PAPIER
  alpha = 0.01
  beta_1 = 0.9
  beta_2 = 0.999
  epsilon = 1e-8

  m_t = 0
  v_t = 0 
  t = 0

  Theta = np.random.normal(0, 0.01, (len(np.unique(y)), X.shape[1]+1))  # weights of shape KxL+1 

  best_theta = None
  best_accuracy = 0
  nb_epochs = 50
  losses_train = []
  losses_val = []
  accuracies = []

  for epoch in range(nb_epochs):
      loss = 0
      accuracy = 0
      
      t+=1
      y_pred = softmax(np.dot(X_train, Theta.T)) 
      # calcul du gradient de la fonction objective
      g_t = get_grads(y_train, y_pred, X_train)	
      # mise à jour des moyennes mobiles du gradient	
      m_t = beta_1*m_t + (1-beta_1)* g_t
      # mise à jour des moyennes mobiles du carré du gradient	
      v_t = beta_2*v_t + (1-beta_2)* (g_t * g_t)	
      # correction des moyennes mobiles
      m_t_c = m_t/(1-(beta_1**t))	
      v_t_c = v_t/(1-(beta_2**t))	
      # mise à jour des poids Theta
      Theta = Theta - (alpha*m_t_c)/(np.sqrt(v_t_c)+epsilon)

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
          #best_W = W
          best_theta = Theta
  accuracy_on_unseen_data = get_accuracy(X_test, y_test, best_theta)
  print("Accuracy on validation data:{:.10f}\n".format(best_accuracy))
  print("Accuracy on unseen data:{:.10f}\n".format(accuracy_on_unseen_data)) 
  return losses_train,losses_val, best_theta, accuracy_on_unseen_data
  
plt.figure(figsize = (14,6))
gridspec.GridSpec(4,8)
losses_train_ADAM, losses_val_ADAM, best_theta, accuracy_on_unseen_data_ADAM = fit_ADAM(X_train,y_train)

best_W = best_theta[:,:64]
best_b = best_theta[:,64:]


ax1 = plt.subplot2grid((4,8), (0,0), colspan=6, rowspan=4)
ax1.plot(losses_train_ADAM, label="train")
ax1.plot(losses_val_ADAM, label="validation")
ax1.set_title('Courbes d\'apprentissage en utilisant Adam')
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
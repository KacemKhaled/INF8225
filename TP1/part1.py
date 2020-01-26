import numpy as np
import matplotlib.pyplot as plt

# les arrays sont batis avec les dimensions suivantes:
# pluie, arroseur, watson, holmes
# et chaque dimension: faux, vrai

prob_pluie = np.array([0.8, 0.2]).reshape(2, 1, 1, 1)
print("Pr(Pluie)={}\n".format(np.squeeze(prob_pluie)))
prob_arroseur = np.array([0.9, 0.1]).reshape(1, 2, 1, 1)
print("Pr(Arroseur)={}\n".format(np.squeeze(prob_arroseur)))
watson = np.array([[0.8, 0.2], [0, 1]]).reshape(2, 1, 2, 1)
print("Pr(Watson|Pluie)={}\n".format(np.squeeze(watson)))
holmes = np.array([[1, 0],[0.1, 0.9],[0,1], [0, 1]]).reshape(2, 2, 1, 2) #TODO --> Done
print("Pr(Holmes|Pluie,arroseur)={}\n".format(np.squeeze(holmes)))
# prob watson mouille - pluie
print("Pr(W = 1|P = 0) = {}\n".format(np.squeeze(watson[0,:,1,:])))
# prob gazon watson mouille
print("Pr(W = 1) = {:.3f}\n".format(((watson * prob_pluie).sum(0).squeeze()[1])))
# prob gazon holmes mouille si arroseur - pluie
print("Pr(H = 1|A = 1, P = 0) = {}\n".format(holmes[0,1,0,1]))
# prob gazon holmes mouille
print("a) Pr(H = 1) = {:.3f}\n".format(
    (prob_pluie * prob_arroseur * holmes)[:,:,:,1].sum()))
print("b) Pr(H = 1|W = 1) = {:.3f}\n".format(
    ((prob_pluie * prob_arroseur * watson * holmes)[:,:,1,1].sum()) /
    ((prob_pluie * prob_arroseur * watson * holmes)[:,:,1,:].sum())
))
print("c) Pr(H = 1|W = 0) = {:.3f}\n".format(
    ((prob_pluie * prob_arroseur * watson * holmes)[:,:,0,1].sum()) /
    ((prob_pluie * prob_arroseur * watson * holmes)[:,:,0,:].sum())
))
print("d) Pr(H = 1|P = 0, W = 1) = {:.3f}\n".format(
    ((prob_pluie * prob_arroseur * watson * holmes)[0,:,1,1].sum()) /
    ((prob_pluie * prob_arroseur * watson * holmes)[0,:,1,:].sum())
))
print("e) Pr(W = 1|H = 1) = {:.3f}\n".format(
    ((prob_pluie * prob_arroseur * watson * holmes)[:,:,1,1].sum()) /
    ((prob_pluie * prob_arroseur * watson * holmes)[:,:,:,1].sum())
))
print("f) Pr(W = 1|H = 1, A = 1) = {:.3f}\n".format(
    ((prob_pluie * prob_arroseur * watson * holmes)[:,1,1,1].sum()) /
    ((prob_pluie * prob_arroseur * watson * holmes)[:,1,:,1].sum())
))

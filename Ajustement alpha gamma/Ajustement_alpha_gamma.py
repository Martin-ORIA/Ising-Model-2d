import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd



Ls = [k for k in range(20, 61)]
Ls += [65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 120, 140, 200]#liste des tailles de systèmes (fichiers contenus dans le dossier "données principales")
                                                                # faire attention que les données soient bien dans le meme dossier


Ps = np.ones(len(Ls), dtype=int)*10   #décalage pour l'ajustement de chaque taille


L_Tc = []
L_X_max = []
L_X_err = []
for L in range(len(Ls)): #extraction du maximum pour chaque taille L
    data = pd.read_csv("dataL=" + str(Ls[L]) + "_T=200_n_mc=500000.csv")
    L_Temp = np.array(data["L_Temp"])
    L_X = np.array(data["L_C"])            #remplacer par "L_X" pour la susceptibilité
    L_Xerr = np.array(data["L_Cerr"])      # remplacer par "L_Xerr" pour la susceptibilité
    L_X_max.append(max(L_X))
    Tmax = np.argmax(L_X[2:])
    Tincer = np.argmax(L_X)
    L_X_err.append(L_Xerr[Tincer])




L_L = np.array(Ls)


def power_law(L, BetaC, c, x=0):
    """fonction de puissance avec BetaC la temperature de curie"""
    return c*L**x + BetaC


sol,cov = curve_fit(power_law, L_L, L_X_max, maxfev=int(1e7))

print(sol[0], sol[1], sol[2])
print(cov)

fig, ax = plt.subplots(figsize=(13, 8))



plt.scatter(L_L, L_X_max, s=4, color="red", label="Données")
#plt.errorbar(L_L, L_X_max, yerr=L_X_err, linewidth=0.5, color="black", ls="None", label="Err")

plt.plot(L_L, power_law(L_L, sol[0], sol[1], sol[2]), color="black", linewidth=1, label="Ajustement")
plt.xlabel("log(L) Taille")
plt.ylabel("log(X_max) Chaleur spécifique")  #remplacer par Chi si on s'intéresse à la susceptibilité

plt.title("Exposant critique α= {} +- {}".format(np.round(sol[2], 4), np.round(np.sqrt(np.diag(cov))[2], 2)))  #affiche l'exposant critique associé
plt.xscale("log")
plt.yscale("log")
plt.legend()

plt.show()

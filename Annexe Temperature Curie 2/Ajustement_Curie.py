import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

def function(T, k=0., gamma=0., Tc=0.):  #fonction du comportement critique. Le paramètre extrait est Tc pour une taille L donnée
    return k * np.abs((T-Tc))**gamma 



Ls = [k for k in range(20, 61)]
Ls += [65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 120, 140, 200]#liste des tailles de systèmes (fichiers contenus dans le dossier "données principales")
                                                                # faire attention que les données soient bien dans le meme dossier


Ps = np.ones(len(Ls), dtype=int)*13   #décalage pour l'ajustement de chaque taille


L_Tc = []

for L in range(len(Ls)): #réalise l'ajustement sur len(Ls) valeurs de L différentes
    data = pd.read_csv("dataL=" + str(Ls[L]) + "_T=200_n_mc=500000.csv")
    L_Temp = np.array(data["L_Temp"])
    L_X = np.array(data["L_X"])           
    Tmax = np.argmax(L_X[2:])
    sol,cov = curve_fit(function, L_Temp[Tmax+Ps[L]:], L_X[Tmax+Ps[L]:], maxfev=int(1e6))  #ajustement


    Ks = sol[0]
    gammas = sol[1]
    TCs = sol[2]
    L_Tc.append(TCs)
    print("Tc :",TCs, "gamma : ", gammas)
    if L%12 == 0:  #permet d'obtenir un graphique lisible sans avoir trop de données
        plt.scatter(L_Temp, L_X, label="L = " + str(Ls[L]), s=1)
        plt.plot(L_Temp[Tmax+Ps[L]:], function(L_Temp, Ks, gammas, TCs)[Tmax+Ps[L]:])
        plt.xlabel("T")
        plt.ylabel("susceptibilité")
        plt.title("Ajustement de la susceptibilité")
        plt.legend()



L_L = np.array(Ls)

L_Tc = np.array(L_Tc)



def power_law(L, BetaC=2., c=1, x=1):
    """fonction puissance pour l'ajustement"""
    return c*L**x + BetaC


sol,cov = curve_fit(power_law, 1/L_L, L_Tc, maxfev=int(1e7))   #ajustement

print(sol[0], sol[1], sol[2])
print(cov)

fig, ax = plt.subplots(figsize=(13, 8))



plt.scatter(1/L_L, L_Tc, s=4, color="red", label="Données")
plt.plot(1/L_L, power_law(1/L_L, sol[0], sol[1], sol[2]), color="black", linewidth=1, label="Ajustement")
plt.xlabel("1/L Taille")
plt.ylabel("log(T_L)")
plt.title("Température de Curie : T_C = {} +- {}".format(np.round(sol[0], 3), np.round(np.sqrt(np.diag(cov))[0], 3)))
plt.yscale("log")
plt.legend()

plt.show()

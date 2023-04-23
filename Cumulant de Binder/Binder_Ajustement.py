import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from multiprocessing import Pool
from itertools import repeat
import pandas as pd
from scipy import interpolate
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


Ls = [11, 13, 15, 16, 20, 40 ]  #liste des tailles des simulations

L_nmc = [500000]*len(Ls)  #liste des pas n_mc des simulations ici 500,000

fig1, ax1 = plt.subplots(dpi=100)


LU = np.zeros((1000000, len(Ls)))  #array contenant le cumulant de binder. La taille 1000000 est fixée par l'interpolation de la ligne 33.
L_c = ["orange", "red", "green", "blue"]
for L in range(len(Ls)):
	data = pd.read_csv("BINDERdataL=" + str(Ls[L]) + "_T=100_n_mc=" + str(L_nmc[L]) + ".csv")
	L_Temp = data["L_Temp"]
	TT = L_Temp[50]
	L_B2 =np.array(data["L_B2"])       #magnetisation puissance 2
	L_B4 = np.array(data["L_B4"])       #magnetisation puissance 4
	L_U = 1-(L_B4/(3*L_B2*L_B2))       #cumulant de binder
	
	f = interp1d(L_Temp, L_U, kind='linear')
	L_Temp = np.linspace(min(L_Temp), max(L_Temp), num=1000000)  #interpolation         

# interpolate the corresponding y-values
	L_U = f(L_Temp)
	LU[:, L] = L_U
#linewidth
	ax1.plot(L_Temp, L_U, linewidth=1, label=str(Ls[L]))
	ax1.set_xlabel("Temperature")
	ax1.set_ylabel("Cumulant Binder")




Lres = []   #liste contenant les points d'intersections

for i in range(LU.shape[1] -1):   #boucles ajoutant à Lres toute combinaison d'intersection des courbes
	L1 = LU[:, i]
	for k in range(1+i, LU.shape[1]):
		L2 = LU[:, k]
		idx = np.argwhere(np.diff(np.sign(L1 - L2))).flatten()
		Lres.append(L_Temp[idx[0]])

#AFFICHAGE DES DONNEES
print(Lres)
print("TC = ", np.mean(Lres), "Inc : ", np.std(Lres))
print("TC = ", 2/(np.log(1+np.sqrt(2))))
ax1.set_title("Cumulant de Binder et Température de curie")
ax1.vlines(np.mean(Lres), ymin=min(L_U), ymax=max(L_U), linestyle="--", linewidth=0.7, label="T_c ", color="black")
ax1.set_title("Cumulant de Binder Tc = {} +- {}".format(np.round(np.mean(Lres), 4), np.round(np.std(Lres), 3)))
plt.legend()
plt.show()
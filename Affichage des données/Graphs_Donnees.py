import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from multiprocessing import Pool
from itertools import repeat
import pandas as pd
from matplotlib.pyplot import figure





Ls = [10]  #Liste des tailles L à afficher (faire attention que le fichier sois dans le meme dossier), on peut par exemple avoir Ls = [20, 21, 50, 45]


L_c = ["orange", "red", "green", "blue"] #Liste des couleurs si plusieurs données doivent etre tracées
for L in range(len(Ls)):
	data = pd.read_csv("dataL=" + str(Ls[L]) + "_T=200_n_mc=1000.csv")  #modifier le 500000 (correspondant aux pas n_mc).Attention seuls des fichiers de données avec les meme paramètre (sauf la taille)
	                                                                      # peuvent etre tracés et comparés sur le meme graph
	L_Temp = data["L_Temp"]
	L_E =data["L_E"]
	L_Eerr = np.array(data["L_Eerr"])
	L_B = data["L_B"]
	L_Berr = np.array(data["L_Berr"]) #on pourra multiplier ceci par un facteur 10 ou 100 pour observer la variation des incertitudes si celles ci sont trop faibles
	L_C = data["L_C"]
	L_Cerr = data["L_Cerr"]
	L_X = data["L_X"]
	L_Xerr = np.array(data["L_Xerr"])
	L = Ls[L]
	N_err = 50   #rapport du nombre d'erreurs
	# N_err = 10   #rapport du nombre d'erreurs

	fig1, ax1 = plt.subplots()
	ax1.scatter(L_Temp, L_E, s=4, color="blue")
	ax1.vlines(2.26918, ymin=min(L_E), ymax=max(L_E), linestyle="--", linewidth=0.7, label="T_c", color="black")  #Ligne du point critique
	ax1.errorbar(L_Temp[::len(L_Temp)//N_err], L_E[::len(L_Temp)//N_err], yerr=L_Eerr[::len(L_Temp)//N_err]*(1/(L*L)), linewidth=0.5, color="black", ls="None", label="Err")
	ax1.set_xlabel("Temperature")
	ax1.set_ylabel("Energie")
	ax1.set_title("Energie totale (Err*100)")
	ax1.legend()

	fig2, ax2 = plt.subplots()
	ax2.scatter(L_Temp, L_B, s=4, color="red")
	ax2.vlines(2.26918, ymin=min(L_B), ymax=max(L_B), linestyle="--", linewidth=0.7, label="T_c", color="black")
	ax2.errorbar(L_Temp[::len(L_Temp)//N_err], L_B[::len(L_Temp)//N_err], yerr=L_Berr[::len(L_Temp)//N_err]*(1/(L*L)), linewidth=0.5, color="black", ls="None", label="Err")
	ax2.set_xlabel("Temperature")
	ax2.set_ylabel("Magnetisation")
	ax2.set_title("Magnetisation totale (Err*100)")
	ax2.legend()


	fig3, ax3 = plt.subplots()
	ax3.scatter(L_Temp, L_C, s=4, color="green")
	ax3.vlines(2.26918, ymin=min(L_C), ymax=max(L_C), linestyle="--", linewidth=0.7, label="T_c", color="black")
	ax3.errorbar(L_Temp[::len(L_Temp)//N_err], L_C[::len(L_Temp)//N_err], yerr=L_Cerr[::len(L_Temp)//N_err]*(1/(L*L)), linewidth=0.5, color="black", ls="None", label="Err")
	ax3.set_xlabel("Temperature")
	ax3.set_ylabel("chaleur spécifique")
	ax3.set_title("chaleur spécifique (Err*1)")
	ax3.legend()

	fig3, ax3 = plt.subplots()
	ax3.scatter(L_Temp, L_X, s=4, color="black")
	ax3.vlines(2.26918, ymin=min(L_X), ymax=max(L_X), linestyle="--", linewidth=0.7, label="T_c", color="black")
	ax3.errorbar(L_Temp[::len(L_Temp)//N_err], L_X[::len(L_Temp)//N_err], yerr=L_Xerr[::len(L_Temp)//N_err]*(1/(L*L)), linewidth=0.5, color="black", ls="None", label="Err")
	ax3.set_xlabel("Temperature")
	ax3.set_ylabel("susceptibilité")
	ax3.set_title("susceptibilité (Err*10)")
	ax3.legend()
	plt.show()


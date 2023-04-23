# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 00:07:33 2023

@author: marti
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from multiprocessing import Pool
from itertools import repeat
import pandas as pd
import sys
from numba import jit, prange
##### CONSTANTES #####

np.set_printoptions(threshold=sys.maxsize)


@jit(nopython=True, boundscheck=False, fastmath=True)
def roll_hor(arr, k=1):
    """implémentation numba de la fonction numpy non compatible np.roll selon l'axe horizontal"""
    return np.concatenate((arr[:,-k:], arr[:,:-k]), axis=1)

@jit(nopython=True, boundscheck=False, fastmath=True)
def roll_vert(arr, k=1):
    """implémentation numba de la fonction numpy non compatible np.roll selon l'axe vertical"""
    return np.concatenate((arr[-k:,:], arr[:-k,:]), axis=0)

@jit(nopython=True, boundscheck=False, fastmath=True)
def Energie(S:np.array) -> np.float64:
    """retourne l'energie de la matrice S"""
    return (-np.sum(S*(roll_vert(S,1) + roll_hor(S,1) + roll_vert(S,-1) + roll_hor(S,-1))))/4

@jit(nopython=True, boundscheck=False, fastmath=True)
def Magnetisation(S:np.array) -> np.float64:
    """retourne la magnetisation de la matrice S"""
    return np.abs(np.sum(S))




@jit(nopython=True, boundscheck=False, fastmath=True)
def Wolf_Loop(S:np.array, L:np.int64, B:np.int64, T:np.int64, P:np.float64) -> np.array:
	"""
	Cette fonction réalise un step de monte carlo
	L est la taille de la matrice
	B est le champ magnétique
	T est la température                                    
	P la probabilitée 
	"""
	Cluster = [[random.randint(0, L-1), random.randint(0, L-1)]]
	for i,j in Cluster:
		
		#i, j = item
		voisins = np.array([[(i - 1)%L, j], [(i + 1)%L, j], [i, (j - 1)%L], [i, (j + 1)%L]])
		Candidats = []

		for v_i, v_j in voisins:
			#v_i, v_j = vois
			if S[v_i, v_j] == S[i, j] and [v_i, v_j] not in Cluster and np.random.uniform(0., 1.) < P:
				Candidats.append([v_i, v_j])
		Cluster += Candidats
	for i, j in Cluster:
		S[i, j] = -S[i, j]
        
	return S




@jit(nopython=True, boundscheck=False, fastmath=True)
def Main_Loop(T:np.int64, L:np.int64, B:np.int64, n_eq:np.int64, n_mc:np.int64, L_Temp:np.array) -> np.array:
    """Boucle principale
    T indice de la température
    L taille du système
    n_eq : pas d'equillibre
    n_mc : pas de monte carlo
    L_Temp liste de températures
    Retourne un array contenant la temperature, l'energie, l'energie au carré, la magnetisation, la magnetisation moyenne au carré et leurs incertitudes"""
    S = np.random.choice(np.array([-1, 1]), size=(L, L))
    E = np.zeros(n_mc)
    M = np.zeros(n_mc)
    P = (np.float64(1) - np.exp(-2/L_Temp[T]))
    print("")
    for iter_eq in range(n_eq): #pas d'equillibre
        S = Wolf_Loop(S, L, B, L_Temp[T], P)
        
    for iter_main in range(n_mc):    #pas de wolff
        S = Wolf_Loop(S, L, B, L_Temp[T], P)
        E[iter_main] = Energie(S)
        M[iter_main] = Magnetisation(S)

    return np.array([L_Temp[T], np.mean(E), np.mean(E*E), np.mean(M), np.mean(M*M), np.std(E), np.std(E*E), np.std(M), np.std(M*M)])


if __name__ == "__main__":
    L = 10       #taille du système
    B = 0        #Champ externe (peut etre un array)
    n_eq = 100   #pas d'equilibre
    n_mc = 1000   #pas de wolf
    L_Temp = np.linspace(1, 4, 200)   #liste de températures

    t1 = time.time()
    with Pool() as pool:  #multiprocessing
        T_arg = [T for T in range(len(L_Temp))]
        total = pool.starmap(Main_Loop, zip(T_arg, repeat(L), repeat(B), repeat(n_eq), repeat(n_mc), repeat(L_Temp)))
    t2 = time.time()
    print(t2-t1)

    total = np.array(total)

    # création des données
    L_E, L_Eerr = total[ : ,1]*(1/(L*L)), total[:,5]
    L_B, L_Berr = total[ : ,3]*(1/(L*L)), total[:,7]

    #Calcul de L_Cerr
    deltaC1 = total[:, 6]*(1/(L*L))
    deltaC2 = (L_E*L_Eerr*np.sqrt(2))/(L*L)
    L_C, L_Cerr = ((1/(L*L))*total[ : ,2]-(1/(L*L))*(total[ : ,1]*total[ : ,1]))*(1/(L_Temp*L_Temp)), (1/(L_Temp*L_Temp))*np.sqrt(deltaC1**2 + deltaC2**2)

    #Calcul de L_Xerr
    deltaX1 = total[:, 8]*(1/(L*L))
    deltaX2 = (L_B*L_Berr*np.sqrt(2))/(L*L)
    L_X, L_Xerr = ((1/(L*L))*total[ : ,4]-(1/(L*L))*(total[ : ,3]*total[ : ,3]))*(1/L_Temp), (1/(L_Temp))*np.sqrt(deltaX1**2 + deltaX2**2)

    L_L = np.ones(len(L_E))*L
    L_Bext = np.ones(len(L_E))*B
    L_neq = np.ones(len(L_E))*n_eq
    L_n_mc = np.ones(len(L_E))*n_mc  
    L_t = np.ones(len(L_E))*(t2-t1)
    
    #Création du fichier CSV
    dataset = True
    if dataset == True:
        data = {
        'L_Temp': L_Temp,
        'L_E': L_E,
        'L_Eerr': L_Eerr,
        'L_B': L_B,
        'L_Berr': L_Berr,
        'L_C': L_C,
        'L_Cerr': L_Cerr,
        'L_X': L_X,
        'L_Xerr': L_Xerr,
        'L_L': L_L,
        'L_Bext': L_Bext,
        'L_neq': L_neq,
        'L_n_mc': L_n_mc,
        'L_t': L_t
        }
        df = pd.DataFrame(data)
    
        # Save the DataFrame as a CSV file
        df.to_csv('dataL=' + str(L) + '_T=' + str(len(L_Temp)) + '_n_mc=' + str(n_mc) + '.csv', index=False)

    # N_err = 10   #rapport du nombre d'erreurs

    # fig1, ax1 = plt.subplots()
    # ax1.scatter(L_Temp, L_E, s=4, color="blue")
    # ax1.errorbar(L_Temp[::len(L_Temp)//N_err], L_E[::len(L_Temp)//N_err], yerr=L_Eerr[::len(L_Temp)//N_err]*(1/(L*L)), linewidth=0.5, color="black", ls="None")
    # ax1.set_xlabel("Temperature")
    # ax1.set_ylabel("Energie")
    # ax1.set_title("Energie totale")

    # fig2, ax2 = plt.subplots()
    # ax2.scatter(L_Temp, L_B, s=4, color="red")
    # ax2.errorbar(L_Temp[::len(L_Temp)//N_err], L_B[::len(L_Temp)//N_err], yerr=L_Berr[::len(L_Temp)//N_err]*(1/(L*L)), linewidth=0.5, color="black", ls="None")
    # ax2.set_xlabel("Temperature")
    # ax2.set_ylabel("Magnetisation")
    # ax2.set_title("Magnetisation totale")


    # fig3, ax3 = plt.subplots()
    # ax3.scatter(L_Temp, L_C, s=4, color="green")
    # ax3.errorbar(L_Temp[::len(L_Temp)//N_err], L_C[::len(L_Temp)//N_err], yerr=L_Cerr[::len(L_Temp)//N_err]*(1/(L*L)), linewidth=0.5, color="black", ls="None")
    # ax3.set_xlabel("Temperature")
    # ax3.set_ylabel("chaleur spécifique")
    # ax3.set_title("chaleur spécifique")

    # fig3, ax3 = plt.subplots()
    # ax3.scatter(L_Temp, L_X, s=4, color="black")
    # ax3.errorbar(L_Temp[::len(L_Temp)//N_err], L_X[::len(L_Temp)//N_err], yerr=L_Xerr[::len(L_Temp)//N_err]*(1/(L*L)), linewidth=0.5, color="black", ls="None")
    # ax3.set_xlabel("Temperature")
    # ax3.set_ylabel("susceptibilité")
    # ax3.set_title("susceptibilité")
    # plt.show()
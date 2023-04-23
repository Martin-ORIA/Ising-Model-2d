import numpy as np
import random
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
import pandas as pd
from numba import jit
import sys
from itertools import repeat
from statsmodels.tsa.stattools import acf, pacf
##### CONSTANTES #####




@jit(nopython=True, boundscheck=False, fastmath=True)
def roll_hor(arr, k=1):
    """implémentation numba de la fonction numpy non compatible np.roll selon l'axe horizontal"""
    return np.concatenate((arr[:,-k:], arr[:,:-k]), axis=1)

@jit(nopython=True, boundscheck=False, fastmath=True)
def roll_vert(arr, k=1):
    """implémentation numba de la fonction numpy non compatible np.roll selon l'axe vertical"""
    return np.concatenate((arr[-k:,:], arr[:-k,:]), axis=0)


@jit(nopython=True, boundscheck=False, fastmath=True)
def EXP_MOD(T : np.int64) -> np.array:
    """Tabulation de la fonction exponentielle"""
    EXP = np.zeros(9)
    EXP[0], EXP[1], EXP[4], EXP[5], EXP[8] = 1., np.exp(8/T), np.exp(-4/T), np.exp(4/T), np.exp(-8/T)
    return EXP


@jit(nopython=True, boundscheck=False, fastmath=True)
def Monte_Carlo(S : np.array, L : np.int64, B : np.int64, T : np.float64, EXP : np.array) -> np.array:
    """
    Cette fonction réalise un step de monte carlo
    L est la taille de la matrice
    B est le champ magnétique
    T est la température
    EXP est un array des valeurs de l'exponentielle tabulée
    retourne la matrice de spins S                                    
    """
    Rand_Coord = np.random.choice(L, size=(2, L*L))
    
    for iter1 in range(L*L):

        i, j = Rand_Coord[:,iter1]

        deltaE = np.float64(2)*S[i,j]*(S[(i+1)%L, j] + S[i, (j+1)%L] + S[(i-1)%L, j] + S[i, (j-1)%L])
        if deltaE < 0 or random.random() < EXP[int(deltaE)]:
            S[i, j] = -S[i, j]


    return S


@jit(nopython=True, boundscheck=False, fastmath=True)
def Energie(S:np.array) -> np.float64:
    """retourne l'energie de la matrice S"""
    return (-np.sum(S*(roll_vert(S,1) + roll_hor(S,1) + roll_vert(S,-1) + roll_hor(S,-1))))/4

@jit(nopython=True, boundscheck=False, fastmath=True)
def Magnetisation(S:np.array) -> np.float64:
    """retourne la magnetisation de la matrice S"""
    return np.sum(S)




@jit(nopython=True, boundscheck=False, fastmath=True)
def Main_Loop(T:np.int64, L:np.int64, B:np.int64, n_eq:np.int64, n_mc:np.int64, L_Temp:np.array) -> np.array:
    """Boucle principale
    n_eq : pas d'equillibre
    n_mc : pas de monte carlo
    L_Temp liste de températures
    Retourne un l'array de la magnétisation"""

    S = np.random.choice(np.array([-1, 1]), size=(L, L))

    M = np.zeros(n_mc)
    EXP = EXP_MOD(L_Temp[T])
    for iter_eq in range(n_eq): #pas d'equilibre
        S = Monte_Carlo(S, L, B, L_Temp[T], EXP)
        
    for iter_main in range(n_mc): #pas de monte carlo
        S = Monte_Carlo(S, L, B, L_Temp[T], EXP)

        M[iter_main] = Magnetisation(S)

    return M



def iterations(L, B, n_eq, n_mc):
    """Permet de lancer la boucle en multi-processing"""
    T_A = np.linspace(1, 2, 35)
    T_B = np.linspace(2, 2.75, 130)
    T_D = np.linspace(2.75, 3.5, 35)
    L_Temp = np.concatenate((T_A, T_B, T_D))
    L_Temp = np.array([1.5]) #ici on souhaite obtenir l'autocorrelation pour T = 1.5

    t1 = time.time()
    with Pool() as pool: #Multiprocessing sur les temperatures
        T_arg = [T for T in range(len(L_Temp))]
        total = pool.starmap(Main_Loop, zip(T_arg, repeat(L), repeat(B), repeat(n_eq), repeat(n_mc), repeat(L_Temp)))
    t2 = time.time()
    print(t2-t1)

    total = np.array(total)
    print(total.shape)
    L_Corr = total[0]
    #Création du Dataset
    L_L = np.ones(len(L_Corr))*L
    L_Bext = np.ones(len(L_Corr))*B
    L_neq = np.ones(len(L_Corr))*n_eq
    L_n_mc = np.ones(len(L_Corr))*n_mc  
    L_t = np.ones(len(L_Corr))*(t2-t1)
    L_Temp = L_Temp[0]*np.ones(len(L_Corr))

    #Création du fichier CSV
    dataset = True
    if dataset == True:
        data = {
        'L_Corr': L_Corr,
        'L_L': L_L,
        'L_Bext': L_Bext,
        'L_neq': L_neq,
        'L_n_mc': L_n_mc,
        'L_t': L_t
        }
        df = pd.DataFrame(data)
    
        # Save the DataFrame as a CSV file
        df.to_csv('AUTOCORRdataL=' + str(B) + '_T=' + str(len(L_Temp)) + '_n_mc=' + str(n_mc) + '.csv', index=False)

if __name__ == "__main__":

    iterations(20, 0, 0, 50000)  #Matrice de taille 20*20, sans pas d'équillibre avec 5,000 pas de monte carlo

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


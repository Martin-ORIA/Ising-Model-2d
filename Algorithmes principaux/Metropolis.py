import numpy as np
import random
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
import pandas as pd
from numba import jit
import sys
from itertools import repeat





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
    S matrice de spins
    L est la taille de la matrice
    B est le champ magnétique
    T est la température
    EXP est un array des valeurs de l'exponentielle tabulée                                    
    """
    Rand_Coord = np.random.choice(L, size=(2, L*L))  #tabulation des points aléatoires à visiter
    
    for iter1 in range(L*L):

        i, j = Rand_Coord[:,iter1]  #point aléatoire

        deltaE = np.float64(2)*S[i,j]*(S[(i+1)%L, j] + S[i, (j+1)%L] + S[(i-1)%L, j] + S[i, (j-1)%L]) + np.float64(2)*B*S[i,j]  
        if deltaE < 0 or random.random() < EXP[int(deltaE)]:
            S[i, j] = -S[i, j]  #flip


    return S


@jit(nopython=True, boundscheck=False, fastmath=True)
def Energie(S:np.array) -> np.float64:
    """retourne l'energie de la matrice S"""
    return (-np.sum(S*(roll_vert(S,1) + roll_hor(S,1) + roll_vert(S,-1) + roll_hor(S,-1))))/4

@jit(nopython=True, boundscheck=False, fastmath=True)
def Magnetisation(S:np.array) -> np.float64:
    """retourne la magnetisation de la matrice S"""
    return np.abs(np.sum(S))




@jit(nopython=True, boundscheck=False, fastmath=True)
def Main_Loop(T:np.int64, L:np.int64, B:np.int64, n_eq:np.int64, n_mc:np.int64, L_Temp:np.array) -> np.array:
    """Boucle principale
    T indice de la temperature
    L taille du systeme
    n_eq : pas d'equillibre
    n_mc : pas de monte carlo
    L_Temp liste de températures
    Retourne un array contenant la temperature, l'energie, l'energie au carré, la magnetisation, la magnetisation moyenne au carré et leurs incertitudes"""
    S = np.random.choice(np.array([-1, 1]), size=(L, L))
    E = np.zeros(n_mc)
    M = np.zeros(n_mc)
    EXP = EXP_MOD(L_Temp[T])
    for iter_eq in range(n_eq):  #pas d'équillibre
        S = Monte_Carlo(S, L, B, L_Temp[T], EXP)
        
    for iter_main in range(n_mc):   #pas de monte carlo
        S = Monte_Carlo(S, L, B, L_Temp[T], EXP)
        E[iter_main] = Energie(S)
        M[iter_main] = Magnetisation(S)

    return np.array([L_Temp[T], np.mean(E), np.mean(E*E), np.mean(M), np.mean(M*M), np.std(E), np.std(E*E), np.std(M), np.std(M*M)])



def iterations(L, B, n_eq, n_mc):
    """Permet de lancer la boucle en multi-processing"""
    T_A = np.linspace(1, 2, 35)
    T_B = np.linspace(2, 2.75, 130)
    T_D = np.linspace(2.75, 3.5, 35)
    L_Temp = np.concatenate((T_A, T_B, T_D)) #Liste des températures

    t1 = time.time()
    with Pool() as pool:
        T_arg = [T for T in range(len(L_Temp))]
        total = pool.starmap(Main_Loop, zip(T_arg, repeat(L), repeat(B), repeat(n_eq), repeat(n_mc), repeat(L_Temp)))
    t2 = time.time()
    print(t2-t1)

    total = np.array(total)

    #Creation de la base de données
    L_E, L_Eerr = total[ : ,1]*(1/(L*L)), total[:,5]*(1/(L*L))
    L_B, L_Berr = total[ : ,3]*(1/(L*L)), total[:,7]*(1/(L*L))

    #Calcul de L_Cerr
    deltaC1 = total[:, 6]*(1/(L*L))
    deltaC2 = (L_E*L_Eerr*np.sqrt(2))/(L*L)
    L_C, L_Cerr = ((1/(L*L))*total[ : ,2]-(1/(L*L))*(total[ : ,1]*total[ : ,1]))*(1/(L_Temp*L_Temp)), (1/(L_Temp*L_Temp))*np.sqrt(deltaC1**2 + deltaC2**2)  #LCerr obtenu par propagation des erreurs

    #Calcul de L_Xerr
    deltaX1 = total[:, 8]*(1/(L*L))
    deltaX2 = (L_B*L_Berr*np.sqrt(2))/(L*L)
    L_X, L_Xerr = ((1/(L*L))*total[ : ,4]-(1/(L*L))*(total[ : ,3]*total[ : ,3]))*(1/L_Temp), (1/(L_Temp))*np.sqrt(deltaX1**2 + deltaX2**2)

    L_L = np.ones(len(L_E))*L
    L_Bext = np.ones(len(L_E))*B
    L_neq = np.ones(len(L_E))*n_eq
    L_n_mc = np.ones(len(L_E))*n_mc  
    L_t = np.ones(len(L_E))*(t2-t1)
    
    #Fichier CSV
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

if __name__ == "__main__":

    iterations(20, 0, 0, 50000)   #iterations pour une matrice de taille 20*20 sans champ externe, sans pas n_eq et avec 50,000 pas n_mc

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


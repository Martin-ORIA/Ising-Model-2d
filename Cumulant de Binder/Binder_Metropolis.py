import numpy as np
import random
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
import pandas as pd
from numba import jit
import sys
from itertools import repeat
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
    Retourne un array contenant la temperature, la magnetisation moyenne au carré, la magnetisation moyenne puissance 4 et leurs incertitudes"""

    S = np.random.choice(np.array([-1, 1]), size=(L, L))
    M = np.zeros(n_mc)
    EXP = EXP_MOD(L_Temp[T])
    for iter_eq in range(n_eq):
        S = Monte_Carlo(S, L, B, L_Temp[T], EXP)
        
    for iter_main in range(n_mc):
        S = Monte_Carlo(S, L, B, L_Temp[T], EXP)
        M[iter_main] = Magnetisation(S)
    M = M*(1/(L*L))
    return np.array([L_Temp[T], np.mean(M*M), np.mean(M*M*M*M), np.std(M*M), np.std(M*M*M*M)])



def  iterations(L, B, n_eq, n_mc):
    """Permet de lancer la boucle en multi-processing"""
    T_A = np.linspace(1, 2, 20)
    T_B = np.linspace(2, 2.75, 60)
    T_D = np.linspace(2.75, 3.5, 20)
    L_Temp = np.concatenate((T_A, T_B, T_D))


    t1 = time.time()
    with Pool() as pool:
        T_arg = [T for T in range(len(L_Temp))]
        total = pool.starmap(Main_Loop, zip(T_arg, repeat(L), repeat(B), repeat(n_eq), repeat(n_mc), repeat(L_Temp)))
    t2 = time.time()
    print(t2-t1)

    total = np.array(total)

    #Creation de la base de données
    L_B2, L_B2err = total[ : ,1], total[:,3]
    L_B4, L_B4err = total[ : ,2], total[:,4]


    L_L = np.ones(len(L_B2))*L
    L_Bext = np.ones(len(L_B2))*B
    L_neq = np.ones(len(L_B2))*n_eq
    L_n_mc = np.ones(len(L_B2))*n_mc  
    L_t = np.ones(len(L_B2))*(t2-t1)
    
    #Fichier CSV
    dataset = True
    if dataset == True:
        data = {
        'L_Temp': L_Temp,
        'L_B2': L_B2,
        'L_B2err': L_B2err,
        'L_B4': L_B4,
        'L_B4err': L_B4err,
        'L_L': L_L,
        'L_Bext': L_Bext,
        'L_neq': L_neq,
        'L_n_mc': L_n_mc,
        'L_t': L_t
        }
        df = pd.DataFrame(data)
    
        # Save the DataFrame as a CSV file
        df.to_csv('BINDERdataL=' + str(L) + '_T=' + str(len(L_Temp)) + '_n_mc=' + str(n_mc) + '.csv', index=False)

if __name__ == "__main__":
    Ls = [11, 13, 15, 16, 20, 40]  #liste de tailles
    for L in Ls:
        print(L)
        iterations(L, 0, 500000, 500000)  #tailles L, pas n_eq = 50,000 , n_mc = 50,000


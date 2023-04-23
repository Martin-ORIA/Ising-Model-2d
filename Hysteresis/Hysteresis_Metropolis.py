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
def Monte_Carlo(S : np.array, L : np.int64, B : np.int64, T : np.float64) -> np.array:
    """
    Cette fonction réalise un step de monte carlo
    S matrice de spins
    L est la taille de la matrice
    B est le champ magnétique
    T est la température
    EXP est un array des valeurs de l'exponentielle tabulée                                    
    """
    Rand_Coord = np.random.choice(L, size=(2, L*L))   #tabulation des points aléatoires
    
    for iter1 in range(L*L):

        i, j = Rand_Coord[:,iter1]

        deltaE = np.float64(2)*S[i,j]*(S[(i+1)%L, j] + S[i, (j+1)%L] + S[(i-1)%L, j] + S[i, (j-1)%L]) + np.float64(2)*B*S[i,j]
        if deltaE < 0 or random.random() < np.exp(-deltaE/T):
            S[i, j] = -S[i, j]   #flip


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
def Main_Loop(S:np.array, B:np.int64, L:np.int64, T:np.int64, n_eq:np.int64, n_mc:np.int64, L_Bext:np.array) -> np.array:
    """Boucle principale
    S matrice de spins
    B champ externe
    L taille du systeme
    n_eq : pas d'equillibre
    n_mc : pas de monte carlo
    L_Temp liste de températures
    Retourne un array contenant magnetisation moyenne ainsi que la matrice de Spins S"""
    
    E = np.zeros(n_mc)
    M = np.zeros(n_mc)
    for iter_eq in range(n_eq):   #pas d'equillibre
        S = Monte_Carlo(S, L, L_Bext[B], T)
        
    for iter_main in range(n_mc):   #pas de monte_carlo
        S = Monte_Carlo(S, L, L_Bext[B], T)
        M[iter_main] = Magnetisation(S)

    return np.array(np.mean(M)), S
    


def iterations(L, L_Bext, n_eq, n_mc, T, color):
    """Permet de lancer la boucle en multi-processing"""
    S = np.random.choice(np.array([-1, 1]), size=(L, L))  #matrice initiale aléatoire
    t1 = time.time()
    L_B = []   #liste de la magnetisation lors du chauffage
    L_B2 = []  #liste de la magnetisation lors du reffroidissement

    #Chauffage du materiau (augmentation de T)
    L_Bext = np.linspace(-2, 2, 500)
    for Bext in range(len(L_Bext)):
        BB, S = Main_Loop(S, Bext, L, T, n_eq, n_mc, L_Bext)
        L_B.append(BB/(L*L))

    #Refroidissement (diminution de la valeur de T) (la matrice d'entrée est la matrice de sortie du chauffage)
    L_Bext2 = L_Bext[::-1]
    for Bext in range(len(L_Bext2)):
        BB, S = Main_Loop(S, Bext, L, T, n_eq, n_mc, L_Bext2)
        L_B2.append(BB/(L*L))


    #Creation de la base de données

    L_B2 = np.array(L_B2)
    L_B = np.array(L_B)
    t2 = time.time()
    print(t2-t1)





    L_Temp = np.ones(len(L_B))*T
    L_L = np.ones(len(L_B))*L
    L_neq = np.ones(len(L_B))*n_eq
    L_n_mc = np.ones(len(L_B))*n_mc  
    L_t = np.ones(len(L_B))*(t2-t1)
    
    #Creation du fichier CSV
    dataset = True
    if dataset == True:
        data = {
        'L_Temp': L_Temp,
        'L_B': L_B,
        'L_L': L_L,
        'L_Bext': L_Bext,
        'L_neq': L_neq,
        'L_n_mc': L_n_mc,
        'L_t': L_t
        }
        df = pd.DataFrame(data)
    
        # Save the DataFrame as a CSV file
        df.to_csv('2CHAMP_EXTERNEdataL=' + str(len(L_Bext)) + '_T=' + str(len(L_Temp)) + '_n_mc=' + str(n_mc) + '.csv', index=False)

    plt.plot(L_Bext, L_B-np.ones(len(L_B))*0, linewidth=1, label="T = "+str(T), c=color)  #affichage des données
    plt.plot(L_Bext2, L_B2, linewidth=1, c=color)
    



if __name__ == "__main__":
    L_B = np.linspace(-1, 1.5, 40)

    iterations(20, L_B, 0, 500, 1.5, "red")  #iterations pour une matrice de taille 30*30, avec le champ L_B, 5000 pas de monte carlo et une temperature T=1.5
    iterations(20, L_B, 0, 500, 2, "blue")
    iterations(20, L_B, 0, 500, 3, "green")

    plt.xlabel("H (Champ externe)")
    plt.ylabel("M (Magnetisation)")
    plt.title("Hysteresis")
    plt.legend()
    plt.show()

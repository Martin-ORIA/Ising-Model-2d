import numpy as np
import matplotlib.pyplot as plt
from itertools import repeat
import time
from multiprocessing import Pool
import pandas as pd
from numba import jit
import sys
import numpy as np
from tqdm import tqdm

sys.setrecursionlimit(10000000)   #augmentation du nombre d'appel d'une fonction à l'interieur d'elle meme (recursion)

@jit(nopython=True)
def roll_hor(arr, k=1):
    """implémentation numba de la fonction numpy non compatible np.roll selon l'axe horizontal"""
    return np.concatenate((arr[:,-k:], arr[:,:-k]), axis=1)

@jit(nopython=True)
def roll_vert(arr, k=1):
    """implémentation numba de la fonction numpy non compatible np.roll selon l'axe vertical"""
    return np.concatenate((arr[-k:,:], arr[:-k,:]), axis=0)

@jit(nopython=True)
def nonzero_coordinates(arr):
    """
    Retourne un array contenant sur la ligne 0 les X et sur la ligne 1 les Y
    """
    x_coords = []
    y_coords = []
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i][j] != 0:
                x_coords.append(i)
                y_coords.append(j)
    return np.array([x_coords, y_coords])



#@jit(nopython=True, boundscheck=False, fastmath=True)
def SW_algorithm(Coord, S, T, L):
    """Retourne la matrice S après avoir réalisé un flip de swendsen wang"""

    non_v = np.full((L, L), True)
    liens = np.zeros((2, L, L))
    proba = np.exp(-2/T)    


    delta_spin_hor, delta_spin_ver = np.abs(S + roll_hor(S,-1))/2, np.abs(S + roll_vert(S,-1))/2
    S_hor = nonzero_coordinates(delta_spin_hor)
    S_ver = nonzero_coordinates(delta_spin_ver)

    #VOISINS IDENTIQUES
    for i in range(np.shape(S_hor)[1]):
        liens[0, S_hor[0,i], S_hor[1,i]] = 0 if np.random.binomial(1, proba) == 1 else 1

    for j in range(np.shape(S_ver)[1]):
    	liens[1, S_ver[0,j], S_ver[1,j]] = 0 if np.random.binomial(1, proba) == 1 else 1

    #BOUCLE PRINCIPALE
    for i in range(L*L):
        cluster = []
        flip_val = 2*np.random.randint(2) - 1 
        X, Y = Coord[0][i], Coord[1][i]
        cluster, S = Walk(X, Y, liens, non_v, cluster, S, flip_val)
          
        
            
    return S



def Walk(x, y, liens, non_v, cluster, S, flip_val):
    """Marche pour visiter chacuns des sous voisins pour construire chacun des clusters"""
    L = 10

    if non_v[x, y]:
        non_v[x, y] = False
        cluster.append([x, y])
        S[x, y] = S[x, y] * flip_val
                
        if liens[0][x][y] == 1:
            n_x = x
            n_y = (y + 1)%L
            cluster, S = Walk(n_x, n_y, liens, non_v, cluster, S, flip_val)
            
        if liens[0][x][(y - 1)%L] == 1:
            n_x = x
            n_y = (y - 1)%L
            cluster, S = Walk(n_x, n_y, liens, non_v, cluster, S, flip_val)
            
        if liens[1][x][y] == 1:
            n_x = (x + 1)%L
            n_y = y
            cluster, S = Walk(n_x, n_y, liens, non_v, cluster, S, flip_val)
           
        if liens[1][(x - 1)%L][y] == 1:
            n_x = (x - 1)%L
            n_y = y
            cluster, S = Walk(n_x, n_y, liens, non_v, cluster, S, flip_val)
            
    return cluster, S



@jit(nopython=True, boundscheck=False, fastmath=True)
def Energie(S : np.array) -> np.float64:
    """retourne l'energie de la matrice S"""
    return (-np.sum(S*(roll_vert(S,1) + roll_hor(S,1) + roll_vert(S,-1) + roll_hor(S,-1))))/4

@jit(nopython=True, boundscheck=False, fastmath=True)
def Magnetisation(S : np.array) -> np.float64:
    """retourne la magnetisation de la matrice S"""
    return np.abs(np.sum(S))




#@jit(nopython=True, boundscheck=False, fastmath=True)
def Main_Loop(T, Coord, L, B, n_eq, n_mc, L_Temp) -> np.array:
    """Boucle principale
    T indice de la température
    Coord meshgrid de chacune des coordonnées des spins de la matrice
    L taille du système
    n_eq : pas d'equillibre
    n_mc : pas de monte carlo
    L_Temp liste de températures
    Retourne un array contenant la temperature, l'energie, l'energie au carré, la magnetisation, la magnetisation moyenne au carré et leurs incertitudes"""
    S = np.random.choice(np.array([-1, 1]), size=(L, L))
    E = np.zeros(n_mc)
    M = np.zeros(n_mc)

    for iter_eq in range(n_eq):
        S = SW_algorithm(Coord, S, L_Temp[T], L)
        
    for iter_main in range(n_mc):
        S = SW_algorithm(Coord, S, L_Temp[T], L)
        E[iter_main] = Energie(S)
        M[iter_main] = Magnetisation(S)

    return np.array([L_Temp[T], np.mean(E), np.mean(E*E), np.mean(M), np.mean(M*M), np.std(E), np.std(E*E), np.std(M), np.std(M*M)])



if __name__ == "__main__":
    L = 10      #taille du système
    B = 0        #champ externe
    n_eq = 10    #pas d'équillibre
    n_mc = 1000    #pas de Swendsen-Wang
    L_Temp = np.linspace(1, 4, 200)
    S = np.random.choice([-1, 1], size=(L, L))
    spin_site_numbers = range(L*L)

    Coord_X, Coord_Y = [range(L), range(L)]  #creation du mesh des coordonnées des spins
    Coord = np.meshgrid(Coord_X, Coord_Y)
    Coord = np.reshape(Coord,(2,-1))

    t1 = time.time()
    with Pool() as pool:   #multiprocessing
        T_arg = [T for T in range(len(L_Temp))]
        total = pool.starmap(Main_Loop, zip(T_arg, repeat(Coord), repeat(L), repeat(B), repeat(n_eq), repeat(n_mc), repeat(L_Temp)))
    t2 = time.time()
    print(t2-t1)
    total = np.array(total)

    #Creation du dataset
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
    
    #Creation du fichier CSV
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

    N_err = 10   #rapport du nombre d'erreurs

    fig1, ax1 = plt.subplots()
    ax1.scatter(L_Temp, L_E, s=4, color="blue")
    ax1.errorbar(L_Temp[::len(L_Temp)//N_err], L_E[::len(L_Temp)//N_err], yerr=L_Eerr[::len(L_Temp)//N_err]*(1/(L*L)), linewidth=0.5, color="black", ls="None")
    ax1.set_xlabel("Temperature")
    ax1.set_ylabel("Energie")
    ax1.set_title("Energie totale")

    fig2, ax2 = plt.subplots()
    ax2.scatter(L_Temp, L_B, s=4, color="red")
    ax2.errorbar(L_Temp[::len(L_Temp)//N_err], L_B[::len(L_Temp)//N_err], yerr=L_Berr[::len(L_Temp)//N_err]*(1/(L*L)), linewidth=0.5, color="black", ls="None")
    ax2.set_xlabel("Temperature")
    ax2.set_ylabel("Magnetisation")
    ax2.set_title("Magnetisation totale")


    fig3, ax3 = plt.subplots()
    ax3.scatter(L_Temp, L_C, s=4, color="green")
    ax3.errorbar(L_Temp[::len(L_Temp)//N_err], L_C[::len(L_Temp)//N_err], yerr=L_Cerr[::len(L_Temp)//N_err]*(1/(L*L)), linewidth=0.5, color="black", ls="None")
    ax3.set_xlabel("Temperature")
    ax3.set_ylabel("chaleur spécifique")
    ax3.set_title("chaleur spécifique")

    fig3, ax3 = plt.subplots()
    ax3.scatter(L_Temp, L_X, s=4, color="black")
    ax3.errorbar(L_Temp[::len(L_Temp)//N_err], L_X[::len(L_Temp)//N_err], yerr=L_Xerr[::len(L_Temp)//N_err]*(1/(L*L)), linewidth=0.5, color="black", ls="None")
    ax3.set_xlabel("Temperature")
    ax3.set_ylabel("susceptibilité")
    ax3.set_title("susceptibilité")
    plt.show()


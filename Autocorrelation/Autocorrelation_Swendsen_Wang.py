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

sys.setrecursionlimit(10000000)  #augmentation du nombre d'appel d'une fonction à l'interieur d'elle meme (recursion)

@jit(nopython=True, boundscheck=False, fastmath=True)
def roll_hor(arr, k=1):
    """implémentation numba de la fonction numpy non compatible np.roll selon l'axe horizontal"""
    return np.concatenate((arr[:,-k:], arr[:,:-k]), axis=1)

@jit(nopython=True, boundscheck=False, fastmath=True)
def roll_vert(arr, k=1):
    """implémentation numba de la fonction numpy non compatible np.roll selon l'axe vertical"""
    return np.concatenate((arr[-k:,:], arr[:-k,:]), axis=0)

@jit(nopython=True)
def nonzero_coordinates(arr):
    """
    Retournes un array de deux dimensions dont la première ligne corresponds au X
    et la seconde aux Y
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
    """Retourne la matrice S de spin après un pas de swendsen wang"""

    not_visited = np.full((L, L), True)
    bonds = np.zeros((2, L, L))
    chance_value = np.exp(-2/T)    


    delta_spin_hor, delta_spin_ver = np.abs(S + roll_hor(S,-1))/2, np.abs(S + roll_vert(S,-1))/2
    S_hor = nonzero_coordinates(delta_spin_hor)
    S_ver = nonzero_coordinates(delta_spin_ver)

    #VOISINS IDENTIQUES
    for i in range(np.shape(S_hor)[1]):
        bonds[0, S_hor[0,i], S_hor[1,i]] = 0 if np.random.binomial(1, chance_value) == 1 else 1

    for j in range(np.shape(S_ver)[1]):
    	bonds[1, S_ver[0,j], S_ver[1,j]] = 0 if np.random.binomial(1, chance_value) == 1 else 1

    #BOUCLE PRINCIPALE
    for i in range(L*L):
        cluster = []
        flip_val = 2*np.random.randint(2) - 1 
        X, Y = Coord[0][i], Coord[1][i]
        cluster, S = Walk(X, Y, bonds, not_visited, cluster, S, flip_val)
          
        
            
    return S



def Walk(x, y, bonds, not_visited, cluster, S, flip_val):
    """Marche permettant de former la liste des clusters ainsi que les spins appartenant à ces derniers"""
    L = 20

    if not_visited[x, y]:
        not_visited[x, y] = False
        cluster.append([x, y])
        S[x, y] = S[x, y] * flip_val
                
        if bonds[0][x][y] == 1:
            n_x = x
            n_y = (y + 1)%L
            cluster, S = Walk(n_x, n_y, bonds, not_visited, cluster, S, flip_val)   #on note ici que l'on réalise une recursion en appelant la fonction dans cette meme fonction.
                                                                                    # dans python cette demarche est lente et limitée (augmentation de cette limite ligne 12)
            
        if bonds[0][x][(y - 1)%L] == 1:
            n_x = x
            n_y = (y - 1)%L
            cluster, S = Walk(n_x, n_y, bonds, not_visited, cluster, S, flip_val)
            
        if bonds[1][x][y] == 1:
            n_x = (x + 1)%L
            n_y = y
            cluster, S = Walk(n_x, n_y, bonds, not_visited, cluster, S, flip_val)
           
        if bonds[1][(x - 1)%L][y] == 1:
            n_x = (x - 1)%L
            n_y = y
            cluster, S = Walk(n_x, n_y, bonds, not_visited, cluster, S, flip_val)
            
    return cluster, S



@jit(nopython=True, boundscheck=False, fastmath=True)
def Energie(S:np.array) -> np.float64:
    """retourne l'energie de la matrice S"""
    return (-np.sum(S*(roll_vert(S,1) + roll_hor(S,1) + roll_vert(S,-1) + roll_hor(S,-1))))/4

@jit(nopython=True, boundscheck=False, fastmath=True)
def Magnetisation(S:np.array) -> np.float64:
    """retourne la magnetisation de la matrice S"""
    return np.sum(S)




#@jit(nopython=True, boundscheck=False, fastmath=True)
def Main_Loop(T, Coord, L, B, n_eq, n_mc, L_Temp) -> np.array:
    """Boucle principale
    T température
    Coord meshgrid de chacune des coordonnées des spins de la matrice
    n_eq : pas d'equillibre
    n_mc : pas de monte carlo
    L_Temp liste de températures
    Retourne un l'array de la magnétisation"""
    S = np.random.choice(np.array([-1, 1]), size=(L, L))

    M = np.zeros(n_mc)

    for iter_eq in range(n_eq):
        S = SW_algorithm(Coord, S, L_Temp[T], L)
        
    for iter_main in range(n_mc):
        S = SW_algorithm(Coord, S, L_Temp[T], L)
        M[iter_main] = Magnetisation(S)

    return M



if __name__ == "__main__":
    L = 20  #taille du système
    B = 0    #champ externe
    n_eq = 0      #pas d'equilibre
    n_mc = 50000       #pas de swendsen wang
    L_Temp = np.array([1.5]) #température cible

    S = np.random.choice([-1, 1], size=(L, L))
    spin_site_numbers = range(L*L)
    Coord_X, Coord_Y = [range(L), range(L)]
    Coord = np.meshgrid(Coord_X, Coord_Y)
    Coord = np.reshape(Coord,(2,-1))

    t1 = time.time()
    with Pool() as pool:   #Multiprocessing sur les temperatures
        T_arg = [T for T in range(len(L_Temp))]
        total = pool.starmap(Main_Loop, zip(T_arg, repeat(Coord), repeat(L), repeat(B), repeat(n_eq), repeat(n_mc), repeat(L_Temp)))
    t2 = time.time()
    print(t2-t1)
    total = np.array(total)
    print(total.shape)
    L_Corr = total[0]

    #Creation de la base de données
    L_L = np.ones(len(L_Corr))*L
    L_Bext = np.ones(len(L_Corr))*B
    L_neq = np.ones(len(L_Corr))*n_eq
    L_n_mc = np.ones(len(L_Corr))*n_mc  
    L_t = np.ones(len(L_Corr))*(t2-t1)
    L_Temp = L_Temp[0]*np.ones(len(L_Corr))

    #Creation du fichier CSV
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
        df.to_csv('SWENDSEN_AUTOCORRdataL=' + str(B) + '_T=' + str(len(L_Temp)) + '_n_mc=' + str(n_mc) + '.csv', index=False)

    ##N_err = 10   #rapport du nombre d'erreurs

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


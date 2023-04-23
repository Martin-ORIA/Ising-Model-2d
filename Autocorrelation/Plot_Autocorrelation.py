import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from multiprocessing import Pool
from itertools import repeat
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf


#Selection des données (modifier le nom du fichier si les paramètres sont différents)
data = pd.read_csv("AUTOCORRdataL=0_T=50000_n_mc=50000.csv")
dataWOLF = pd.read_csv("WOLFAUTOCORRdataL=0_T=50000_n_mc=50000.csv")
dataSWENDSEN = pd.read_csv("SWENDSEN_AUTOCORRdataL=0_T=50000_n_mc=50000.csv")

fig1, ax1 = plt.subplots()


#Cette section affiche un graphique de l'evolution de la magnetisation en fonction des pas des algorithmes
L_c = ["orange", "red", "green", "blue"]
L_C = np.array(data["L_Corr"])
L_CWOLF = np.array(dataWOLF["L_Corr"])
L_CSWENDSEN = np.array(dataSWENDSEN["L_Corr"])
ax1.plot(np.linspace(0, 5000, len(L_C)), np.abs(L_C)/(400), linewidth=1, color="blue", label="Metropolis")
ax1.plot(np.linspace(0, 5000, len(L_CWOLF)), np.abs(L_CWOLF)/(400), linewidth=1, color="red", label="Wolf")
ax1.plot(np.linspace(0, 5000, len(L_CSWENDSEN)), np.abs(L_CSWENDSEN)/(400), linewidth=1, color="green", label="Swendsen-Wang")

ax1.set_xlabel("Pas")
ax1.set_ylabel("Magnetisation M")
ax1.set_title("Magnetisation")
plt.legend()


#Calcul de l'autocorrelation
L_C = acf(L_C, nlags=5000, fft=False)
L_CWOLF = acf(L_CWOLF, nlags=5000, fft=False)
L_CSWENDSEN = acf(L_CSWENDSEN, nlags=5000, fft=False)

fig1, ax1 = plt.subplots()
ax1.plot(np.linspace(0, 50000/500, len(L_C)), L_C, linewidth=1, color="blue", label="Metropolis")
ax1.plot(np.linspace(0, 50000/500, len(L_CWOLF)), L_CWOLF, linewidth=1, color="red", label="Wolf")
ax1.plot(np.linspace(0, 50000/500, len(L_CSWENDSEN)), L_CSWENDSEN, linewidth=1, color="green", label="Swendsen-Wang")
ax1.set_xlabel("Pas/Lags")
ax1.set_ylabel("Autocorrelation")
ax1.set_title("Autocorrelation de la magnetisation T=Tc")
plt.legend()
plt.show()
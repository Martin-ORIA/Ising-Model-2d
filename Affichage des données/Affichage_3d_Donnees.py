from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation



Ls = [k for k in range(2, 61)]
Ls += [65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 120, 140, 200]   #liste des tailles de systèmes (fichiers contenus dans le dossier "données principales")
                                                                   # faire attention que les données soient bien dans le meme dossier
                                                                   #Attention à ce que les fichier aient les memes paramètres n_mc, n_eq, L_Temp

L_plot = np.zeros((4, len(Ls), 200))  #array d'affichage
L_time = []  #liste contenant les temps d'execution

L_Temp = np.zeros((len(Ls), 200))
SPINFLIPS = 0   #compteur de flips de spins
for L in range(len(Ls)):
	data = pd.read_csv("dataL=" + str(Ls[L]) + "_T=200_n_mc=500000.csv")
	L_Temp[L, :] = data["L_Temp"]
	L_plot[0, L, :] = np.array(data["L_B"])
	L_plot[1, L, :] = np.array(data["L_E"])
	L_plot[2, L, :] = np.array(data["L_X"])
	L_plot[3, L, :] = np.array(data["L_C"])
	L_time.append(data["L_t"][0])
	SPINFLIPS += Ls[L]*Ls[L]*1000000*200
print("SPINS FLIPS : ", SPINFLIPS/(10**9))
print("Temps simulation : ", sum(L_time)/3600)



L_label = ["Magnétisation", "Energie", "Susceptibilité", "chaleur spécifique", "Susceptibilité erreur"]
L_view = [0, 180, 100, 100]  #angle de vue pour le plot 3d de chacune des grandeurs


for plot in range(4):
# 	print("a")
	fig = plt.figure()
	ax = plt.axes(projection='3d')
	def update(angle):
		"""retourne le plot avec l'angle de vue "angle" """

		ax.view_init(25, L_view[plot] + angle)

		X = L_Temp[0]
		Y = Ls
		X, Y = np.meshgrid(X, Y[:68] )
		ax.plot_surface(X, Y, L_plot[plot][:68] , cmap='inferno', edgecolor='none')  #création de la surface

		X = L_Temp[-1]
		Y = Ls-np.ones(len(Ls))*5
		X, Y = np.meshgrid(X, Y[68:] )
		ax.plot_surface(X, Y, L_plot[plot][68:] , cmap='inferno', edgecolor='none')  #création de la seconde surface car les pas de températures sont différents pour les grandes tailles
		                                                                             #(plus centrées sur la température de Curie)

		ax.set_xlabel("Température")
		ax.set_ylabel("L")
		ax.set_zlabel(L_label[plot])
		ax.set_title(L_label[plot])
	ani = FuncAnimation(fig, update, frames=range(0, 90, 5), interval=100)   #animation 3d
	ani.save(L_label[plot] + '.gif', writer='imagemagick')
plt.show()




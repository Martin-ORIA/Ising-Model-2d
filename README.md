Chaque partie décrit l'utilisation d'un dossier :
Données principales Github : https://github.com/Martin-ORIA/Ising-Model-2d


# 1 - Affichage des données :
	- Affichage_3d_Donnees.py : permet de tracer en trois dimensions les grandeurs pour plusieurs tailles de L.
					    Pour utiliser soit télécharger les données "Donnees principales" du Github et les
					    placer dans le meme dossier et executer le programme.
					    Sinon on peut créer des fichier de données à l'aide de l'un des codes présent
					    dans le dossier Algorithmes principaux et placer le fichier csv résultant dans le dossier de ce code.
					    Modifier ensuite la ligne 9 et 10 pour spécifier la liste des tailles L des simulations.	

	- Graphs_Donnees.py : Trace un maximum de quatres fichiers de données avec les incertitudes (contrairement au précédent)
				    Pour utiliser on doit placer dans le meme dossier le ou les fichiers csv de 
                            simulation (github ou nouvelle simulation). On note que les simulations doivent avoir les memes 
                            paramètres n_eq, n_mc, et L_Temp.


# 2 - Ajustement alpha gamma :
	-Ajustement_alpha_gamma.py : permet de déterminer par ajustement l'exposant critique alpha ou gamma.
					     Pour utiliser, nous recommandons fortement d'utiliser l'intégralité des fichiers de donnée présent
					     sur le Github afin d'obtenir une bonne estimation (données obtenues en 55h de simulations).
					     Placer les fichiers CSV dans le meme dossier. On peut modifier les tailles étudiées à la ligne 8-9
					     Remplacer "L_C" avec "L_X", et "L_Cerr" avec "L_Xerr" à la ligne 22 et 23 respectivement pour
					     obtenir l'exposant gamma associé à la susceptibilité.


# 3 - Algorithmes principaux :
	-Metropolis.py : Permet de réaliser une simulation en utilisant la méthode de Monte Carlo. Modifier la ligne 158 pour changer
			     les paramètres de la simulation : iterations(20, 0, 0, 50000) -> L=20, B=0, n_eq=0, n_mc=50000.
			     Avec L la taille de la matrice, B le champ exterieur, n_eq les pas d'equilibre, n_mc les pas de Monte Carlo
			     
	-Swendsen_Wang.py : Permet de réaliser une simulation en utilisant la méthode de Swendsen Wang. Modifier la ligne 75 et 146 pour changer
			        les paramètres de la simulation :  L=10, B=0, n_eq=10, n_mc=1000.
			        Avec L la taille de la matrice, B le champ exterieur, n_eq les pas d'equilibre, n_mc les pas de Swendsen Wang
				
	-Wolff.py : Permet de réaliser une simulation en utilisant la méthode de Wolff. Modifier la ligne 101 à 104 pour changer
			les paramètres de la simulation :  L=10, B=0, n_eq=100, n_mc=1000.
			Avec L la taille de la matrice, B le champ exterieur, n_eq les pas d'equilibre, n_mc les pas de Wolff

# 4 - Annexe Temperature Curie 2 :
	-Ajustement_Curie.py : Permet d'obtenir la température de curie par ajustement des phénomènes de taille finie.
				     Placer les fichiers de données (Github ou Nouveau) dans le dossier. Si l'on utilise de nouveaux fichiers
	                       de données on doit modifier les lignes 11-12 pour prendre en compte les nouvelles tailles (ajouter à la
				     liste Ls).

# 5 - Autocorrelation : 
	- Autocorrelation_Metropolis.py : Fonctionnement identique à l'algorithme de metropolis. Ce dernier retourne uniquement la magnetisation.
						    Pour des modifications des paramètres voir fonctionnement du code dans 3 - Metropolis.py
						    
	- Autocorrelation_Swendsen_Wang.py : Fonctionnement identique à l'algorithme de Swendsen_Wang. Ce dernier retourne uniquement la
                                           magnetisation. Pour des modifications des paramètres voir fonctionnement du code dans 
							 3 - Swendsen_Wang.py
							 
	- Autocorrelation_Wolff.py : Fonctionnement identique à l'algorithme de Wolff. Ce dernier retourne uniquement la
                                           magnetisation. Pour des modifications des paramètres voir fonctionnement du code dans 
							 3 - Wolff.py
							 
	- Plot_Autocorrelation.py : Trace un premier graphique de l'evolution de la magnétisation pour chacun des algorithmes. 
                                  Trace également un second graphique de l'autocorrelation de la magnétisation en fonction des pas.
					    Pour utiliser, executer pour des paramètres identiques les 3 programmes ci dessus. Une fois réalisé, 
					    modifier le nom des fichier des lgnes 13-15 avec les nouveaux.

# 6 - Cumulant de Binder :
	- Binder_Metropolis.py : Fonctionnement identique à l'algorithme de metropolis. Ce dernier retourne uniquement la magnetisation au
					 carré ainsi que la magnetisation puissance 4 et leur incertitudes.
					 Pour des modifications des paramètres voir fonctionnement du code dans 3 - Metropolis.py
					 
	- Binder_Ajustement.py : Permet de tracer les cumulants de binder pour les differentes tailles de simulation réalisées. 
					 Détermine également l'intersection de ces derniers et retourne la températue de curie associée.
					 Pour utiliser, modifier la ligne 15 avec les tailles de simulations réalisées avec Binder_Metropolis.py
					 Modifier également la ligne 17 "500000" et remplacer par le pas n_mc choisit.

# 7 - Hysteresis :
	- Hysteresis_Metropolis.py : Fonctionnement identique à l'algorithme de metropolis. Ce dernier réalise 2 simulations dans la meme 
					     execution. La première place la matrice S dans un champ externe augmentant de -2 a 2. La seconde 
					     La place dans un champs décroissant de 2 à -2 en prenant comme matrice initiale la matrice de sortie
					     de la première simulation.
					     Cette étape est réalisée pour 3 températures et affiche la courbe d'hystérésis.
					     On peut modifier les ligns 152-154 pour changer les paramètres. Les paramètres de la fonction iterations
					     sont identiques que pour 3 - Metropolis.py

# 8 - Ising Julia :
	- Metropolis.jl : Programme fondamentalement identique à 3 - Metropolis.py mais implémenté en Julia

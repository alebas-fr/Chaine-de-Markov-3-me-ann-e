# -*-coding:utf-8 -*

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.linalg import null_space
import time

"""def est_vecteur_stochastique(pi0):
	ret=True
	somme=0
	i=0
	if isinstance(pi0,list):
		while i<len(pi0) and ret:
			if isinstance(pi0[i],float)==True or isinstance(pi0[i],int)==True:
				if pi0[i]>=0.0:
					somme+=pi0[i]
				else:
					ret=False
					print("Données incorrectes "+str(pi0[i])+" : Il ne peut pas y avoir des valeurs négatives dans un vecteur stochastique")
			else:
				ret=False
				print("Données incorrectes "+str(pi0[i])+" : Il ne peut y avoir que des nombre réels dans un vecteur stochastique")
		if somme!=1:
			print("Données incorrectes : La somme des composantes d'un vecteur stochastique doit être égale à 1")
			ret=False
	else:
		ret=False
		print("Données incorrectes : PI0 n'est pas un vecteur de type list")
	return ret
"""
def verification_dimension(P,pi0):
	ret=True
	condition=isinstance(P,np.ndarray) # Verifier si P est bien un tableau de numpy
	if condition:
		if len(P.shape)==2: # Verifier si le tuplé renvoyé par la méthode shape à bien 2 élements pour éviter les bugs et vérifier dans un premier temps 
			if P.shape[0]==P.shape[1]: # Verifier si le nombre de lignes est bien égale au nombre de colonnes
				if isinstance(pi0,list) or isinstance(pi0,np.ndarray): # Verifier si le vecteur stochastique initiale est bien une liste (vecteur)
					if len(pi0)==P.shape[0]:
						ret=True
					else:
						ret=False
						print("Dimensions incorrectes : Le nombre de lignes du vecteur est different du nombre de colonnes de la matrice")
				else:
					ret=False
					print("Dimensions incorrectes : pi0 n'est pas un vecteur")
			else:
				ret=False
				print("Dimensions incorrectes : Le nombre de colonnes et le nombre de lignes sont different")
		else:
			ret=False
			print("Dimensions incorrectes : Le tableau rentré n'est pas une matrice")
	else:
		ret=False
		print("""Dimensions incorrectes : P n'est pas un tableau de type "numpy.ndarray" """)
	return ret

def stochastique (P):
	ret=True
	somme=0
	i=0
	while i<P.shape[0]:
		somme=0
		j=0
		while j<P.shape[1]:
			if P[i][j]<0:
				ret=False
				print("Erreur de donnée "+str(P[i][j])+" : Il ne peut pas y avoir un nombre négatif dans une matrice stochastique.")
			somme+=P[i][j]
			j+=1
		if somme!=1:
			print("Erreur : la ligne "+str(i+1)+" de la matrice n'est pas un vecteur stochastique.")
			ret=False
		i+=1
		somme=0
	return ret

def puits(P,i):
	ret=False
	if stochastique(P) and isinstance(i,int) and i<=P.shape[0]:
		if P[i-1][i-1]==1:
			ret=True
	else:
		print("Donnée incorrectes")
	return ret
			
def simulation(P, pi0, t0, tf):
# Simulation numerique d'une chaine de Markov en temps discret
# P	    : matrice de transition
# pi0	: vecteur stochastique initial (a l'instant t0)
# t0	: instant initial (debut de la simulation)
# tf	: instant final
# pi	: matrice des valeurs successives du vecteur stochastique
# t     : liste des instants (t0 <= t <= tf)
	t = np.arange(t0,tf+1)

	if isinstance(t0,int) and  isinstance(tf,int):
		if t0>tf:
			print("Le temps initial "+str(t0)+" ne peut pas être superieur au temps final "+str(tf))
		condition = verification_dimension(P,pi0) and stochastique(P) and t0<=tf
		if condition:
			# evolution du vecteur stochastique
			pi = np.array(np.zeros((len(t),P.shape[1]))) # On crée un tableau numpy remplie de 0 de taille (nombre d'élement dans le tableaux qui represente les temps, nombre de colonnes dans la matrice)
			pi[0] = pi0  
			for i in range(1,len(t)): 
				pi[i] = pi[i-1].dot(P)
			plt.plot(t,pi)
			bleu=mpatches.Patch(color='blue',label="composante 1")
			orange=mpatches.Patch(color='orange',label="composante 2")
			green=mpatches.Patch(color='green',label="composante 3")
			plt.legend(handles=[bleu,orange,green])
			plt.title("Evolution dans le temps du vecteur stochastique")
			plt.xlabel("Evolution du temps t")
			plt.ylabel("Evolution des composantes du vecteur stochastique")

			plt.show()
			return t,pi
	else:
		print(t0)
		print(tf)
		print("Les temps doivent être des nombres entier (de type int)")
	return 0,0


def verifier_si_egale(P):
	ret=True
	j=0
	while j<P.shape[1] and ret:
		i=1
		valeur_compare=round(P[0][j],8)
		while i<P.shape[0] and ret:
			if valeur_compare!=round(P[i][j],8):
				ret=False
			i+=1
		j+=1
	return ret

def créer_vecteur_stationnaire(P):
	res=[]
	j=0
	condition=verifier_si_egale(P)
	if not condition:
		print("La matrice envoyé ne permet pas de créer le vecteur stochastique stationnaire")
	while j<P.shape[1] and condition:
		res.append(round(P[0][j],8))
		j+=1
	return res	

def stationnaire (P):
	res=[]	 # Créer une liste pour y mettre le vecteur stochastique stationnaire
	P_puissance=P
	if  stochastique(P):
		if verifier_si_egale(P): #Cas ou P est une matrice qui permet de créer le vecteur stochastque stationnaire
			res=créer_vecteur_stationnaire(P)
		condition=True
		while condition: # Sinon on doit trouver la puissance N de la matrice P qui le permet
			P_puissance=np.dot(P_puissance,P)
			#time.sleep(1)
			if verifier_si_egale(P_puissance):
				condition=False
				res=créer_vecteur_stationnaire(P_puissance)

	else:
		print("La matrice P n'est pas stochastique")

	return res

# A faire si j'ai le temps
"""def stationnaire2 (P):
	res=[]
	if stochastique(P):
		Ident=np.eye(P.shape[0])
		tP=P.T
		piSt=[]
		i<0
		while i<len(piSt):
			j=0
			valeurs=0.0
			"""
				
def diagonalisation(P,pi0,t):
	if stochastique:
		print(np.linalg.eig(P))
		D,V=np.linalg.eig(P)
		det=np.linalg.det(V)
		print(D)
		input("Continue ")
		if det!=0:
			print("La matrice V est invertible")
			V1=np.linalg.inv(V)
			print("Calcul des 5 premieres puissances de P par les deux méthodes")
			print("Par le calcul classique avec P^n = P^n-1 . P")
			print("Par le calcul classique P^2")
			P2=np.dot(P,P)
			print(P2)
			print("Par le calcul classique P^3")
			P3=np.dot(P2,P)
			print(P3)
			print("Par le calcul classique P^4")
			P4=np.dot(P3,P)
			print(P4)
			print("Par le calcul classique P^5")
			P5=np.dot(P4,P)	
			print(P5)		
			print("Par l'autre méthode avec avec P^n = V.D^n.v^-1")
			print(" Par l'autre méthode P^2")
			D2=np.dot(D,D)
			P2a=np.dot(np.dot(V,D2),V1)
			print(P2a)
			print(" Par l'autre méthode P^3")
			D3=np.dot(D2,D)
			P3a=np.dot(np.dot(V,D3),V1)
			print(P3a)
			print(" Par l'autre méthode P^4")
			D4=np.dot(D3,D)
			P4a=np.dot(np.dot(V,D4),V1)
			print(P4a)
			print(" Par l'autre méthode P^5")
			D5=np.dot(D4,D)
			P5a=np.dot(np.dot(V,D5),V1)
			print(P5a)
		else:
			print("La matrice V n'est pas invertible")
		if D[0]==1:
			print("1 est bien la premiere valeur propre de la matrice P")
		i=1
		while i<P.shape[0]:
			if D[i]<1 and D[i]>0:
				print("La valeur propre n°"+str(i+1)+" a bien un module compris strictement entre 0 et 1")
			i+=1
		t,pi=simulation(P,pi0,0,t)
		i=0
		list_trier=[]
		while i<D.shape[0]:
			list_trier.append(D[i]) # On ajouter les valeur propres à une liste
			i+=1
		list_trier.sort(reverse=True)
		i=0
		valeurs_propre = np.array(np.zeros((len(t))))
		while i<len(t):
			valeurs_propre[i]=list_trier[1]
			i+=1
		input("Appuyez sur une touche pour continuer")
		plt.plot(t,pi)
		plt.plot(t,valeurs_propre,label="Module de la 2ème valeur propre")
		bleu=mpatches.Patch(color='blue',label="composante 1")
		orange=mpatches.Patch(color='orange',label="composante 2")
		green=mpatches.Patch(color='green',label="composante 3")
		red=mpatches.Patch(color='red',label="Module de la 2éme valeur propre")
		plt.legend(handles=[bleu,orange,green,red])
		plt.title("Evolution dans le temps et en fonction du module de la 2éme valeur propre du vecteur stochastique ")
		plt.xlabel("Evolution du temps t")
		plt.ylabel("Evolution des composantes du vecteur stochastique")
		plt.show()
	return
			

				
		

P = np.array([ [5/6 , 1/12, 1/12],  [ 1/4, 1/2, 1/4] , [ 1/4, 0, 3/4] ])
Pegale=np.array([ [1/4 , 1/2, 1/4],  [ 1/4, 1/2, 1/4] , [ 1/4,1/2,1/4] ])
pi0 = [1,0,0]
#pi0 = [1.0 , 0.0, 0.0]


Res1=0
Res2=0
Res1,Res2=simulation(P,pi0,0,15)
print("Simulation terminée")
print(stationnaire(Pegale))
List=stationnaire(P)
print(List)
print(List[0]+List[1]+List[2])
diagonalisation(P,pi0,5)
diagonalisation(P,pi0,10)
diagonalisation(P,pi0,50)
diagonalisation(P,pi0,100)




input()
os.system("cls")
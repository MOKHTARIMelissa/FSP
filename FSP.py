import numpy as np
import pandas as pd
from collections import deque

# Class FlowShop qui défint les méthodes de résolution du FSP
class FlowShop:

    def __init__(self, data = None):
        
        self.data = data
        self.N = data.shape[1]
        self.M = data.shape[0]
    
    # La fonction objective à minimiser - Appels recursifs (On suppose la numérotation commence de 1)
    def Cmax_r(self, num_machine, sequence):

        if((num_machine == 1) and (len(sequence) == 1)):
            return self.data[num_machine -1][int(sequence[0]) -1] 

        if(len(sequence) == 1):
            return self.Cmax_r(num_machine-1, sequence) + self.data[num_machine -1][int(sequence[-1]) -1]

        if(num_machine == 1):
            return self.Cmax_r(num_machine, sequence[:-1]) + self.data[num_machine -1][int(sequence[-1]) -1]

        return max(self.Cmax_r(num_machine,sequence[:-1]), self.Cmax_r(num_machine-1, sequence)) + self.data[num_machine -1][int(sequence[-1]) -1]

    # La fonction objective à minimiser - Programmation dynamique (On suppose la numérotation commence de 1)
    def Cmax(self, num_machine, sequence):
        cmax = np.zeros(self.data.shape)

        cmax[0,0] = self.data[0][int(sequence[0]) -1]

        for i in range(1, self.M):
            cmax[i,0] = cmax[i-1,0] + self.data[i][int(sequence[0]) -1]

        for i in range(1,len(sequence)):
            cmax[0,i] = cmax[0,i-1] + self.data[0][int(sequence[i]) -1]

        for i in range(1, self.M):
            for j in range(1,len(sequence)):
                cmax[i,j] = max(cmax[i,j-1], cmax[i-1,j]) + self.data[i][int(sequence[j]) -1]

        return cmax[num_machine-1, len(sequence)-1]
        

    # La fonction d'evaluation (LB)
    def lowerBound(self, sequence):
        
        LB = max(self.Cmax(j+1, sequence) +
             np.delete(self.data, sequence.astype(int)-1, 1)[j, :].sum() +
             min(np.delete(self.data, sequence.astype(int)-1, 1)[j+1:,i].sum() for i in range(self.N-len(sequence)))
          for j in range(self.M))

        return LB

    
    # Heuristique d'initialisation de la borne supérieure UP
    def initial_UB(self):
        
        # Générer tous les noeuds (Jobs) à partir de la racine
        sigma =  []
        sigmaBar = list(filter(lambda x: x not in sigma, np.arange(1, self.N + 1)))
        
        # Pour chaque niveau, ordonner les noeuds générés selon leurs évaluations LB, et choisir celui qui a le min des LB
        for i in range(self.N - 1):
            succ = []
            for node in sigmaBar:
                succ.append((node, self.lowerBound(np.array([*sigma, *[node]]))))
            sigmaBar = sorted(succ, key=lambda tab: tab[1])
            sigmaBar = [x[0] for x in sigmaBar]
            
            # Rajouter le noeud à sigma et générer ces successeurs
            sigma.append(sigmaBar[0])
            sigmaBar = list(filter(lambda x: x not in sigma, np.arange(1, self.N + 1)))

        return self.Cmax(self.M, np.array([*sigma, *sigmaBar]))
    
    # Fonction d'initialisation de la borne inférieure LB
    def initial_LB(self):
        
        LBt = max(self.data[j,:].sum() for j in range(self.M))

        LBj = max((min(self.data[:j, i].sum() for i in range(self.N)) + 
                   min(self.data[j+1:, i].sum() for i in range(self.N)) +
                   self.data[j, :].sum()) 
              for j in range(self.M))

        return max(LBt, LBj)
    
    # Recherche de la permutation optimale par l'algorithme : Branch and Bound
    def branchBound(self):
        
        optimal_sequence = np.array([])
        optimal_time = 0
        
        #Initialisation des noeuds de parcours et de la pile
        u = Node()
        v = Node(level = 0, path = np.array([]))
        PQ = deque()

        # Initilialisation des bornes
        min_length = self.initial_UB()
        v.bound = self.initial_LB()
        
        # mettre v(la racine) dans la pile
        PQ.append(v)
        
        while PQ:

            v = PQ.pop() #On dépile 
            if v.bound < min_length: #L'evaluation du noeud est plus petite que la solution optimale actuelle
                #On passe au niveau suivant et on génere les successeurs 
                u.level = v.level + 1 
                sigmaBar = list(filter(lambda x: x not in v.path, np.arange(1, self.N + 1)))
                
                #Si on n'est pas dans une feuille, on ordonne les noeuds selon LB
                if u.level != self.N: 
                    succ = []
                    for node in sigmaBar:
                        succ.append((node, self.lowerBound(np.array([*v.path, *[node]]))))
                    sigmaBar = sorted(succ, key=lambda tab: tab[1], reverse=True)
                    sigmaBar = [x[0] for x in sigmaBar]
                
                #on parcours les noeuds non deja visités
                for i in sigmaBar:
                    u.path = v.path
                    #On ajoute le noeud u au chemin du noued actuel u 
                    u.path = np.append(u.path, i)
                    
                    #Si on est dans une feuille, on calcule Cmax de sigma
                    if u.level == self.N: 
                        _len = self.Cmax(self.M, u.path[:]) 
                        
                        #Si on trouve une meilleure permutation, on met à jour les paramètres du parcours
                        if _len < min_length:
                            min_length = _len
                            optimal_time = _len
                            optimal_sequence = u.path[:]
                            
                    #Sinon, on évalue le noeud
                    else:
                        u.bound = self.lowerBound(u.path)
                        
                        #Si LB(u) < UB, on continue sinon on coupe 
                        if u.bound < min_length:
                            PQ.append(u)

                    u = Node(level=u.level)


        return optimal_sequence, optimal_time
        

# Class Node qui représente un Job et ces caractéristiques (Niveau, Chemin, Evaluation)
class Node(object):
    
    def __init__(self, level=None, path=None, bound=None):
        self.level = level
        self.path = path
        self.bound = bound

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
    
    
    # L'algorithme de Johnson (Appliqué qu'aux FSP avec 2 machines (Utilisé dans CDS))
    def johnson(self, data):
        # La séquence issu de la 1ere machine (Insertion au debut)
        machine_1_sequence = [j for j in range(self.N) if data[0][j] <= data[1][j]]
        machine_1_sequence.sort(key=lambda x: data[0][x])
        
        # La séquence issu de la 2eme machine (Insertion à la fin)
        machine_2_sequence = [j for j in range(self.N) if data[0][j] > data[1][j]]
        machine_2_sequence.sort(key=lambda x: data[1][x], reverse=True)
        
        # Regroupement des Jobs pour la construction de la sequence (On commence la numerotation à partir de 1)
        seq = machine_1_sequence + machine_2_sequence
        seq = np.array([j+1 for j in seq])
        
        return seq
    
    
    # Recherche de la permutation aprochée par l'heuristique : CDS
    def CDS(self):
        
        # Generation des (M-1) sequences
        seq = []
        for i in range(self.M-1):
            m1 = m2 = np.zeros(self.N)
            for j in range(i+1):
                m1 = m1 + self.data[j, :]
                m2 = m2 + self.data[self.M-1-j, :]
            
            # Recuperation de la permutation optimale pour chaque sequence (M1, M2) par l'algorithme de Johnson
            seq.append(self.johnson(np.vstack((m1, m2))))
        
        # Choix de la permutation qui minimise Cmax
        sigma = min(seq, key=lambda p: self.Cmax(self.M, p))
        cmax = self.Cmax(self.M, sigma)
       
        return sigma, cmax
    
    
    # Recherche de la permutation aprochée par l'heuristique : NEH
    def NEH(self):

        sums_jobs = []
        # Calculer la somme des durées d'éxecution de chaque job pour les m machines
        for job_id in range(self.N):
            sums_job = sum([self.data[j][job_id] for j in range(self.M)])
            sums_jobs.append((job_id, sums_job))

        # Ordonner les jobs par en ordre décroissant par rapport à la somme déja calculée
        sums_jobs.sort(key=lambda elem: elem[1], reverse=True)

        # Obtenir la sequence optimale
        sequence = []
        for job in sums_jobs:
            cands = []
            # Pour faire toutes les combainaisons possibles
            for i in range(0, len(sequence) + 1):
                cand = sequence[:i] + [job[0]] + sequence[i:]
                cands.append((cand, self.Cmax(self.M, [i+1 for i in cand])))
            sequence = min(cands, key=lambda x: x[1])[0]

        return [i+1 for i in sequence], self.Cmax(self.M, sequence)
    
    def palmer_heuristic(self):

        optimal_sequence=list(np.arange(0,self.N))
        
        #on calcule les poids des machines
        weight_machine=np.zeros(self.M)
        for j in np.arange(0,self.M):#on parcours les machines 
            weight_machine[j]=(self.M-2*(j+1)+1)
            
        #on ordonne selon l'index de pente
        weights=np.zeros(self.N)
        for i in range(self.N):
            for j in range(self.M):
                weights[i]+=weight_machine[j]*self.data[j,i]
        optimal_sequence=list(list(np.argsort(weights)+1))
        optimal_time=self.Cmax(self.M,optimal_sequence)
        return optimal_sequence,optimal_time    

# Class Node qui représente un Job et ces caractéristiques (Niveau, Chemin, Evaluation)
class Node(object):
    
    def __init__(self, level=None, path=None, bound=None):
        self.level = level
        self.path = path
        self.bound = bound

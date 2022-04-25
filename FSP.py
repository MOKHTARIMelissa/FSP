import numpy as np
import pandas as pd
import random
import math
from random import randint
import timeit

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
    
    
    # La fonction objective à minimiser - Programmation dynamique (On suppose la numérotation commence de 1)
    def CtMatrix(self, sequence):
        cmax = np.zeros(self.data.shape)

        cmax[0,0] = self.data[0][int(sequence[0]) -1]

        for i in range(1, self.M):
            cmax[i,0] = cmax[i-1,0] + self.data[i][int(sequence[0]) -1]

        for i in range(1,len(sequence)):
            cmax[0,i] = cmax[0,i-1] + self.data[0][int(sequence[i]) -1]

        for i in range(1, self.M):
            for j in range(1,len(sequence)):
                cmax[i,j] = max(cmax[i,j-1], cmax[i-1,j]) + self.data[i][int(sequence[j]) -1]

        return cmax
        

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
    
    
    def Palmer_MPI(self):

        optimal_sequence=list(np.arange(0,self.N))
        
        #on calcule les poids des machines
        weight_machine=np.zeros(self.M)
        for j in np.arange(0,self.M):#on parcours les machines 
            weight_machine[j]=(self.M-2*(j+1))
            
        #on ordonne selon l'index de pente
        
        #premiere seqquence +0
        weights=np.zeros(self.N)
        for i in range(self.N):
            for j in range(self.M):
                weights[i]+=weight_machine[j]*self.data[j,i]
        optimal_sequence=list(list(np.argsort(weights)+1))
        optimal_time=self.Cmax(self.M,optimal_sequence)
        
        # deuxième sequence +1 
        weights_1=np.zeros(self.N)
        for i in range(self.N):
            for j in range(self.M):
                weights_1[i]+=(weight_machine[j]+1)*self.data[j,i]
        optimal_sequence_1=list(list(np.argsort(weights_1)+1))
        optimal_time_1=self.Cmax(self.M,optimal_sequence_1)
        
        # troisième sequence +2
        weights_2=np.zeros(self.N)
        for i in range(self.N):
            for j in range(self.M):
                weights_2[i]+=(weight_machine[j]+2)*self.data[j,i]
        optimal_sequence_2=list(list(np.argsort(weights_2)+1))
        optimal_time_2=self.Cmax(self.M,optimal_sequence_2)
        
        #on choisit la meilleure sequence
        if(optimal_time_1<optimal_time):
            optimal_time=optimal_time_1
            optimal_sequence=optimal_sequence_1
        if(optimal_time_2<optimal_time):
            optimal_time=optimal_time_2
            optimal_sequence=optimal_sequence_2
        
        return optimal_sequence,optimal_time  
    
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
    def CDS(self, nbSeq = None):
        
        if (nbSeq == None): nbSeq = self.M-1
        
        # Generation des nbSeq sequences
        seq = []
        for i in range(nbSeq):
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
    
    
     # Recherche de la permutation aprochée par l'heurestique : NEH
    def NEH(self, order = True, pas = 1,  tie = "First"):
        sums_jobs = []

        # Calculer la somme des durées d'éxecution de chaque job pour les m machines
        for job_id in range(self.N):
            sums_job = sum([self.data[j][job_id] for j in range(self.M)])
            sums_jobs.append((job_id+1, sums_job))
        
        # Ordonner les jobs par en ordre décroissant par rapport à la somme déja calculée
        sums_jobs.sort(key=lambda x: x[1], reverse= order)
        
        # Obtenir la sequence approchee
        sequence = []
        for job in sums_jobs:
            cands = []
            # Pour faire toutes les combainaisons possibles
            for i in range(0, len(sequence) + 1, pas):
                cand = sequence[:i] + [job[0]] + sequence[i:]
                cands.append((cand, self.Cmax(self.M,cand)))
            # Prendre la premiere sequence avec le Cmax minimum
            if (tie == "Last") :
                cands.reverse()
                sequence = min(cands, key=lambda x: x[1])[0]
            elif (tie == "Random") :
                minimum = min(cands, key=lambda x: x[1])[1]
                cands_min = [cand_min for i, cand_min in enumerate(cands) if cand_min[1] == minimum]
                sequence = random.choice(cands_min)[0]
            else:
                sequence = min(cands, key=lambda x: x[1])[0]

        return sequence, min(cands, key=lambda x: x[1])[1]
    
    
    def NEH_ameliore(self, order = True, pas = 1,  tie = "First"):
        priority_jobs = []

        # Calculer la somme des durées d'éxecution de chaque job pour les m machines
        for job_id in range(self.N):
            AVGi = sum([self.data[j][job_id] for j in range(self.M)]) / self.M
            STDi = math.sqrt(sum([(self.data[j][job_id] - AVGi)**2 for j in range(self.M)]) / (self.M - 1))
            SKEi = (sum([(self.data[j][job_id] - AVGi)**3 for j in range(self.M)]) / self.M) / (math.sqrt(sum([(self.data[j][job_id] - AVGi)**2 for j in range(self.M)]) / (self.M)) ** 3)
            
            PRSKEi = AVGi + STDi + abs(SKEi)
            priority_jobs.append((job_id+1, PRSKEi))
        
        # Ordonner les jobs par en ordre décroissant par rapport à la somme déja calculée
        priority_jobs.sort(key=lambda x: x[1], reverse= order)
        
        # Obtenir la sequence approchee
        sequence = []
        for job in priority_jobs:
            cands = []
            # Pour faire toutes les combainaisons possibles
            for i in range(0, len(sequence) + 1, pas):
                cand = sequence[:i] + [job[0]] + sequence[i:]
                cands.append((cand, self.Cmax(self.M,cand)))
            
            if (tie == "Last"):
                cands.reverse()
                sequence = min(cands, key=lambda x: x[1])[0]

            elif (tie == "Random"):
                minimum = min(cands, key=lambda x: x[1])[1]
                cands_min = [cand_min for i, cand_min in enumerate(cands) if cand_min[1] == minimum]
                sequence = random.choice(cands_min)[0]

            elif (tie == "KK2"):
                a = np.sum([((self.M-1)*(self.M-2)/2 + self.M - j)*self.data[j-1][job[0]-1] for j in range(1,self.M+1)])
                b = np.sum([((self.M-1)*(self.M-2)/2 + j - 1)*self.data[j-1][job[0]-1] for j in range(1,self.M+1)])
                if (a >= b):
                    sequence = min(cands, key=lambda x: x[1])[0]
                else:
                    cands.reverse()
                    sequence = min(cands, key=lambda x: x[1])[0]

            elif (tie == "SMM"):
                minimum = min(cands, key=lambda x: x[1])[1]
                mins = [v[0] for i, v in enumerate(cands) if v[1] == minimum]
                SSMs = [(seq, np.mean(self.CtMatrix(seq)[:,len(seq)-1])) for seq in mins]
                sequence = min(SSMs, key=lambda x: x[1])[0]

            else: # tie == "First"
                sequence = min(cands, key=lambda x: x[1])[0]

        return sequence, min(cands, key=lambda x: x[1])[1]  
    
    # Simulated annealing Meta-heuristic
    #################################################################################################################
    
    def recuit_simule(self, init = "NEH", voisinage = "Insertion", TempUpdate = "Geometrique", palier = 1, nbsolrej = math.inf , nbItrMax = 5000, Ti = 700,Tf = 2 ,alpha = 0.9):
        
        #Nombre de jobs
        n = self.N
        
        #Initialisation de la solution initiale
        if (init == "Palmer"):
            old_seq,old_cmax = self.Palmer_MPI()
        elif (init == "NEH"):
            old_seq,old_cmax = self.NEH_ameliore(tie = "SMM")
        elif (init == "CDS"):
            old_seq,old_cmax = self.CDS()
            
        new_seq = []       
        delta = 0
        
        #Initialisation de la temperature initiale
        T = Ti
        
        # numero de l'iteration
        itr = 0
        
        # numero de l'iteration
        numsol_nonaccep = 0
        
        # Initialisation de la fonction de voisinage
        GenerVois = None
        if (voisinage == "Insertion"):
            GenerVois = self.Insertion
        elif (voisinage == "Swap"):
            GenerVois = self.Swap
        elif (voisinage == "Interchange"):
            GenerVois = self.Interchange
               
        # Initialisation de la fonction de maj de la temperature
        Tupdate = None
        if (TempUpdate == "Geometrique"):
            Tupdate = self.updateTempGeo
        elif (TempUpdate == "Linear"):
            Tupdate = self.updateTempLin
        elif (TempUpdate == "Slow"):
            Tupdate = self.updateTempSlow
                      
        while (T >= Tf and  itr <= nbItrMax) : 
            # Voisinage
            new_seq = GenerVois(old_seq)
                      
            new_cmax = self.Cmax(self.M, new_seq)
            delta = new_cmax - old_cmax
            if delta < 0:
                old_seq = new_seq
                old_cmax = new_cmax
                numsol_nonaccep = 0
            else :
                prob = np.exp(-(delta/T))
                if prob > np.random.uniform(0,1):
                    old_seq = new_seq
                    old_cmax = new_cmax
                    numsol_nonaccep = 0
                else :
                    #La solution est ignoree
                    numsol_nonaccep += 1
             
            if (numsol_nonaccep > nbsolrej) :
                T = self.updateTempGeo(T, 1/alpha)
            elif (itr%palier == 0) :
                T = Tupdate(T, alpha)
            itr += 1

        return old_seq, old_cmax
                      
      
    def Insertion(self, seq):
        job = seq.pop(randint(0,self.N-1)) 
        seq.insert(randint(0,self.N-1),job)
        return seq
        
    def Swap(self, seq):
        i = randint(0,self.N-2)
        seq[i+1], seq[i] = seq[i], seq[i+1]
        return seq
            
    def Interchange(self, seq):
        i1, i2 = random.sample(range(0, self.N - 1), 2)  
        seq[i1], seq[i2] = seq[i2], seq[i1]
        return seq

    def updateTempGeo(self, T, alpha):
        return T*alpha
    
    def updateTempLin(self, T, alpha):
        return T - alpha
    
    def updateTempSlow(self, T, beta):
        return T/(1 + beta*T) 
    
    
    
    # Itereative Local Search Meta-heuristic
    #################################################################################################################
    
    # Local Search
    def LS(self, best_sequence, best_cmax, neibourhoodType='swap', selectionStrategy = 'best'):
        
        # Swapping: Swap the element (i) with (i+1) ----- Neighbourhood size = N
        if (neibourhoodType == 'swap'):
            for i in range(self.N):
                sigma = best_sequence.copy()
                sigma[i], sigma[(i+1) % self.N] = sigma[(i+1) % self.N], sigma[i]
                
                 # Search for the best solution within the neighbourhood
                cmax = self.Cmax(self.M, sigma)
                if(cmax < best_cmax):
                    best_sequence, best_cmax = sigma, cmax
                    if(selectionStrategy == 'first'): 
                        return best_sequence, best_cmax
                
        # Interchanging: Swap the element (i) with (j) ----- Neighbourhood size = N(N-1)/2
        elif (neibourhoodType == 'interchange'):
            for i in range(self.N-1):
                for j in range(i+1, self.N):
                    sigma = best_sequence.copy()
                    sigma[i], sigma[j] = sigma[j], sigma[i]
                    
                    # Search for the best solution within the neighbourhood
                    cmax = self.Cmax(self.M, sigma)
                    if(cmax < best_cmax):
                        best_sequence, best_cmax = sigma, cmax
                        if(selectionStrategy == 'first'): 
                            return best_sequence, best_cmax
                    
        # Insertion: Swap the element (i) with (j) ----- Neighbourhood size = (N-1)²
        elif (neibourhoodType == 'insertion'):
            for i in range(self.N):
                if i==0: sup = self.N
                else: sup = i+self.N-1
                for j in range(i+1, sup):   
                    sigma = best_sequence.copy()
                    save = sigma[i]
                    sigma.pop(i)
                    sigma.insert(j%self.N, save)
                    
                    # Search for the best solution within the neighbourhood
                    cmax = self.Cmax(self.M, sigma)
                    if(cmax < best_cmax):
                        best_sequence, best_cmax = sigma, cmax
                        if(selectionStrategy == 'first'): 
                            return best_sequence, best_cmax
            
        return best_sequence, best_cmax
    
    # Perturbation
    def perturbation(self, sequence, perturbationType="random"):
        
        if (perturbationType == "random"):
            random.shuffle(sequence)
            
        elif (perturbationType == "swap"):
            pos1 = random.randint(0, len(sequence)-1)
            j = -1
            while (j < 0):
                pos2=random.randint(0, len(sequence)-1)
                if (pos1 != pos2): j = 1
            sequence[pos1], sequence[pos2] = sequence[pos2], sequence[pos1] 
                
        elif (perturbationType == "insertion"):
            element = random.randint(0, len(sequence)-1)
            save = sequence[element]
            j = -1
            while (j < 0):
                pos = random.randint(0, len(sequence)-1)
                if (element != pos): j = 1
            sequence.pop(element)
            sequence.insert(pos, save)
            
        return sequence
    
    # Iterative Local Search
    def ILS(self, init = "NEH", neibourhoodType='insertion', selectionStrategy = 'best', perturbationType="insertion", stopCriteria = 'iteration', maxCriteria = 100):
        
        #Initialization 
        if (init == "Palmer"):
            sequence, cmax = self.Palmer_MPI()
        elif (init == "NEH"):
            sequence, cmax = self.NEH_ameliore(tie = "SMM")
        elif (init == "CDS"):
            sequence, cmax = self.CDS()
        
        best_sequence, best_cmax  = self.LS(sequence, cmax, neibourhoodType, selectionStrategy)
        
        
        if (stopCriteria == 'iteration'): 
            criteria = 0
        elif (stopCriteria == 'duration'): 
            start = timeit.default_timer()
            criteria = timeit.default_timer() - start
            
        while(criteria < maxCriteria ):
            sequence = self.perturbation(best_sequence, perturbationType)
            cmax = self.Cmax(self.M, sequence)
            new_sequence, new_cmax = self.LS(sequence, cmax, neibourhoodType, selectionStrategy)
            
            if(new_cmax < best_cmax):
                best_sequence, best_cmax = new_sequence, new_cmax
            
            if (stopCriteria == 'iteration'): 
                criteria += 1
            elif (stopCriteria == 'duration'):
                criteria = timeit.default_timer() - start
            
        return best_sequence, best_cmax

# Class Node qui représente un Job et ces caractéristiques (Niveau, Chemin, Evaluation)
class Node(object):
    
    def __init__(self, level=None, path=None, bound=None):
        self.level = level
        self.path = path
        self.bound = bound

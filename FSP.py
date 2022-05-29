import numpy as np
import pandas as pd
import random
import math
from random import randint
import timeit
import os
import time
from functools import reduce
import numpy as np


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
    
    def recuit_simule(self, seq=None,init = "NEH", voisinage = "Insertion", TempUpdate = "Geometrique", palier = 1, nbsolrej = math.inf , nbItrMax = 5000, Ti = 700,Tf = 2 ,alpha = 0.9):
        
        #Nombre de jobs
        n = self.N
        
        #Initialisation de la solution initiale
        if (init == "Palmer"):
            seq,old_cmax = self.Palmer_MPI()
        elif (init == "NEH"):
            seq,old_cmax = self.NEH_ameliore(tie = "SMM")
        elif (init == "CDS"):
            seq,old_cmax = self.CDS()
            seq = seq.tolist()
        elif init=="":
            old_cmax=self.Cmax(self.M,seq)
            
            
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
            new_seq = GenerVois(seq)
                      
            new_cmax = self.Cmax(self.M, new_seq)
            delta = new_cmax - old_cmax
            if delta < 0:
                seq = new_seq
                old_cmax = new_cmax
                numsol_nonaccep = 0
            else :
                prob = np.exp(-(delta/T))
                if prob > np.random.uniform(0,1):
                    seq = new_seq
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

        return seq, old_cmax
                      
      
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
    def ILS(self, init = "NEH",seq=None, neibourhoodType='insertion', selectionStrategy = 'best', perturbationType="insertion", stopCriteria = 'iteration', maxCriteria = 100):
        
        #Initialization 
        if (init == "Palmer"):
            seq, cmax = self.Palmer_MPI()
        elif (init == "NEH"):
            seq, cmax = self.NEH_ameliore(tie = "SMM")
        elif (init == "CDS"):
            seq, cmax = self.CDS()
        elif init=="":
            cmax=self.Cmax(self.M,seq)
        
        best_sequence, best_cmax  = self.LS(seq, cmax, neibourhoodType, selectionStrategy)
        
        
        if (stopCriteria == 'iteration'): 
            criteria = 0
        elif (stopCriteria == 'duration'): 
            start = timeit.default_timer()
            criteria = timeit.default_timer() - start
            
        while(criteria < maxCriteria ):
            seq = self.perturbation(best_sequence, perturbationType)
            cmax = self.Cmax(self.M, seq)
            new_sequence, new_cmax = self.LS(seq, cmax, neibourhoodType, selectionStrategy)
            
            if(new_cmax < best_cmax):
                best_sequence, best_cmax = new_sequence, new_cmax
            
            if (stopCriteria == 'iteration'): 
                criteria += 1
            elif (stopCriteria == 'duration'):
                criteria = timeit.default_timer() - start
            
        return best_sequence, best_cmax
    
    # Genetic Algorithm
    #####################################################################################################################################
    def genetic_algorithm(self,seq=None, population_number,nb_stag_max=50,inter_population_number=100,it_number=5000, p_crossover=1.0, p_mutation=1.0,mode_init="random",mode_parent_selection="random",mode_mutation="swap",mode_update="enfants_population",mode_arret="and",mode_crossover="2_points", mode_sorti="None"):
        if population_number is None:
            population_number = self.N**2
        if inter_population_number is None:
            inter_population_number=population_number
        optimal = [4534, 920, 1302]
        opt = 0
        no_of_jobs, no_of_machines = self.N, self.M
        processing_time = []
        for i in range(no_of_jobs):
            temp = []
            for j in range(no_of_machines):
                temp.append(self.data[j][i])
            processing_time.append(temp)
        # generate an initial population proportional to no_of_jobs
        number_of_population = population_number

        # Initialize population
        population = self.initialize_population(number_of_population,mode_init,seq)
        costed_population = []
        
        for individual in population:
            ind_makespan = (self.Cmax(self.M,individual), individual)
            costed_population.append(ind_makespan)
        costed_population.sort(key=lambda x: x[0])
        
        #Initialize best solution
        seq = list(map(int, costed_population[0][1]))
        makespan = self.Cmax(self.M, seq) 
        
        nb_it=0
        nb_stag=0
        while self.critere_arret(nb_it,it_number,nb_stag,nb_stag_max,mode=mode_arret):
            
            # Select parents
            parent_list,parent = self.select_parent(population,inter_population_number, mode_parent_selection)
            childs = []

            # Apply crossover to generate children
            for parents in parent_list:
                r = np.random.rand()
                
                if r < p_crossover:
                    childs.append(self.crossover(parents,mode=mode_crossover))
                    childs.append(self.crossover((parents[1],parents[0]),mode=mode_crossover))
                else:
                    childs.append(parents[0])
                    childs.append(parents[1])

            # Apply mutation operation to change the order of the n-jobs
            mutated_childs = []
            for child in childs:
                r = np.random.rand()
                if r < p_mutation:
                    mutated_child = self.mutation(child,mode=mode_mutation)
                    mutated_childs.append(mutated_child)

            childs.extend(mutated_childs)
            
            #update population
            if len(childs) > 0:
                    population=self.update_population(population,parent,childs,mode_update)
                    
            #verify stagnation
            makespan_an=makespan
            costed_population = []
            for individual in population:
                ind_makespan = (self.Cmax(self.M,individual), individual)
                costed_population.append(ind_makespan)
            costed_population.sort(key=lambda x: x[0])
            seq = list(map(int, costed_population[0][1]))
            makespan = self.Cmax(self.M, seq)
            if makespan_an<=makespan:
                nb_stag+=1
            else:
                nb_stag=0
                
            nb_it+=1
        #hybridation with LS or RS
        if mode_sorti=="recuit_simule":
            pop = []
            for individual in population:
                ind = self.recuit_simule(seq=list(individual),init ="")[0]
                pop.append(ind)
            
        else:
            if mode_sorti=="local_search":
                pop = []
                for individual in population:
                    ind = self.LS(list(individual),self.Cmax(self.M,individual))[0]
                    pop.append(ind)   
            else:
                pop=population
            
        costed_population = []
        for individual in pop:
            ind_makespan = (self.Cmax(self.M,list(individual)), individual)
            costed_population.append(ind_makespan)
        costed_population.sort(key=lambda x: x[0])

        seq = list(map(int, costed_population[0][1]))
        makespan = self.Cmax(self.M, seq)  
        return seq, makespan 

    
    def initialize_population(self,population_size,mode="random",seq=None):
        number_of_jobs=self.N
        if mode=="init":
            population = []
            i = 0
            j=0
            #Use the initial seq to be the first ind to work with
            population.append(list(seq))
            #generate (n-1) ind using mutation 
            while i < (population_size-1):
                ind=self.mutation(list(seq))
                if(ind not in population or j>20):
                    population.append(ind)
                    i+=1
                    j=0
                else:
                    j+=1
            
        if mode=="random" :
            population = []
            i = 0
            j=0
            while i < population_size:
                individual = list(np.random.permutation(number_of_jobs))
                for pos in range(number_of_jobs):
                    individual[pos]=individual[pos]+1
                if individual not in population or j>20:
                    population.append(individual)
                    i += 1
                    j=0
                else:
                    j+=1
                    
        elif mode=="palmer":
            population = []
            i = 0
            j=0
            #generate the first ind to work with
            seq, cmax = self.palmer_heuristic()
            population.append(list(seq))
            #generate (n-1) ind using mutation 
            while i < (population_size-1):
                ind=self.mutation(list(seq))
                if(ind not in population or j>20):
                    population.append(ind)
                    i+=1
                    j=0
                else:
                    j+=1
            
        elif mode=="NEH":                               
            population = []
            i = 0
            j=0
            seq, cmax = self.NEH_ameliore()
            population.append(list(seq))
            while i < (population_size-1):
                ind=self.mutation(list(seq))
                if(ind not in population or j>20 ):
                    population.append(ind)
                    i+=1
                    j=0
                else:
                    j+=1
        elif mode=="CDS":  
            
            population = []
            i = 0
            j=0
            seq, cmax = self.CDS()
            population.append(list(seq))
            while i < (population_size-1):
                ind=self.mutation(list(seq))
                if(ind not in population or j>20):
                    population.append(ind)
                    i+=1
                    j=0
                else:
                    j+=1

        return population

    # Two-point/one-point crossover is that the set of jobs between 
    # two randomly selected points is always inherited from one parent to the child, 
    # and the other jobs are placed in the same manner as the one-point crossover. 
    def crossover(self,parents,mode="2_points"):
        parent1 = parents[0]
        parent2 = parents[1]
        if mode=="2_points":
            length_of_parent = self.N
            first_point = int(length_of_parent / 2 - length_of_parent / 4)
            second_point = int(length_of_parent - first_point)

            intersect = parent1[first_point:second_point]
            child = []
            index = 0
            for pos2 in range(len(parent2)):
                if first_point <= index < second_point:
                    child.extend(intersect)
                    index = second_point
                if parent2[pos2] not in intersect:
                        child.append(parent2[pos2])
                        index += 1
            return list(child)
        elif mode=="1_point":
            length_of_parent = self.N
            first_point = int(length_of_parent / 2)               
            intersect = parent1[:first_point]
            child = intersect
            for pos in range(len(parent2)):
                if parent1[pos] not in child:
                     child.append(parent1[pos])
            return list(child)

    # apply mutation to an existing solution using swap, interchange,Ls,RS
    def mutation(self,solution,mode="swap"):
        
        if mode=="interchange":
            # pick 2 positions i and j to swap randomly
            return list(self.Interchange(solution))
        elif mode=="swap":
            # pick 2 positions i and i+1 to interchange randomly
            return list(self.Swap(solution))
        elif mode=="local_search":
            #LS on the neighberhod of the ind
            sol,c=self.LS(list(solution),self.Cmax(self.M,solution))
            return list(sol)
        elif mode=="recuit_simule":
            #apply RS to the ind
            sol,c=self.recuit_simule(seq=list(solution),init ="")
            return list(sol)

    # Selects parent 
    def select_parent(self,population,inter_population_number,mode="tournois"):
        parents=[]
        processing_time=self.data
        number_of_jobs=self.N
        number_of_machines=self.M
        #randomly chosing 2 parents and then choose the best one as the first parent and then choose another one 
        if mode=="tournois":
            parent_pairs = []
            # randomly choose how many parent pairs will be selected
            parent_pair_count = random.randint(2, int(len(population)/2))
            for k in range(parent_pair_count):
                parent1 = self.binary_tournament(population)
                parent2 = self.binary_tournament(population)
                if parent1 != parent2 and (parent1, parent2) not in parent_pairs and (parent2, parent1) not in parent_pairs:
                    parent_pairs.append((parent1, parent2))
                    if parent1 not in parents:
                        parents.append(parent1)
                    if parent2 not in parents:
                        parents.append(parent2)
         #randomly choose parents
        elif mode=="random":
            parent_pairs = []
            # randomly choose how many parent pairs will be selected
            parent_pair_count = random.randint(2, int(len(population)/2))
            for k in range(parent_pair_count):
                parent1 = random.choice(population)
                parent2 = random.choice(population)
                if parent1 != parent2 and (parent1, parent2) not in parent_pairs and (parent2, parent1) not in parent_pairs:
                    parent_pairs.append((parent1, parent2))
                    if parent1 not in parents:
                        parents.append(parent1)
                    if parent2 not in parents:
                        parents.append(parent2)
        #we choose the best individuals as parents 
        elif mode=="elit":
            parent_pairs=[]
            parent_pair_count = random.randint(2, int(len(population)/2))
            pop_cmax=[self.Cmax(number_of_machines,indiv) for indiv in population]
            ordre=np.argsort(pop_cmax)
            pop=[population[pos] for pos in ordre[:inter_population_number] ]

            for k in range(parent_pair_count):
                parent1 = random.choice(pop)
                parent2 = random.choice(pop)
                if parent1 != parent2 and (parent1, parent2) not in parent_pairs and (parent2, parent1) not in parent_pairs:
                    parent_pairs.append((parent1, parent2))
                    if parent1 not in parents:
                        parents.append(parent1)
                    if parent2 not in parents:
                        parents.append(parent2)                  
        #the weight of each indiv is =the max of Cmax in the population - the Cmax of the indiv +1
        elif mode=="roulette":
            parent_pairs=[]
            parent_pair_count = random.randint(2, int(len(population)/2))
            pop_cmax=[self.Cmax(number_of_machines,indiv) for indiv in population]   
            pop_cmax=np.max(pop_cmax)-pop_cmax+1

            for k in range(parent_pair_count):
                parent1 = random.choices(population, weights=pop_cmax,k=1)[0]
                parent2 = random.choices(population, weights=pop_cmax,k=1)[0]
                if parent1 != parent2 and (parent1, parent2) not in parent_pairs and (parent2, parent1) not in parent_pairs:
                    parent_pairs.append((parent1, parent2))
                    if parent1 not in parents:
                        parents.append(parent1)
                    if parent2 not in parents:
                        parents.append(parent2)  
        #the weight of each ind is its rank( the best has the rank N , and the worst 1)
        elif mode=="ranking":
            parent_pairs=[]
            parent_pair_count = random.randint(2, int(len(population)/2))
            pop_cmax=[self.Cmax(number_of_machines,indiv) for indiv in population]   
            ordre=np.argsort(pop_cmax)
            ordre=np.flip(ordre)
            pop=[population[pos] for pos in ordre[:inter_population_number]]
            
            for k in range(parent_pair_count):
                parent1 = random.choices(pop, weights=(ordre[:inter_population_number]+1),k=1)[0]
                parent2 = random.choices(pop, weights=(ordre[:inter_population_number]+1),k=1)[0]
                if parent1 != parent2 and (parent1, parent2) not in parent_pairs:
                    parent_pairs.append((parent1, parent2))
                    if parent1 not in parents:
                        parents.append(parent1)
                    if parent2 not in parents:
                        parents.append(parent2)    
        return parent_pairs,parents        

    def binary_tournament(self, population):
        parent = []
        number_of_jobs=self.N
        number_of_machines=self.M
        processing_time=self.data
        candidates = random.sample(population, 2)
        makespan1 = self.Cmax(self.M,candidates[0])
        makespan2 = self.Cmax(self.M,candidates[1])
        if makespan1 < makespan2:
            parent = list(candidates[0])
        else:
            parent = list(candidates[1])
        #print("parent binarry",parent)
        return parent
    
    #To update the population after each iteration
    def update_population(self,population,parents,children,mode="enfants_population"):
        processing_time=self.data
        no_of_jobs=self.N
        no_of_machines=self.M
        #The childreen replace the parents 
        if mode=="enfants":
            costed_children=[]
            for individual in children:
                ind_makespan = (self.Cmax(self.M,individual), individual)
                costed_children.append(ind_makespan)
            costed_children.sort(key=lambda x: x[0])
            j=0
            for i in range(min(len(parents),len(children))):
                if parents[i] in population:
                    population.remove(parents[i])
                    population.append(costed_children[j][1])
                    j+=1
            
            return population
        # We chose the best beetwen childreen and actuel population
        elif mode=="enfants_population":
            costed_population = []
            for individual in population:
                ind_makespan = (self.Cmax(self.M,individual), individual)
                costed_population.append(ind_makespan)

            for individual in children:
                ind_makespan = (self.Cmax(self.M,individual), individual)
                costed_population.append(ind_makespan) 
            costed_population.sort(key=lambda x: x[0])    
            lenght=len(population)
            new_pop=[]
            for i in range(len(population)):
                new_pop.append(costed_population[i][1])
            return new_pop
        
        #the children replace the worst indiv in the population 
        elif mode=="least_good":
            
            costed_population = []
            for individual in population:
                ind_makespan = (self.Cmax(self.M,individual), individual)
                costed_population.append(ind_makespan)
            costed_population.sort(key=lambda x: x[0], reverse=True)

            costed_children = []
            for individual in children:
                ind_makespan = (self.Cmax(self.M,individual), individual)
                costed_children.append(ind_makespan)
            costed_children.sort(key=lambda x: x[0])
            
            if len(children)>= int(0.5*len(population)):
                lenght=int(0.5*len(population))
            else:
                lenght=int(len(children))
          
            #on supprime les moins bons et on les remplace par les enfants
            for child in costed_children[:lenght]:
                if (list(child[1]) not in population):
                    population.remove(costed_population[0][1])
                    costed_population.remove(costed_population[0])       
                    population.append(list(child[1]))
                #on peut ajouter une verification si les enfants sont toujours plus bons que la population sinon break
        return population 
        

    #Stop cond of the iterations.    
    def critere_arret(self,nb_it,nb_it_max,nb_stag,nb_stag_max,mode="and"):
        if mode=="and":
            return (nb_it<nb_it_max and nb_stag<nb_stag_max)
        elif mode=="stagnation":
            return (nb_stag<nb_stag_max)
        elif mode=="iteration":
            return (nb_it<nb_it_max)
        
        
    
    
# Class Node qui représente un Job et ces caractéristiques (Niveau, Chemin, Evaluation)
class Node(object):
    
    def __init__(self, level=None, path=None, bound=None):
        self.level = level
        self.path = path
        self.bound = bound

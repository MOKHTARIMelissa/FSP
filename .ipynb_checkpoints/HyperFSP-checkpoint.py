import random
import pickle
import os
import FSP as fsp
import RandomSearch
import numpy as np
import pandas as pd
import timeit
class HyperFSP:
    
    def __init__(self, data="None"):

        self.data = data
        #self.N = data.shape[1]
        #self.M = data.shape[0]
    
    
    # Itereative Local Search Meta-heuristic
    #################################################################################################################
    
    # Local Search
    def LS(self, nbHeurstic, best_sequence, best_cmax, best_seqFSP, heuristics, parameters, neibourhoodType='swap', selectionStrategy = 'best', nb_iter_LS = 10):
        count = 0;
        while (count < nb_iter_LS) :
            # Swapping: Swap the element (i) with (i+1) ----- Neighbourhood size = N
            if (neibourhoodType == 'swap'):
                for i in range(nbHeurstic):
                    sigma = best_sequence.copy()
                    sigma[i], sigma[(i+1) % nbHeurstic] = sigma[(i+1) % nbHeurstic], sigma[i]

                    # Search for the best solution within the neighbourhood
                    seqFSP,  cmax = self.cmaxSeq(sigma, heuristics, parameters)

                    if(cmax < best_cmax):
                        best_sequence, best_cmax, best_seqFSP = sigma, cmax, seqFSP
                        if(selectionStrategy == 'first'): 
                            return best_sequence, best_cmax, best_seqFSP

            # Interchanging: Swap the element (i) with (j) ----- Neighbourhood size = N(N-1)/2
            elif (neibourhoodType == 'interchange'):
                for i in range(nbHeurstic -1):
                    for j in range(i+1, nbHeurstic):
                        sigma = best_sequence.copy()
                        sigma[i], sigma[j] = sigma[j], sigma[i]

                        # Search for the best solution within the neighbourhood
                        seqFSP,  cmax = self.cmaxSeq(sigma, heuristics, parameters)
                        if(cmax < best_cmax):
                            best_sequence, best_cmax, best_seqFSP = sigma, cmax, seqFSP
                            if(selectionStrategy == 'first'): 
                                return best_sequence, best_cmax, best_seqFSP

            # Insertion: Swap the element (i) with (j) ----- Neighbourhood size = (N-1)²
            elif (neibourhoodType == 'insertion'):
                for i in range(nbHeurstic):
                    if i==0: sup = nbHeurstic
                    else: sup = i+nbHeurstic-1
                    for j in range(i+1, sup):   
                        sigma = best_sequence.copy()
                        save = sigma[i]
                        sigma.pop(i)
                        sigma.insert(j%nbHeurstic, save)

                        # Search for the best solution within the neighbourhood
                        seqFSP,  cmax = self.cmaxSeq(sigma, heuristics, parameters)
                        if(cmax < best_cmax):
                            best_sequence, best_cmax, best_seqFSP = sigma, cmax, seqFSP
                            if(selectionStrategy == 'first'): 
                                return best_sequence, best_cmax, best_seqFSP
            
            count = count + 1
            
        return best_sequence, best_cmax, best_seqFSP
    
    # Perturbation
    def perturbation(self, sequence, heuristics):
        i = random.randint(0,len(sequence) -1)
        j = random.randint(0,len(heuristics)-1)
        sequence[i] = j
        return sequence
    
    # Iterative Local Search
    def ILS(self,sequence, nbHeurstic, heuristics, parameters, neibourhoodType='insertion', selectionStrategy = 'best', stopCriteria = 'iteration', maxCriteria = 100, nb_iter_LS = 1):
        
        seqFSP, cmax = self.cmaxSeq(sequence, heuristics, parameters)
        best_sequence, best_cmax, best_seqFSP  = self.LS(nbHeurstic, sequence, cmax, seqFSP, heuristics, parameters, neibourhoodType, selectionStrategy, nb_iter_LS)
         
        if (stopCriteria == 'iteration'): 
            criteria = 0
        elif (stopCriteria == 'duration'): 
            start = timeit.default_timer()
            criteria = timeit.default_timer() - start
            
        while(criteria < maxCriteria ):
            sequence = self.perturbation(best_sequence, heuristics)
            seqFSP, cmax = self.cmaxSeq(sequence, heuristics, parameters)
            new_sequence, new_cmax, new_seqFSP = self.LS(nbHeurstic, sequence, cmax, seqFSP, heuristics, parameters, neibourhoodType, selectionStrategy, nb_iter_LS)

            if(new_cmax < best_cmax):
                best_sequence, best_cmax, best_seqFSP = new_sequence, new_cmax, new_seqFSP
                
            if (stopCriteria == 'iteration'): 
                criteria += 1
            elif (stopCriteria == 'duration'):
                criteria = timeit.default_timer() - start
            
            
        best_heurestic = [heuristics[best_sequence[i]] for i in range(len(best_sequence))]
        best_param = [parameters[best_sequence[i]] for i in range(len(best_sequence))]
        
        return best_heurestic, best_param, best_cmax, best_seqFSP
                            
                            
    def cmaxSeq(self, sequence, heuristics, parameters):
            
        for i in range(len(sequence)):
            if (i == 0):
                seqFSP, cmax = heuristics[sequence[i]](**parameters[sequence[i]])
            else:
                parameters[sequence[i]]["seq"] = seqFSP
                seqFSP, cmax = heuristics[sequence[i]](**parameters[sequence[i]])
                            
        return seqFSP, cmax
    
    def entrainement(self,nbHeurstic=2, neibourhoodType='insertion', selectionStrategy = 'best', stopCriteria = 'duration', maxCriteria = 1, nb_iter_LS = 1):
        #on parcours les fichiers contenant les instances
        files=os.listdir("./Instances_Taillard_Entrainement")
        for filename in files:

            data=np.loadtxt("./Instances_Taillard_Entrainement/"+filename)
            print(filename)
            flowshop = fsp.FlowShop(data)
            heuristics = [
                          flowshop.recuit_simule, flowshop.recuit_simule, flowshop.recuit_simule, flowshop.recuit_simule, flowshop.recuit_simule, flowshop.recuit_simule, flowshop.recuit_simule, flowshop.recuit_simule, flowshop.recuit_simule, 

                          flowshop.ILS, flowshop.ILS, flowshop.ILS, flowshop.ILS, flowshop.ILS, flowshop.ILS, flowshop.ILS, flowshop.ILS, flowshop.ILS, 
                          flowshop.ILS, flowshop.ILS, flowshop.ILS, flowshop.ILS, flowshop.ILS, flowshop.ILS, flowshop.ILS, flowshop.ILS, flowshop.ILS,   

                          flowshop.genetic_algorithm
                         ]

            parameters = [
                # Recuit simulé
                { "init": "NEH", "voisinage": "Insertion", "TempUpdate": "Geometrique", "palier": 2, "nbsolrej": 40, "nbItrMax": 5200, "Ti": 550, "Tf": 4, "alpha": 0.3},
                { "init": "NEH", "voisinage": "Insertion", "TempUpdate": "Linear", "palier": 2, "nbsolrej": 40, "nbItrMax": 5200, "Ti": 550, "Tf": 4, "alpha": 0.3},
                { "init": "NEH", "voisinage": "Insertion", "TempUpdate": "Slow", "palier": 2, "nbsolrej": 40, "nbItrMax": 5200, "Ti": 550, "Tf": 4, "alpha": 0.3},
                { "init": "NEH", "voisinage": "Swap", "TempUpdate": "Geometrique", "palier": 2, "nbsolrej": 40, "nbItrMax": 5200, "Ti": 550, "Tf": 4, "alpha": 0.3},
                { "init": "NEH", "voisinage": "Swap", "TempUpdate": "Linear", "palier": 2, "nbsolrej": 40, "nbItrMax": 5200, "Ti": 550, "Tf": 4, "alpha": 0.3},
                { "init": "NEH", "voisinage": "Swap", "TempUpdate": "Slow", "palier": 2, "nbsolrej": 40, "nbItrMax": 5200, "Ti": 550, "Tf": 4, "alpha": 0.3},
                { "init": "NEH", "voisinage": "Interchange", "TempUpdate": "Geometrique", "palier": 2, "nbsolrej": 40, "nbItrMax": 5200, "Ti": 550, "Tf": 4, "alpha": 0.3},
                { "init": "NEH", "voisinage": "Interchange", "TempUpdate": "Linear", "palier": 2, "nbsolrej": 40, "nbItrMax": 5200, "Ti": 550, "Tf": 4, "alpha": 0.3},
                { "init": "NEH", "voisinage": "Interchange", "TempUpdate": "Slow", "palier": 2, "nbsolrej": 40, "nbItrMax": 5200, "Ti": 550, "Tf": 4, "alpha": 0.3},

                # ILS
                { "init": "NEH", "neibourhoodType": 'insertion', "selectionStrategy": 'best', "perturbationType": "insertion", "stopCriteria": 'iteration', "maxCriteria": 200  },
                { "init": "NEH", "neibourhoodType": 'insertion', "selectionStrategy": 'best', "perturbationType": "swap", "stopCriteria": 'iteration', "maxCriteria": 200  },
                { "init": "NEH", "neibourhoodType": 'insertion', "selectionStrategy": 'best', "perturbationType": "random", "stopCriteria": 'iteration', "maxCriteria": 200  },
                { "init": "NEH", "neibourhoodType": 'insertion', "selectionStrategy": 'first', "perturbationType": "insertion", "stopCriteria": 'iteration', "maxCriteria": 200  },
                { "init": "NEH", "neibourhoodType": 'insertion', "selectionStrategy": 'first', "perturbationType": "swap", "stopCriteria": 'iteration', "maxCriteria": 200  },
                { "init": "NEH", "neibourhoodType": 'insertion', "selectionStrategy": 'first', "perturbationType": "random", "stopCriteria": 'iteration', "maxCriteria": 200  },
                { "init": "NEH", "neibourhoodType": 'swap', "selectionStrategy": 'best', "perturbationType": "insertion", "stopCriteria": 'iteration', "maxCriteria": 200  },
                { "init": "NEH", "neibourhoodType": 'swap', "selectionStrategy": 'best', "perturbationType": "swap", "stopCriteria": 'iteration', "maxCriteria": 200  },
                { "init": "NEH", "neibourhoodType": 'swap', "selectionStrategy": 'best', "perturbationType": "random", "stopCriteria": 'iteration', "maxCriteria": 200  },
                { "init": "NEH", "neibourhoodType": 'swap', "selectionStrategy": 'first', "perturbationType": "insertion", "stopCriteria": 'iteration', "maxCriteria": 200  },
                { "init": "NEH", "neibourhoodType": 'swap', "selectionStrategy": 'first', "perturbationType": "swap", "stopCriteria": 'iteration', "maxCriteria": 200  },
                { "init": "NEH", "neibourhoodType": 'swap', "selectionStrategy": 'first', "perturbationType": "random", "stopCriteria": 'iteration', "maxCriteria": 200  },
                { "init": "NEH", "neibourhoodType": 'interchange', "selectionStrategy": 'best', "perturbationType": "insertion", "stopCriteria": 'iteration', "maxCriteria": 200  },
                { "init": "NEH", "neibourhoodType": 'interchange', "selectionStrategy": 'best', "perturbationType": "swap", "stopCriteria": 'iteration', "maxCriteria": 200  },
                { "init": "NEH", "neibourhoodType": 'interchange', "selectionStrategy": 'best', "perturbationType": "random", "stopCriteria": 'iteration', "maxCriteria": 200  },
                { "init": "NEH", "neibourhoodType": 'interchange', "selectionStrategy": 'first', "perturbationType": "insertion", "stopCriteria": 'iteration', "maxCriteria": 200  },
                { "init": "NEH", "neibourhoodType": 'interchange', "selectionStrategy": 'first', "perturbationType": "swap", "stopCriteria": 'iteration', "maxCriteria": 200  },
                { "init": "NEH", "neibourhoodType": 'interchange', "selectionStrategy": 'first', "perturbationType": "random", "stopCriteria": 'iteration', "maxCriteria": 200  },

                # AG
                {  "population_number": 200, "nb_stag_max" : 40, "it_number" : 120, "p_crossover": 0.6, "p_mutation": 0.8, "mode_init": "random", "mode_parent_selection": "tournois", "mode_mutation": "swap", "mode_update": "enfants", "mode_arret": "iteration", "mode_crossover": "2_points", "mode_sorti": "None"},
                {  "population_number": 200, "nb_stag_max" : 40, "it_number" : 120, "p_crossover": 0.6, "p_mutation": 0.8, "mode_init": "random", "mode_parent_selection": "tournois", "mode_mutation": "swap", "mode_update": "enfants", "mode_arret": "iteration", "mode_crossover": "2_points", "mode_sorti": "recuit_simule"},
                {  "population_number": 200, "nb_stag_max" : 40, "it_number" : 120, "p_crossover": 0.6, "p_mutation": 0.8, "mode_init": "random", "mode_parent_selection": "tournois", "mode_mutation": "swap", "mode_update": "enfants", "mode_arret": "iteration", "mode_crossover": "2_points", "mode_sorti": "local_search"},
                {  "population_number": 200, "nb_stag_max" : 40, "it_number" : 120, "p_crossover": 0.6, "p_mutation": 0.8, "mode_init": "random", "mode_parent_selection": "tournois", "mode_mutation": "swap", "mode_update": "enfants", "mode_arret": "iteration", "mode_crossover": "1_point", "mode_sorti": "None"},
                {  "population_number": 200, "nb_stag_max" : 40, "it_number" : 120, "p_crossover": 0.6, "p_mutation": 0.8, "mode_init": "random", "mode_parent_selection": "tournois", "mode_mutation": "swap", "mode_update": "enfants", "mode_arret": "iteration", "mode_crossover": "1_point", "mode_sorti": "recuit_simule"},
                {  "population_number": 200, "nb_stag_max" : 40, "it_number" : 120, "p_crossover": 0.6, "p_mutation": 0.8, "mode_init": "random", "mode_parent_selection": "tournois", "mode_mutation": "swap", "mode_update": "enfants", "mode_arret": "iteration", "mode_crossover": "1_point", "mode_sorti": "local_search"},
                {  "population_number": 200, "nb_stag_max" : 40, "it_number" : 120, "p_crossover": 0.6, "p_mutation": 0.8, "mode_init": "random", "mode_parent_selection": "tournois", "mode_mutation": "interchange", "mode_update": "enfants", "mode_arret": "iteration", "mode_crossover": "2_points", "mode_sorti": "None"},
                {  "population_number": 200, "nb_stag_max" : 40, "it_number" : 120, "p_crossover": 0.6, "p_mutation": 0.8, "mode_init": "random", "mode_parent_selection": "tournois", "mode_mutation": "interchange", "mode_update": "enfants", "mode_arret": "iteration", "mode_crossover": "2_points", "mode_sorti": "recuit_simule"},
                {  "population_number": 200, "nb_stag_max" : 40, "it_number" : 120, "p_crossover": 0.6, "p_mutation": 0.8, "mode_init": "random", "mode_parent_selection": "tournois", "mode_mutation": "interchange", "mode_update": "enfants", "mode_arret": "iteration", "mode_crossover": "2_points", "mode_sorti": "local_search"},
                {  "population_number": 200, "nb_stag_max" : 40, "it_number" : 120, "p_crossover": 0.6, "p_mutation": 0.8, "mode_init": "random", "mode_parent_selection": "tournois", "mode_mutation": "interchange", "mode_update": "enfants", "mode_arret": "iteration", "mode_crossover": "1_point", "mode_sorti": "None"},
                {  "population_number": 200, "nb_stag_max" : 40, "it_number" : 120, "p_crossover": 0.6, "p_mutation": 0.8, "mode_init": "random", "mode_parent_selection": "tournois", "mode_mutation": "interchange", "mode_update": "enfants", "mode_arret": "iteration", "mode_crossover": "1_point", "mode_sorti": "recuit_simule"},
                {  "population_number": 200, "nb_stag_max" : 40, "it_number" : 120, "p_crossover": 0.6, "p_mutation": 0.8, "mode_init": "random", "mode_parent_selection": "tournois", "mode_mutation": "interchange", "mode_update": "enfants", "mode_arret": "iteration", "mode_crossover": "1_point", "mode_sorti": "local_search"},
                {  "population_number": 200, "nb_stag_max" : 40, "it_number" : 120, "p_crossover": 0.6, "p_mutation": 0.8, "mode_init": "random", "mode_parent_selection": "tournois", "mode_mutation": "local_search", "mode_update": "enfants", "mode_arret": "iteration", "mode_crossover": "2_points", "mode_sorti": "None"},
                {  "population_number": 200, "nb_stag_max" : 40, "it_number" : 120, "p_crossover": 0.6, "p_mutation": 0.8, "mode_init": "random", "mode_parent_selection": "tournois", "mode_mutation": "local_search", "mode_update": "enfants", "mode_arret": "iteration", "mode_crossover": "2_points", "mode_sorti": "recuit_simule"},
                {  "population_number": 200, "nb_stag_max" : 40, "it_number" : 120, "p_crossover": 0.6, "p_mutation": 0.8, "mode_init": "random", "mode_parent_selection": "tournois", "mode_mutation": "local_search", "mode_update": "enfants", "mode_arret": "iteration", "mode_crossover": "2_points", "mode_sorti": "local_search"},
                {  "population_number": 200, "nb_stag_max" : 40, "it_number" : 120, "p_crossover": 0.6, "p_mutation": 0.8, "mode_init": "random", "mode_parent_selection": "tournois", "mode_mutation": "local_search", "mode_update": "enfants", "mode_arret": "iteration", "mode_crossover": "1_point", "mode_sorti": "None"},
                {  "population_number": 200, "nb_stag_max" : 40, "it_number" : 120, "p_crossover": 0.6, "p_mutation": 0.8, "mode_init": "random", "mode_parent_selection": "tournois", "mode_mutation": "local_search", "mode_update": "enfants", "mode_arret": "iteration", "mode_crossover": "1_point", "mode_sorti": "recuit_simule"},
                {  "population_number": 200, "nb_stag_max" : 40, "it_number" : 120, "p_crossover": 0.6, "p_mutation": 0.8, "mode_init": "random", "mode_parent_selection": "tournois", "mode_mutation": "local_search", "mode_update": "enfants", "mode_arret": "iteration", "mode_crossover": "1_point", "mode_sorti": "local_search"},
                {  "population_number": 200, "nb_stag_max" : 40, "it_number" : 120, "p_crossover": 0.6, "p_mutation": 0.8, "mode_init": "random", "mode_parent_selection": "tournois", "mode_mutation": "recuit_simule", "mode_update": "enfants", "mode_arret": "iteration", "mode_crossover": "2_points", "mode_sorti": "None"},
                {  "population_number": 200, "nb_stag_max" : 40, "it_number" : 120, "p_crossover": 0.6, "p_mutation": 0.8, "mode_init": "random", "mode_parent_selection": "tournois", "mode_mutation": "recuit_simule", "mode_update": "enfants", "mode_arret": "iteration", "mode_crossover": "2_points", "mode_sorti": "recuit_simule"},
                {  "population_number": 200, "nb_stag_max" : 40, "it_number" : 120, "p_crossover": 0.6, "p_mutation": 0.8, "mode_init": "random", "mode_parent_selection": "tournois", "mode_mutation": "recuit_simule", "mode_update": "enfants", "mode_arret": "iteration", "mode_crossover": "2_points", "mode_sorti": "local_search"},
                {  "population_number": 200, "nb_stag_max" : 40, "it_number" : 120, "p_crossover": 0.6, "p_mutation": 0.8, "mode_init": "random", "mode_parent_selection": "tournois", "mode_mutation": "recuit_simule", "mode_update": "enfants", "mode_arret": "iteration", "mode_crossover": "1_point", "mode_sorti": "None"},
                {  "population_number": 200, "nb_stag_max" : 40, "it_number" : 120, "p_crossover": 0.6, "p_mutation": 0.8, "mode_init": "random", "mode_parent_selection": "tournois", "mode_mutation": "recuit_simule", "mode_update": "enfants", "mode_arret": "iteration", "mode_crossover": "1_point", "mode_sorti": "recuit_simule"},
                {  "population_number": 200, "nb_stag_max" : 40, "it_number" : 120, "p_crossover": 0.6, "p_mutation": 0.8, "mode_init": "random", "mode_parent_selection": "tournois", "mode_mutation": "recuit_simule", "mode_update": "enfants", "mode_arret": "iteration", "mode_crossover": "1_point", "mode_sorti": "local_search"},

            ]
            #ILS:
            #Initialization 
            sequence = [random.randint(0, len(heuristics)-1) for i in range(nbHeurstic)]
            #lancement of the hyperheuristic
            best_heurestic, best_param, best_cmax, best_seqFSP=self.ILS(sequence=sequence,nbHeurstic=nbHeurstic, heuristics=heuristics, parameters=parameters, neibourhoodType=neibourhoodType, selectionStrategy = selectionStrategy , stopCriteria = stopCriteria, maxCriteria = maxCriteria, nb_iter_LS = nb_iter_LS)
            #Serialization
            data = {
                'instance': flowshop.data,
                'N': flowshop.N,
                'M':flowshop.M,
                'best_heurestic':best_heurestic,
                'best_heurestic_name':[elem.__name__ for elem in best_heurestic],
                'best_param':best_param,
                'best_cmax':best_cmax,
                'best_seqFSP':best_seqFSP
            }
            filename='data.pickle'
            self.Serialization(data,filename)            
            
    
#######################################################################################################################    
############################################        for the interne memory    #########################################
    #serialisation d'une instance et d'une solution
    def Serialization(self,data,filename):
        with open(filename, 'a+b') as f:
            # Pickle the 'data' list using the highest protocol available.
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    #deserialisation d'une list de solution/instance       
    def Desserialization(self,data,filename):
        data=[]
        with open(filename, 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            while True :
                try:
                    data.append(pickle.load(f))
                except EOFError:
                    break
        f.close()    
        return data                           
                            
                            
                            
                            
                            
    
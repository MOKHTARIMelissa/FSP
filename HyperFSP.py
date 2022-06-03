import random

class HyperFSP:
    
    def __init__(self, data = None):

        self.data = data
        self.N = data.shape[1]
        self.M = data.shape[0]
    
    
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
    def ILS(self, nbHeurstic, heuristics, parameters, neibourhoodType='insertion', selectionStrategy = 'best', stopCriteria = 'iteration', maxCriteria = 100, nb_iter_LS = 1):
        
        #Initialization 
        sequence = [random.randint(0, len(heuristics)-1) for i in range(nbHeurstic)]
        
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
            
            
        best_heurestic = [heurestic[sequence[i]] for i in range(len(sequence))]
        best_param = [parameters[sequence[i]] for i in range(len(sequence))]
        
        return best_heurestic, best_param, best_cmax, best_seqFSP
                            
                            
    def cmaxSeq(self, sequence, heuristics, parameters):
            
        for i in range(len(sequence)):
            if (i == 0):
                seqFSP, cmax = heuristics[sequence[i]](**parameters[sequence[i]])
            else:
                parameters[sequence[i]]["seq"] = seqFSP
                seqFSP, cmax = heuristics[sequence[i]](**parameters[sequence[i]])
                            
        return seqFSP, cmax
                            
                            
                            
                            
                            
                            
                            
    
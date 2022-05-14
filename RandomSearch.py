# Class RandomSearch pour le parameter tunning
import random
import math


class RandomSearch:

    def __init__(self):
        pass
        
    def run(self, fonction, param, nb_ite, nb_exec_iter = 1):
        best_cmax = math.inf
        best_param = {}
        best_sequence = []
        
        for i in range(1, nb_ite): 
            dict_result = {}
            for elem in param:
                dict_result[elem] = random.choice(param[elem])

            cmax_min = math.inf
            sequence_min  = []
            for i in range(1, nb_exec_iter + 1):   
                sequence, cmax = fonction(**dict_result)
                if (cmax < cmax_min):
                    cmax_min = cmax
                    sequence_min = sequence.copy()
                
            if ( cmax_min < best_cmax ) :
                best_cmax = cmax_min
                best_param = dict_result.copy()
                best_sequence = sequence_min.copy()

        return best_param, best_cmax, best_sequence
                

  
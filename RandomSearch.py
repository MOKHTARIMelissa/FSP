# Class RandomSearch pour le parameter tunning
import random
import math
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
    
    
    def generer_dataset(self, fonction, param, nb_ite):

        data = []
        
        for i in range(1, nb_ite): 
            list_result = []
            for elem in param:
                list_result.append(random.choice(param[elem]))

            sequence, cmax = fonction(*list_result)
            data.append(list_result + [int(cmax)])
        
        return pd.DataFrame(data)
    
    
    
    def generer_Xtrain_Ytrain(self, data):
        
        X_train = data.iloc[:, :-1].values  
        Y_train = data.iloc[:, -1].values
            
        label_encoder = LabelEncoder()
        
        types = data.dtypes
        
        for i in range(data.shape[1]-1):
            if (types[i] == "bool" or types[i] =="object"):
                X_train[:,i] = label_encoder.fit_transform(X_train[:,i]) 
        
        return X_train, Y_train
        
  
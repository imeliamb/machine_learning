"""
   - Take a R(q) curve as input
   - Predict number of layers
   - Given predicted number of layers, predict best parameters
   - Run standard fit to optimize
"""
import json
import pickle
from tensorflow.keras.models import model_from_json
import numpy as np
import os
import refl1d
from refl1d.names import *

class StructurePredictor:

    def __init__(self, settings_file, config_number):
        with open(settings_file) as fd:
            settings = json.load(fd)

        # Load knn model
        self.classifier = self.load_knn(settings["classifier"])

        # Load parameter predictors
        self.parameter_predictors = {}
        self.parameter_ranges={}
        self.load_parameter_predictors(settings["parameter_predictors"])
        
      

    def load_knn(self, model_path):
        with open(model_path, 'rb') as f:
            knn_model = pickle.load(f)
        return knn_model

    def load_cnn(self, model_path):
        json_file=open(model_path+".json", "r")
        loaded_model_json=json_file.read()
        json_file.close()

        model = model_from_json(loaded_model_json)
        model.load_weights(model_path+".h5")
        return model

    def load_parameter_predictors(self, model_paths):
        for model_file in model_paths:
            self.parameter_predictors[model_file[0]] = self.load_cnn(model_file[1])
            with open(model_file[2]) as fd:
                self.parameter_ranges[model_file[0]] = json.load(fd)

    def predict(self, data):
        number_of_layers=[]
        for i in enumerate(data):
            number_of_layers.append(self.classifier.predict(data[i[0]]))
        
        number_of_layers=np.asarray(number_of_layers)
        parameters=[]
        for i in range(len(data[0])):
            predictor = self.parameter_predictors[number_of_layers[0][i]]
            pars = predictor.predict(np.asarray([data[0][i]]))
            parameters.append(pars[0])
        #for i,d in enumerate(data):
         #   print("here",number_of_layers[i][1])
          #  predictor = self.parameter_predictors[number_of_layers[i]]
           # print(d.shape)
            #pars = predictor.predict(np.asarray([d[i]]))
            #pars=np.asarray(pars)
            #parameters.append(pars)
            #print("Data %s, Number of layers= %s, Pars= %s" %(i, number_of_layers[i], pars))

        # Perform parameter optimization
        parameters=np.asarray(parameters, dtype=object)
        parameters = self.optimize_parameters(parameters, data)
        final_parameters=[]
        for i in range(len(data[0])):
            pars = self.rescale_parameters(parameters[i], number_of_layers[0][i])
            final_parameters.append(pars)

            
       

        # Perhaps save some useful info here
        return number_of_layers, final_parameters
    
    def big_predict (self,data):
        number_of_layers=[]
        for i in enumerate(data):
            number_of_layers.append(self.classifier.predict(data[i[0]]))
        
        number_of_layers=np.asarray(number_of_layers)
        parameters=[]

        for idx in range(len(data[0])):
            one_data_pars=[]
            for i in range (1,5):
                predictor = self.parameter_predictors[i]
                pars = predictor.predict(np.asarray([data[0][idx]]))
                one_data_pars.append(pars[0])
            parameters.append(one_data_pars)
        # Perform parameter optimization
        
        parameters=np.asarray(parameters, dtype=object)
        parameters = self.optimize_parameters(parameters, data)
        final_parameters=[]
        for k in range(len(parameters)):
            set_of_final=[]
            for i in range (4):
                set_of_final.append (self.rescale_parameters(parameters[k][i], i+1))
            final_parameters.append(set_of_final)
        # Perhaps save some useful info here
        return number_of_layers, final_parameters


    def optimize_parameters(self, structure, data):
        """
            Mat will write this
        """
        return structure
    
    def rescale_parameters(self, parameters, number_of_layers):
        new_parameters=[]
        for i in range(len (parameters)):
            in_min=-1
            in_max=1
            out_min,out_max=self.load_ranges(number_of_layers)
            x=parameters[i]
            new_parameters.append((x - in_min) * (out_max[i] - out_min[i]) / (in_max - in_min) + out_min[i])
        return new_parameters
            
    def load_ranges(self, number_of_layers):
        scaler=self.parameter_ranges[number_of_layers]
        real_mins=[]
        real_maxes=[]
        for i in range (len( scaler["models"][0]["parameters"])):
            real_mins.append(scaler["models"][0]["parameters"][i]["bounds"][0])
            real_maxes.append(scaler["models"][0]["parameters"][i]["bounds"][1])
        return real_mins,real_maxes
    
    def rescale_real_pars(self, parameters, number_of_layers):
        final_parameters=[]
        for i in range(len (parameters)):
            new_parameters=[]
            for p in range(len (parameters[i])):
                in_min=-1
                in_max=1
                out_min,out_max=self.load_ranges(number_of_layers)
                x=parameters[i][p]
                new_parameters.append((x - in_min) * (out_max[p] - out_min[p]) / (in_max - in_min) + out_min[p])
            final_parameters.append(new_parameters)
        return final_parameters
    
    

def calculate_reflectivity(q, parameters, q_resolution=0.025):
    """
        Reflectivity calculation using refl1d
    """
    #zeros = np.zeros(len(q))
    dq = q_resolution * q / 2.355

    # The QProbe object represents the beam
    probe = QProbe(q, dq, data=(None, None))

    n_layers = int((len(parameters)-1)/3)
    
    sample = Slab(material=SLD('Si', rho=2.07), interface=parameters[0])
    
    for i in range(n_layers):
        sample = sample | Slab(material=SLD(name='l%s' % i, rho=parameters[3*i+1]),
                               thickness=parameters[3*i+2], interface=parameters[3*i+3])

    sample = sample | Slab(material=SLD('Air', rho=0))
    
    expt = Experiment(probe=probe, sample=sample)

    q, r = expt.reflectivity()
    z, sld, _ = expt.smooth_profile()
    return q, r, z, sld
        
    


if __name__ == "__main__":
    predictor = StructurePredictor("settings.json")

    refl = load_reflectivity_data()

    prediction = predictor.predict(refl)
    print(prediction)

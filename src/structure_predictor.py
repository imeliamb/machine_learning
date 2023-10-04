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
from bumps.fitters import fit
import matplotlib
import matplotlib.pyplot as plt

class StructurePredictor:

    def __init__(self, settings_file):
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
            number_of_layers.append(self.classifier.predict(data[i[0]], verbose=0))
        
        number_of_layers=np.asarray(number_of_layers)
        parameters=[]
        for i in range(len(data[0])):
            predictor = self.parameter_predictors[number_of_layers[0][i]]
            pars = predictor.predict(np.asarray([data[0][i]]), verbose=0)
            parameters.append(pars[0])
        
        # Perform parameter optimization
        parameters=np.asarray(parameters, dtype=object)
        final_parameters=[]
        for i in range(len(data[0])):
            pars = self.rescale_parameters(parameters[i], number_of_layers[0][i])
            final_parameters.append(pars)

            
       

        # Perhaps save some useful info here
        return number_of_layers, final_parameters
    
    def big_predict (self, data, q_values, optimization, graphing):
        number_of_layers=[]
        for i in enumerate(data):
            number_of_layers.append(self.classifier.predict(data[i[0]]))
        number_of_layers=np.asarray(number_of_layers)
        parameters=[]

        for idx in range(len(data[0])):
            one_data_pars=[]
            for i in range (1,5):
                predictor = self.parameter_predictors[i]
                pars = predictor.predict(np.asarray([data[0][idx]]), verbose=0)
                one_data_pars.append(pars[0])
            parameters.append(one_data_pars)
        parameters=np.asarray(parameters, dtype=object)
        almost_parameters=[]
        for k in range(len(parameters)):
            set_of_almost=[]
            for i in range (4):
                set_of_almost.append (self.rescale_parameters(parameters[k][i], i+1))
            almost_parameters.append(set_of_almost)
            
        #Optional optimization
        if optimization==False and graphing==False:
            return number_of_layers, almost_parameters, parameters
        
        if optimization==True and graphing==False:
            final_parameters=self.big_predict_optimize(data, q_values, almost_parameters)
            return number_of_layers, final_parameters, parameters
        
        if optimization==False and graphing==True:
            final_parameters=almost_parameters
            self.big_predict_graph(final_parameters, q_values, number_of_layers, data)
            return number_of_layers, almost_parameters, parameters
        
        if optimization==True and graphing==True:
            final_parameters=self.big_predict_optimize(data, q_values, almost_parameters)
            self.big_predict_graph(final_parameters, q_values, number_of_layers, data)
            return number_of_layers, final_parameters, parameters
        
        
    def big_predict_optimize(self, data, q_values, almost_parameters):
        refl_corrected=self.correct_refl(data, q_values)
        final_parameters=[]
        for i in range( len(data[0])):
            set_of_final=[]
            for k in range (4):
                final_pars, error=fit_data(q_values, refl_corrected[i], almost_parameters[i][k])
                set_of_final.append(final_pars)
            final_parameters.append(set_of_final)
        return final_parameters
        
        
    def big_predict_graph(self, final_parameters, q_values, number_of_layers, data):
        data=self.correct_refl(data, q_values)
        for i in range(len(final_parameters)):
            fig, ax = plt.subplots(2, 1, dpi=150, figsize=(5, 4.1))
            plt.subplot(2,1,1)
            plt.plot(q_values, data[i], label= 'True')
            for j in range(4):
                #label = '*' if j == number_of_layers[0][i]-1 else ' '
                #par_str = ' '.join(['%-6.3f ' % p for p in final_parameters[i][j]])
                q, r, z, sld = calculate_reflectivity(q_values, final_parameters[i][j])
                plt.subplot(2,1,1)
                plt.plot(q_values, r*10**(j+1), label= str(j+1))
                plt.subplot(2,1,2)
                plt.plot(z, sld, label=str(j+1))
            plt.subplot(2,1,1)
            plt.xlabel('Q ($1/\AA$)', fontsize=10)
            plt.ylabel('Reflectivity', fontsize=10)
            plt.yscale('log')
            plt.xscale('log')
            plt.legend()
            plt.subplot(2,1,2)
            plt.legend()
    def correct_refl(self, data, q_values):
        refl_corrected=[]
        for i in range( len(data[0])):
            refl_corrected_ = np.power(10, data[0][i])/q_values**2*q_values[0]**2
            refl_corrected.append(refl_corrected_)
        return refl_corrected



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
    

def fit_data(q, data, parameters, errors=None, q_resolution=0.025):
    if errors is None:
        errors = 0.07 * data

    expt = create_fit_experiment(q, parameters, data, errors, q_resolution=0.02)
    problem = FitProblem(expt)
    results = fit(problem, method='dream', samples=2000, burn=2000, pop=20, verbose=False)
    
    # The results are in a different order: rough, SLD, thickness
    fit_pars = [results.x[0]]
    fit_errs = [results.dx[0]]

    n_layers = int((len(parameters)-1)/3)
    for i in range(n_layers):
        fit_pars.extend([results.x[3*i+2], results.x[3*i+3], results.x[3*i+1]]) 
        fit_errs.extend([results.dx[3*i+2], results.dx[3*i+3], results.dx[3*i+1]])
    return fit_pars, fit_errs


def create_fit_experiment(q, parameters, data=None, errors=None, q_resolution=0.025):
    #zeros = np.zeros(len(q))
    dq = q_resolution * q / 2.355

    # The QProbe object represents the beam
    probe = QProbe(q, dq, data=(data, errors))
    probe.background = Parameter(value=0.000001,name='background')
    
    n_layers = int((len(parameters)-1)/3)
    
    sample = Slab(material=SLD('Si', rho=2.07), interface=parameters[0])
    
    for i in range(n_layers):
        sample = sample | Slab(material=SLD(name='l%s' % i, rho=parameters[3*i+1]),
                               thickness=parameters[3*i+2], interface=parameters[3*i+3])
        sample['l%s' % i].thickness.range(20.0, 950.0)
        sample['l%s' % i].material.rho.range(-1.0, 7.0)
        sample['l%s' % i].interface.range(20.0, 60.0)

    sample['Si'].interface.range(4.0, 30.0)
    sample = sample | Slab(material=SLD('Air', rho=0))
    
    return Experiment(probe=probe, sample=sample)


def calculate_reflectivity(q, parameters, q_resolution=0.02):
    """
        Reflectivity calculation using refl1d
    """

    expt = create_fit_experiment(q, parameters, q_resolution=0.02)
    q, r = expt.reflectivity()
    z, sld, _ = expt.smooth_profile()
    return q, r, z, sld  


if __name__ == "__main__":
    predictor = StructurePredictor("settings.json")

    refl = load_reflectivity_data()

    prediction = predictor.predict(refl)
    print(prediction)

# Using the Structure Predictor

## Importing the packages

In order to use the Structure Predictor, the sys, os, and structure_predictor pacakges need to be imported. In order to import the structure_predictor pacakge, sys and os need to be used, as shown in the code below.
 
   import sys
   import os
   sys.path.append(os.path.expanduser("~imeliamb/git/machine_learning/src"))
   import structure_predictor as sp
 
## Loading the Predictor

To load the predictor a variable has to be made to call the functions in the structure predictor class. There is only one neccessary input for the initilization of the class, the settings file. This file is a json file containing information such as the paths to the neural networks and fit parameters. Below is a line of code loading the predictor using the settings file we created.

   predictor = sp.StructurePredictor(os.path.expanduser("~imeliamb/git/machine_learning/src/settings.json"))
 
## Using the predictor

The predictor will run a given refl curve through a kNN model which will predict the number of layers in a thin film. It will then run the curve through 4 CNNs which will all give different predictions for the parameters of the film. Each CNN is specified to a certain number of layers in a film, either 1, 2, 3, or 4. To use the predictor four inputs are needed; the refl curves, the q values, and true or false twice. The refl curves can be entered as either an array of curves or a single one, an array of data has to be inputed within an array (in a set of brackets) and a single curve has to be inputed within two arrays (in two sets of brackets). The true and false parameters correlate to optimization and graphing. If the third parameter is True then optimization is togled on, if the fourth parameter is True then graphing of the predicted refl and SLD curves is togled on. The function will ouput 3 things, the preditcted number of layers, the predicted parameters for each layer, and the unscaled version of these layers. 

   predicted_layers, predicted_pars, unscaled_preds = predictor.big_predict([refl], q_values, True, True)
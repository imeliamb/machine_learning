import sys
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import model_from_json
import pickle
from pathlib import Path
sys.path.append(os.path.expanduser("~/git/machine_learning/src"))
import structure_predictor as sp
predictor = sp.StructurePredictor(os.path.expanduser("~imeliamb/git/machine_learning/src/settings.json"))


def gaussian_with_constant(x, a, center, width, background, noise=None):


    y = a * np.exp(-(center-x)**2/width**2) + background
    if noise is not None:
        y = np.random.normal(y, noise*np.sqrt(np.fabs(y)))
    return y


def cnn_encoder(n_data, n_features, n_outputs ):
    

    padding = 'same'

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=40, kernel_size=5, strides=1,
                            activation='relu', padding=padding, input_shape=(n_features, 1)),
        tf.keras.layers.MaxPool1D(pool_size=2),
        tf.keras.layers.Conv1D(filters=40, kernel_size=5, strides=1,
                            activation='relu', padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='selu', kernel_initializer='lecun_normal'),
        tf.keras.layers.Dense(n_outputs)
    ])
    
    return model

def cnn_encoder_onelayer(n_data, n_features, n_outputs ):
    

    padding = 'same'

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=40, kernel_size=5, strides=1,
                            activation='relu', padding=padding, input_shape=(n_features, 1)),
        tf.keras.layers.MaxPool1D(pool_size=2),
        tf.keras.layers.Conv1D(filters=40, kernel_size=5, strides=1,
                            activation='relu', padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='selu', kernel_initializer='lecun_normal'),
        tf.keras.layers.Dense(n_outputs)
    ])
    
    return model

def cnn_encoder_twolayer(n_data, n_features, n_outputs ):
    

    padding = 'same'

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=40, kernel_size=5, strides=1,
                            activation='relu', padding=padding, input_shape=(n_features, 1)),
        tf.keras.layers.MaxPool1D(pool_size=2),
        tf.keras.layers.Conv1D(filters=40, kernel_size=5, strides=1,
                            activation='relu', padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='selu', kernel_initializer='lecun_normal'),
        tf.keras.layers.Dense(n_outputs)
    ])
    
    return model

def cnn_encoder_threelayer(n_data, n_features, n_outputs ):
    

    padding = 'same'

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=40, kernel_size=5, strides=1,
                            activation='relu', padding=padding, input_shape=(n_features, 1)),
        tf.keras.layers.MaxPool1D(pool_size=2),
        tf.keras.layers.Conv1D(filters=40, kernel_size=5, strides=1,
                            activation='relu', padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='selu', kernel_initializer='lecun_normal'),
        tf.keras.layers.Dense(n_outputs)
    ])
    
    return model

def cnn_encoder_fourlayer(n_data, n_features, n_outputs ):
    

    padding = 'same'

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=40, kernel_size=5, strides=1,
                            activation='relu', padding=padding, input_shape=(n_features, 1)),
        tf.keras.layers.MaxPool1D(pool_size=2),
        tf.keras.layers.Conv1D(filters=40, kernel_size=5, strides=1,
                            activation='relu', padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='selu', kernel_initializer='lecun_normal'),
        tf.keras.layers.Dense(n_outputs)
    ])
    
    return model
    
def cnn_decoder(n_data, n_features, n_outputs):

    
    padding = 'same'

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=40, kernel_size=5, strides=1,
                            activation='relu', padding=padding, input_shape=(n_features, 1)),
        tf.keras.layers.MaxPool1D(pool_size=2),
        tf.keras.layers.Conv1D(filters=40, kernel_size=5, strides=1,
                            activation='relu', padding='same'),
        tf.keras.layers.MaxPool1D(pool_size=2),
        tf.keras.layers.Conv1D(filters=50, kernel_size=10, strides=1,
                               activation='relu', padding=padding),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(n_outputs)
    ])

   
    return model

def auto_encoder(n_data, n_features, n_outputs):
    encoder = cnn_encoder(n_data, n_features, n_outputs)
    decoder = cnn_decoder(n_data, n_outputs, n_features)
    
    ae_model = tf.keras.models.Sequential([encoder, decoder]) 

    ae_model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam())
    print("Auto-encoder ready")

    
    return ae_model, encoder, decoder

def save_ae(ae_model, encoder, decoder, name, output_dir):
    encoder_json=encoder.to_json()
    with open(os.path.join(output_dir, "%s-encoder.json" % name), "w") as json_file:
              json_file.write(encoder_json)
    encoder.save_weights(os.path.join(output_dir, "%s-encoder.h5" % name))
              
    decoder_json=decoder.to_json()
    with open(os.path.join(output_dir, "%s-decoder.json" % name), "w") as json_file:
              json_file.write(decoder_json)
    decoder.save_weights(os.path.join(output_dir, "%s-decoder.h5" % name))
              
    ae_model_json=ae_model.to_json()
    with open(os.path.join(output_dir, "%s-ae_model.json" % name), "w") as json_file:
              json_file.write(ae_model_json)
    encoder.save_weights(os.path.join(output_dir, "%s-ae_model.h5" % name))
    



def load_ae(name, output_dir):
    json_file=open(os.path.join(output_dir, "%s-encoder.json" % name), "r")
    loaded_model_json=json_file.read()
    json_file.close()
    
    encoder= model_from_json(loaded_model_json)
    encoder.load_weights(os.path.join(output_dir, "%s-encoder.h5" % name))
    
    
    json_file=open(os.path.join(output_dir, "%s-decoder.json" % name), "r")
    loaded_model_json=json_file.read()
    json_file.close()
    
    decoder=model_from_json(loaded_model_json)
    decoder.load_weights(os.path.join(output_dir, "%s-decoder.h5" % name))
    
    json_file=open(os.path.join(output_dir, "%s-ae_model.json" % name), "r")
    loaded_model_json=json_file.read()
    json_file.close()
    
    ae_model=model_from_json(loaded_model_json)
    ae_model.load_weights(os.path.join(output_dir, "%s-ae_model.h5" % name))
    
def save_model( model, name, output_dir):
    model_json=model.to_json()
    with open(os.path.join(output_dir, "%s-model.json" % name), "w") as json_file:
              json_file.write(model_json)
    model.save_weights(os.path.join(output_dir, "%s-model.h5" % name))
    
def save_knn( model, name, output_dir):
    output_dir= output_dir+ "/%s" % name
    filepath = output_dir
    pickle.dump(model, open(filepath, 'wb'))
    
def load_knn( name, output_dir):
    with open('filename.pkl', 'rb') as f:
        clf = pickle.load(f)
    
def load_model(name, output_dir):
    json_file=open(os.path.join(output_dir, "%s-model.json" % name), "r")
    loaded_model_json=json_file.read()
    json_file.close()
    
    model= model_from_json(loaded_model_json)
    model.load_weights(os.path.join(output_dir, "%s-model.h5" % name))
    return model

def accuracy( testpars, pred_class):
    count = 0
    class_counts = []
    for i in range(1,5):
        _counts = []
        _confusion=[]
        f=0
        k=0
        for j in range(1,5):
            _c = np.sum([int(testpars[idx]==i and pred_class[idx]==j) for idx in range(len(testpars))])
            _counts.append(_c)
            f=f+_c
        for idx in range(len(testpars)):
            if (testpars[idx]==pred_class[idx]==i):
                k+=1
        for idx in range(0,4):
            confusion=_counts[idx]/f*100
            _confusion.append(confusion)
        print(_confusion)
    for idx in range(len(pred_class)):
        if testpars[idx]==pred_class[idx]:
            count += 1
    print("Accuracy: %g" % (count/len(pred_class)))
    
    


def chi_layer_preds(predicted_pars,real_parameters, q_values, k, ranges):
    min_chi=999999999999999
    layer=0
    for i in range(4):
        q, r_real, z, sld = sp.calculate_reflectivity(q_values, real_parameters[k], ranges)
        q, r, z, sld = sp.calculate_reflectivity(q_values, predicted_pars[k][i], ranges)
        print(r_real)
        chi2=np.mean((r_real-r)**2/(0.1*r)**2)
        if chi2<min_chi:
            min_chi=chi2
            layer=i+1
    return layer
    
def big_predict_accuracy(predicted_pars, real_parameters, q_values, ranges):
    real_number_of_layers=[]
    layers=[]
    for k in range(len(real_parameters)):
        real_number_of_layers.append((len(real_parameters[k])-1)/3)
    for k in range (len(predicted_pars)):
        layer=chi_layer_preds(predicted_pars,real_parameters, q_values, k,ranges)
        layers.append(layer)
    for i in range (1,5):
        _counts = []
        _confusion=[]
        f=0
        for j in range(1,5):
            _c = np.sum([int(layers[idx]==i and real_number_of_layers[idx]==j) for idx in range(len(predicted_pars))])
            _counts.append(_c)
            f=f+_c
        for idx in range(4):
            confusion=_counts[idx]/f*100
            _confusion.append(confusion)
        print(_confusion)
        
        
    
    

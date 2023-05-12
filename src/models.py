import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import model_from_json

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
        tf.keras.layers.Dense(100, activation='relu'),
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

def save_model(ae_model, encoder, decoder, name, output_dir):
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


def load_model(name, output_dir):
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
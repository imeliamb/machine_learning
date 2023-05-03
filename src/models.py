import tensorflow as tf
import numpy as np
import os

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

def save_model(ae_model, encoder, decoder, output_dir):
    encoder_json=encoder.to_json()
    with open(os.path.join(output_dir, "encoder.json"), "w") as json_file:
              json_file.write(encoder_json)
    encoder.save_weights(os.path.join(output_dir, "encoder.h5"))
              
    decoder_json=decoder.to_json()
    with open(os.path.join(output_dir, "decoder.json"), "w") as json_file:
              json_file.write(decoder_json)
    decoder.save_weights(os.path.join(output_dir, "decoder.h5"))
              
    ae_model_json=ae_model.to_json()
    with open(os.path.join(output_dir, "ae_model.json"), "w") as json_file:
              json_file.write(ae_model_json)
    encoder.save_weights(os.path.join(output_dir, "ae_model.h5"))


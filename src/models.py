def cnn_encoder(fit=True):
    n_data = trainset.shape[0]
    n_features = trainset.shape[1]
    n_outputs = trainsetout.shape[1]

    padding = 'same'

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=40, kernel_size=5, strides=1,
                            activation='relu', padding=padding, input_shape=(n_features, 1)),
        tf.keras.layers.MaxPool1D(pool_size=2),
        tf.keras.layers.Conv1D(filters=40, kernel_size=5, strides=1,
                            activation='relu', padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(trainsetout.shape[1])
    ])

    # If we are just building the model, stop here
    if not fit:
        return model

    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam())
    print("Encoder ready")

    history = model.fit(trainset, trainsetout,
                        epochs=20, batch_size=128,
                        validation_data=(testset, testsetout))
    return model, history

def cnn_decoder(fit=True):
    n_data = trainset.shape[0]
    n_outputs = trainset.shape[1]
    n_features = trainsetout.shape[1]
    
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

    if not fit:
        return model

    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam())
    print("Decoder ready")

    history = model.fit(trainsetout, trainset,
                        epochs=20, batch_size=128,
                        validation_data=(testsetout, testset))
    return model, history

def auto_encoder():
    encoder = cnn_encoder(False)
    decoder = cnn_decoder(False)
    
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


"""
   - Take a R(q) curve as input
   - Predict number of layers
   - Given predicted number of layers, predict best parameters
   - Run standard fit to optimize
"""
import json


class StructurePredictor:

    def __init__(self, settings_file):
        settings = json.loads(settings_file)

        # Load knn model
        self.classifier = self.load_knn(settings["classifier"])

        # Load parameter predictors
        self.parameter_predictors = {}
        self.load_parameter_predictors(settings["parameter_predictors"])

    def load_knn(self, model_path):
        # This is the pickle
        return knn_model

    def load_cnn(self, model_path):
        json_file=open(model_path, "r")
        loaded_model_json=json_file.read()
        json_file.close()

        model = model_from_json(loaded_model_json)
        model.load_weights(os.path.join(output_dir, "%s-model.h5" % name))
        return model

    def load_parameter_predictors(self, model_paths):
        for model_file in model_paths:
            self.parameter_predictors[model_file[0]] = self.load_cnn(model_file[1])

    def predict(self, data):
        number_of_layers = self.classifier.predict(data)

        predictor = self.parameter_predictors[number_of_layers]
        parameters = predictor.predict(data)

        # Perform parameter optimization
        parameters = self.optimize_parameters(parameters, data)

        # Perhaps save some useful info here

        return parameters

    def optimize_parameters(self, structure, data):
        """
            Mat will write this
        """
        return structure


if __name__ == "__main__":
    predictor = StructurePredictor("settings.json")

    refl = load_reflectivity_data()

    prediction = predictor.predict(refl)
    print(prediction)

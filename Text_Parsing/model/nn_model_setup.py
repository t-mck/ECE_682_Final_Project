from Text_Parsing.model import model_hyperparameter_factory as mhf, model_factory as mf, \
    model_prediction_factory as mpf

from Text_Parsing.data_code import data_factory as df

import torch
import random
import numpy as np


class NNSetup:
    def __init__(self):
        self.model_hyperparams = mhf.HyperParameterFactory()
        self.model_factory = mf.AbstractModelFactory()
        self.data_factory = df.AbstractDatasetFactory()
        self.prediction_factory = mpf.ModelPredictionFactory()

    @staticmethod
    def set_random_seed(random_seed: int = 2):
        # -This will affect which observations end up in the training set, validation set, and test set,
        # and how the model trains.
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)

    def get_language_model_setup(self,
                                 random_seed: int = 2):
        self.set_random_seed(random_seed=random_seed)
        self.data_factory = df.LanguageDatasetFactory()
        self.model_factory = mf.LanguageModelFactory()
        return self.model_hyperparams, self.model_factory, self.data_factory, self.prediction_factory

    def get_image_model_setup(self,
                              random_seed: int = 2):
        raise NotImplementedError("Currently image models are not implemented for the DeepNeuralNetwork package")
    
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:43:34 2019

@author: Andrew
"""
#need to set random seeds before imports for consistent results?
from numpy.random import seed
seed(1)
try:
    from tensorflow import set_random_seed
except:
    from tensorflow.compat.v1.random import set_random_seed
set_random_seed(2)
from Constants import Constants
from keras.models import Sequential, Model
from keras.layers import Dense, Activation
from keras import losses, optimizers,regularizers, layers

class Normalizer():

    def __init__(self):
        self.std = 1
        self.mean = 0

    def fit(self, x):
        self.std = x.std(axis = 0)
        self.mean = x.mean(axis = 0)

    def transform(self, x):
        return (x - self.mean)/self.std

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def unnormalize(self, x):
        return x*self.std + self.mean

class Denoiser():

    def __init__(self, noise = 1, dropout = .1, n_features = None, normalize = True, verbose = 0):
        if n_features is None:
            n_features = Constants.num_organs
        self.verbose = verbose
        input_x = layers.Input(shape=(n_features,))
        encoder = Sequential([
                layers.GaussianDropout(dropout),
                Dense(2*n_features, activation = 'linear'),
                layers.GaussianNoise(noise),
                Dense(n_features, activation = 'linear'),
                ])(input_x)
        model = Model(input_x, encoder)
        self.model = model
        if normalize:
            self.normalizer = Normalizer()
        else:
            self.normalizer = None

    def fit(self, features, normalize = True, lr = .001,
                     epochs = 800, batch_size = 4):
        optimizer = optimizers.Adam(lr=lr)
        self.model.compile(loss = losses.mean_squared_error,
                      optimizer = optimizer)
        if self.normalizer is not None:
            x = self.normalizer.fit_transform(features)
        else:
            x = features
        self.model.fit(x,x,
                       epochs = epochs,
                       batch_size = batch_size,
                       verbose = self.verbose)

    def transform(self, features, normalize = True):
        if self.normalizer is not None:
            x = self.normalizer.transform(features)
        else:
            x = features
        y = self.model.predict(x)
        if self.normalizer is not None:
            y = self.normalizer.unnormalize(y)
        return y

    def fit_transform(self, features, normalize = True, lr = .001,
                     epochs = 800, batch_size = 4):
        self.fit(features, normalize = normalize, lr = lr,
                 epochs = epochs, batch_size=batch_size)
        return self.transform(features, normalize= normalize)

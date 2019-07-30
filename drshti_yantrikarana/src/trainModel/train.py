"""
Reference: 
Usage:

About: Script to train a keras model

Author: Satish Jasthi
"""
from typing import Optional

from tensorflow.python import keras

from drshti_yantrikarana.config import (model_optimizer, model_loss,
                                        model_metrics, model_steps_per_epoch,
                                        model_epochs, train_verbose, train_callbacks,
                                        train_validation_steps, train_validation_freq,
                                        train_class_weight, train_max_queue_size, train_workers,
                                        train_use_multiprocessing, train_shuffle,
                                        train_initial_epoch)
from drshti_yantrikarana.src.data.dataGenerators.createGenerator import getImageLablesFromDataset


class TrainModel(object):

    def __init__(self, network=None, network_name=None):
        self.network = network()
        self.network_name = network_name
        if not self.network.data_type=='custom':
            self.standard_data=False
            (self.x_train, self.y_train), (self.x_test, self.y_test) = getattr(keras.datasets, self.network.data_type).load_data()

    def train(self)->(keras.callbacks.History, keras.Model):
        """
        Method to train the model
        :return: keras.Model
        """
        self.model = self.network.getkerasModel(name=self.network_name)
        self.model.compile(optimizer=model_optimizer,
                           loss=model_loss,
                           metrics=model_metrics
                           )
        if not self.network.data_type == 'custom':
            train_gen = getImageLablesFromDataset(mode='train')
            validation_gen = getImageLablesFromDataset(mode='test')
        else:
            train_gen = self.x_train, self.y_train
            validation_gen = self.x_test, self.y_test
        model_history = self.model.fit(train_gen,
                                 steps_per_epoch=model_steps_per_epoch,
                                 epochs=model_epochs,
                                 verbose=train_verbose,
                                 callbacks=train_callbacks,
                                 validation_data=validation_gen,
                                 validation_steps=train_validation_steps,
                                 validation_freq=train_validation_freq,
                                 class_weight=train_class_weight,
                                 max_queue_size=train_max_queue_size,
                                 workers=train_workers,
                                 use_multiprocessing=train_use_multiprocessing,
                                 shuffle=train_shuffle,
                                 initial_epoch=train_initial_epoch)
        return model_history, self.model


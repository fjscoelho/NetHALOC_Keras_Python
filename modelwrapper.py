# -*- coding: utf-8 -*-
###############################################################################
# Name        : ModelWrapper
# Description : Simple wrapper to ease the acces to one specific Keras model.
# Notes       : See example_modelwrapper.py for an unsage example.
# Author      : Antoni Burguera (antoni.burguera@uib.es)
# History     : 27-June-2019 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

# from keras import layers
# from keras import models
from pickle import dump,load
from os.path import splitext
# from keras.api.models import load_model
import matplotlib.pyplot as plt
# from keras.api.models import Model

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

class ModelWrapper:
    # Constructor. Please note that it does NOT create the model
    # Input  : inputShape - The size of input images.
    #          outputSize - Size of the model output (dense part of network)
    def __init__(self,inputShape=(240,320,3),outputSize=128*3): # CUIDADO 128*3 porque es la longitud del HALOC
        self.theModel=None
        self.cnnModel=None
        self.trainHistory=None
        self.inputShape=inputShape
        self.outputSize=outputSize

    # Creates the model (theModel) and a second model (cnnModel) neglecting the
    # dense layers.        
    def create(self):
        # Define the model
        self.theModel = Sequential()

        # Add an input layer
        self.theModel.add(Input(shape=self.inputShape))  # RGB images of size 240x320

        # Add a hidden layers
        self.theModel.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(2,2), activation='sigmoid'))
        self.theModel.add(MaxPool2D(pool_size=(3,3),strides=(2,2)))
        self.theModel.add(Conv2D(filters=64,kernel_size=(3,3), strides=(1,1),activation='sigmoid'))
        self.theModel.add(MaxPool2D(pool_size=(3,3),strides=(2,2)))
        self.theModel.add(Conv2D(filters=4,kernel_size=(3,3),strides=(1,1),activation='sigmoid'))
        self.theModel.add(Flatten())
        self.theModel.add(Dense(512, activation='sigmoid'))
        self.theModel.add(Dense(1024, activation='sigmoid'))
        self.theModel.add(Dense(self.outputSize, activation='sigmoid'))

        # Compile the model
        self.theModel.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])

        # # Summary of the model
        self.theModel.summary()

        self.__define_cnnmodel__()
      
    # Private method to define the aforementioned cnnModel
    def __define_cnnmodel__(self):
        self.cnnModel=Model(inputs=self.theModel.inputs,outputs=self.theModel.layers[5].output) # A saída da CNN é a camada Flatten

    # Just a helper to build filenames
    def __build_filenames__(self,fileName):
        baseName,theExtension=splitext(fileName)
        modelFName=baseName+'.h5'
        histFName=baseName+'_HISTORY.pkl'
        return modelFName,histFName
        
    # Saves the model (as a .h5 file) and the training history (by means of
    # pickle)
    def save(self,fileName):
        modelFName,histFName=self.__build_filenames__(fileName)
        self.theModel.save(modelFName)
        with open(histFName,'wb') as histFile:
            dump(self.trainHistory,histFile)
        
    # Loads the model and the training history
    def load(self,fileName):
        modelFName,histFName=self.__build_filenames__(fileName)
        self.theModel=load_model(modelFName, compile = False)    # inseri a flag compile = False em 02/09     
        with open(histFName,'rb') as histFile:
            self.trainHistory=load(histFile)
        self.inputShape=self.theModel.input_shape 
        self.outputSize=self.theModel.output_shape
        self.__define_cnnmodel__()
        
    # Plots the training history
    def plot_training_history(self,plotTitle='TRAINING EVOLUTION'):
        plt.plot(self.trainHistory['loss'])
        plt.plot(self.trainHistory['val_loss'])
        plt.title(plotTitle)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train','validation'], loc='upper left')
        plt.show()

    # Trains the model. Only useable with data generators. Please use those
    # defined in datagenerators.py. Also note that the outputSize has to
    # coincide with the one provided by the data generators.
    def train(self,trainGenerator,valGenerator,nEpochs=100):
        # Configura o EarlyStopping
        early_stopping = EarlyStopping(
            monitor='val_loss',      # Métrica a ser monitorada (val_loss, val_mae, etc.)
            patience=10,             # Número de épocas sem melhoria após o qual o treinamento será interrompido
            restore_best_weights=True # Restaura os melhores pesos do modelo encontrados durante o treinamento
        )

        # Treina o modelo com o EarlyStopping
        self.trainHistory = self.theModel.fit(
            trainGenerator,
            epochs=nEpochs,
            validation_data=valGenerator,
            callbacks=[early_stopping] # Adiciona o callback
        ).history
        
    # Evaluate the model
    def evaluate(self,testGenerator):
        return self.theModel.evaluate_generator(testGenerator)
        
    # Output the model predictions. Use the parameter useCNN to select the
    # output of the dense layers (useCNN=False) or the output of the convolu-
    # tional layers (useCNN=True).
    def predict(self,theImages,useCNN=True):
        if useCNN:
            return self.cnnModel.predict(theImages)
        return self.theModel.predict(theImages)

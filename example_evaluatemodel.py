# -*- coding: utf-8 -*-
###############################################################################
# Name        : example_modelwrapper
# Description : A basic example on how to use the ModelWrapper class
# Notes       : Just run the script. Be sure that dataset.py is accessible,
#               and the DATASETn.TXT files are accessible, and that the
#               paths specified within DATASETn.TXT are correct.
# Author      : Francisco Bonin Font (francisco.bonin@uib.es)
# History     : 2-July-2019 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

from dataset import DataSet
from datagenerator import DataGeneratorHALOCImages
from modelwrapper import ModelWrapper

# Create an empty model, 384 is the lenght of the hash
print('[[ CREATING THE MODEL ]]')
theModel=ModelWrapper(outputSize=384)

# Load a trained model
print('[[ LOADING THE MODEL ]]')
theModel.load('TRAINED_MODELS/ORIGINAL_MODEL_40EP_10ES/TEST_MODEL_trainDS2_valDS1')
print('[[ MODEL SAVED ]]')

# Plot the training history
print('[[ PLOTTING TRAINING HISTORY ]]')
theModel.plot_training_history()
print('[[ PLOT DONE ]]')

# Load the test dataset
print('[[ LOADING DATASET 3 ]]')
dataSet=DataSet('DATASETS/DATASET3.TXT')
print('[[ DATASET LOADED ]]')

# Creating the tester
from tester import Tester

print('[[ CREATING THE TESTER ]]')
theTester=Tester(theModel,dataSet)
print('[[ TESTER CREATED ]]')

# Computing and plotting hit ratio evolution
print('[[ COMPUTING AND PLOTTING HIT RATIO EVOLUTION ]]')
# theHR,thaAUC=theTester.compute_hitratio_evolution()
# theTester.plot_hitratio_evolution()
# print('[[ HIT RATIO EVOLUTION COMPUTED AND PLOTTED ]]')

# Computing and printing full stats
print('[[ COMPUTING AND PRINTING FULL STATS ]]')
tp,fp,tn,fn,tdist,tloops=theTester.compute_fullstats()
print('[[ FULL STATS COMPUTED AND PRINTED ]]')
# -*- coding: utf-8 -*-
###############################################################################
# Name        : example_dataset
# Description : A basic example on how to use the DataSet class
# Notes       : Just run the script. Be sure that DATASET1.TXT, DATASET2.TXT
#               and DATASET3.TXT are within the DATASETS folder and that
#               the paths specified in these files for database and query
#               images folders are correct.
# Author      : Antoni Burguera (antoni.burguera@uib.es)
# History     : 27-June-2019 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

from dataset import DataSet
import matplotlib.pyplot as plt
import sys

# Dataset 1
dataSetPath = 'DATASETS/DATASET1.TXT'

dataSetName =  dataSetPath.split('/')
dataSetName = dataSetName[-1]
dataSetName = dataSetName.split('.')
dataSetName = dataSetName[0]

# Load dataset
print('[[ LOADING DATASETS ]]')
dataSet=DataSet(dataSetPath)
print('[[DATASETS LOADED ]]\n\n')

# Let's print the dataSet info
print('[[ PRINTINT '+ dataSetName +' INFO ]]')
dataSet.print()
print('[[ '+ dataSetName +' PRINTED ]]\n\n')

print('[[ PLOTTING ALL LOOPS]]')

print('[[ PLOTTING ONE IMAGE WITH 3 LOOPS ]]')

# Plot all loops
loopCount = 0
for curLoop in dataSet.theLoops:
    dbFileName, qFileName = dataSet.get_loop(loopCount)
    loopCount +=1 
    plt.figure(loopCount)
    plt.figtext(0.5, 0.7, 'Loop '+ str(loopCount), ha = "center")
    plt.subplot(1,2,1)
    # Query image
    plt.imshow(qFileName)

    plt.subplot(1,2,2)
    # DataBase image
    plt.imshow(dbFileName)
    plt.figtext(0.5, 0.2, 'Q: '+ dataSet.qImageFns[curLoop[1]] + '      DB: '+dataSet.dbImageFns[curLoop[0]], ha="center")
    # images[loopCount-1] = plt.figure(loopCount)
    plt.show()

# for queryIndex in range(dataSet.numQImages):
#     theLoops=dataSet.get_qloop(queryIndex)
#     numLoops=len(theLoops)
#     if numLoops>=3:
#         break

# # If loop not found, just exit
# if numLoops<3:
#     sys.exit('[[ ERROR: UNABLE TO FOUND 3 LOOP CLOSURES IN DATASET 3]]')
    
# # Otherwise, plot the query and three loops
# plt.figure()
# plt.subplot(2,2,1)
# plt.imshow(dataSet.get_qimage(queryIndex))
    
# for i in range(3):
#     plt.subplot(2,2,i+2)
#     plt.imshow(dataSet.get_dbimage(theLoops[i]))
    
# plt.show()
# print('[[ PLOT DONE ]]')
'''
Created on Sep 25, 2015

@author: Matthias
'''
from urllib import urlopen
import json
import DatabaseActions
from DatabaseActions import championdictionary
import numpy
import scipy
import pybrain
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.supervised.trainers.backprop import BackpropTrainer
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.structure.networks.feedforward import FeedForwardNetwork
from pybrain.structure.modules.linearlayer import LinearLayer
from pybrain.structure.modules.sigmoidlayer import SigmoidLayer
from pybrain.structure.connections.full import FullConnection
from pybrain.structure.modules.biasunit import BiasUnit

# Next we transform the data into a vectorized format so that it can be used as a training set
aramdata = open("ARAMData.txt","r")

#ChampionDictionary holds all the riot static data about each champion. The Riot IDs are the keys of the dictionary
championdictionary = DatabaseActions.CreateChampionDictionary()

#Creates a Neural Network of Appropriate size
predictionNet = FeedForwardNetwork()

inLayer = LinearLayer(len(championdictionary))
hiddenLayer = SigmoidLayer(5)
outLayer = SigmoidLayer(1)

predictionNet.addInputModule(inLayer)
predictionNet.addModule(hiddenLayer)
predictionNet.addOutputModule(outLayer)
predictionNet.addModule(BiasUnit(name = 'bias'))

in_to_hidden = FullConnection(inLayer,hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer,outLayer)

predictionNet.addConnection(in_to_hidden)
predictionNet.addConnection(hidden_to_out)
predictionNet.addConnection(FullConnection(predictionNet['bias'],hiddenLayer))
predictionNet.addConnection(FullConnection(predictionNet['bias'],outLayer))
predictionNet.sortModules()


trainingSet = SupervisedDataSet(len(championdictionary),1)

#Takes each game and turns it into a vector. -1 is stored if the champion is on the opposing team, 1 if the champion is on the player's team
#and 0 if it wasn't played. The vector is then fed into the Neural Network's Training Set
print "Adding Games to NN"
for game in aramdata.readlines():
 
    aramgame = json.loads(game)
    teammates,opponents, gameWin = DatabaseActions.GetResults(aramgame)
    
    #writes a vector of which champions were on which team followed by the result
    gamevector = [0]*len(championdictionary)
    for champion in teammates:
        gamevector[championdictionary[str(champion)]["Id"]-1] = 1
    for champion in opponents:
        gamevector[championdictionary[str(champion)]["Id"]-1] = -1
    
    #Feeds that result into our Neural Network's training set
    trainingSet.appendLinked(gamevector,int(gameWin))
    
#Creates a Backpropagation trainer and proceeds to train on our set. This step can take a while due to the volume of our data.
print "Training NN"
trainer = BackpropTrainer(predictionNet,trainingSet)
trainer.trainUntilConvergence(dataset = trainingSet, maxEpochs = 300, verbose = True, continueEpochs = 10, validationProportion=0.1)
    
#Saves our trained Network by exporting it to an XML file
NetworkWriter.writeToFile(predictionNet, "TrainedNetwork.xml")

aramdata.close()

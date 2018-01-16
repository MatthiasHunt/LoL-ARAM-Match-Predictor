'''
Created on Sep 25, 2015

@author: Matthias
'''

import json
import DatabaseActions
import time
import pybrain
#from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.supervised.trainers.backprop import BackpropTrainer
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.structure.networks.feedforward import FeedForwardNetwork
from pybrain.structure.modules.linearlayer import LinearLayer
from pybrain.structure.modules.sigmoidlayer import SigmoidLayer
from pybrain.structure.connections.full import FullConnection
from pybrain.structure.modules.biasunit import BiasUnit

#The purpose of this program is to build an ARAM Database of arbitrary length and train a neural network to predict outcomes based on champion selection

#Due to Riot API limitations, we must build a database by starting with existing players and using their game histories. 
#The Seed Ids are my Riot account and three others I found that tend to play many ARAMs.
seedIds=[40872575,24198414,39005178,30426242]

#This is the size of database we hope to reach. Currently set at 5,000
DBSize = 5000

#While the database is building we keep track of Player Ids and Game Ids to avoid data duplication.
playerIds, gameIds = seedIds, []

#We use the list of player Ids to both get ARAM games and to get new players to look at.
#We use the variable currentplayerId to keep track of which users have already been called from our list.
currentplayerId = 0

#Opens the txt file that we will be writing our ARAM Game data into
aramdata = open("ARAMData.txt","w")

#Keep finding more data until our database reaches the requested size
while len(gameIds) < DBSize:
    
    #Prints our current progress
    print ("Starting data of player " + str(playerIds[currentplayerId]) + " " +str(100*len(gameIds)/float(DBSize)) + "% complete")    
    
    #GetHistory returns the Riot API data given a player's ID
    playerhistory = DatabaseActions.GetHistory(playerIds[currentplayerId])
    
    #UsableGames removes all games that are repeats, too old, or not ARAMs from a player's history
    playerhistory = DatabaseActions.UsableGames(playerhistory,gameIds)
    
    #Adds new gameIds to our list and writes the game data to the file "ARAMData.txt"
    for game in playerhistory["games"]:
        gameIds.append(game["gameId"])
        aramdata.write(json.dumps(game))
        aramdata.write("\n")
        ##I'm not sure why python had issues combining the last two lines. Tried a couple of fixes and then moved on. whatever.
        
    #Looks at the games we recently added and adds the Ids of the other players from those games to our list of IDs if they weren't already there
    for game in playerhistory["games"]:
        newIds = [ player["summonerId"] for player in game["fellowPlayers"] ]
        playerIds = playerIds + newIds
        
    currentplayerId += 1
    
    #Due to throttling on the part of Riot API, we must pause for 1 second every loop to keep our speed down
    ##The worst
    time.sleep(1)
    
aramdata.close()

# Next we transform the data into a vectorized format so that it can be used as a training set
aramdata = open("ARAMData.txt","r")

#ChampionDictionary holds all the riot static data about each champion. The Riot IDs are the keys of the dictionary
championdictionary = DatabaseActions.CreateChampionDictionary()

#Creates a Neural Network of Appropriate size
predictionNet = FeedForwardNetwork()

inLayer = LinearLayer(len(championdictionary))
hiddenLayer = SigmoidLayer(20)
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
trainer.trainUntilConvergence(dataset = trainingSet, maxEpochs = 20, verbose = True, continueEpochs = 10, validationProportion=0.1)
    
#Saves our trained Network by exporting it to an XML file
NetworkWriter.writeToFile(predictionNet, "TrainedNetwork.xml")

aramdata.close()

'''
Created on Oct 18, 2015

@author: Matthias
'''
#This is a tool to provide benchmarks for the NN Trainer stored in the TrainedNetwork.xml file

import json
import pybrain
import DatabaseActions
from DatabaseActions import championdictionary
from pybrain.tools.customxml.networkreader import NetworkReader

#Predict game takes a NN and the raw data for a game. It returns the strength of its prediction (always over 0.5) followed by
#a boolean of whether or not it successfully predicted the game
def PredictGame(wizard,aramgame):
    aramgame = json.loads(aramgame)
    teammates,opponents, gameWin = DatabaseActions.GetResults(aramgame)
    
    #writes a vector of which champions were on which team followed by the result
    gamevector = [0]*len(championdictionary)
    for champion in teammates:
        gamevector[championdictionary[str(champion)]["Id"]-1] = 1
    for champion in opponents:
        gamevector[championdictionary[str(champion)]["Id"]-1] = -1
    
    prediction = wizard.activate(gamevector)[0]
    
    #Check to see if we were correct
    correct = (True if int(round(prediction)) == gameWin else False)
    
    #For data sorting we will describe all our certainties as affirmative cases (over 50%)
    if prediction < 0.5:
        prediction = 1 - prediction
        
    return prediction,correct

def BiggestTheoreticalBlowoutGame(wizard):
    idealblowout = [0]*127
    bestchamps = [41,109,93,55,59] #Jayce,Varus,Sona,Leona,Lux
    worstchamps = [68,24,107,72,83] #Nasus,Evelynn,Udyr,Nunu,Rumble
    for i in bestchamps:
        idealblowout[i-1] = 1
    for i in worstchamps:
        idealblowout[i-1] = -1
    return wizard.activate(idealblowout)[0]

def DisplayResults(aramgame):
    aramgame = json.loads(aramgame)
    teammates,opponents, gameWin = DatabaseActions.GetResults(aramgame)
    if gameWin == False:
        print "---------------------\nTEAM ONE\n---------------------"
    else:
        print"---------------------\nTEAM ONE -- WINNERS\n---------------------"
    for champion in teammates:
        print championdictionary[str(champion)]["championName"]
    if gameWin == False:
        print "---------------------\nTEAM TWO -- WINNERS\n---------------------"
    else:
        print"---------------------\nTEAM TWO\n---------------------"
    for champion in opponents:
        print championdictionary[str(champion)]["championName"]
    print "\n\n\n"
    
def ErrorTest(wizard,datafile):
    aramgames = open(datafile,"r")
    #Initialize our results list. Currently there are 6 classes of prediction
    predictionsMade,correctPredictions = [0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]
    for game in aramgames.readlines():
        
        predictionStrength,correct = PredictGame(wizard, game)
        
        if predictionStrength<0.55:
            predictionsMade[0] += 1
            if correct == True:
                correctPredictions[0] += 1
        elif predictionStrength<0.60:
            predictionsMade[1] += 1
            if correct == True:
                correctPredictions[1] += 1
        elif predictionStrength<0.65:
            predictionsMade[2] += 1
            if correct == True:
                correctPredictions[2] += 1
        elif predictionStrength<0.70:
            predictionsMade[3] += 1
            if correct == True:
                correctPredictions[3] += 1
        elif predictionStrength<0.75:
            predictionsMade[4] += 1
            if correct == True:
                correctPredictions[4] += 1
        elif predictionStrength<0.80:
            predictionsMade[5] += 1
            if correct == True:
                correctPredictions[5] += 1
        elif predictionStrength<0.85:
            predictionsMade[6] += 1
            if correct == True:
                correctPredictions[6] += 1
        elif predictionStrength<0.90:
            predictionsMade[7] += 1
            if correct == True:
                correctPredictions[7] += 1
        elif predictionStrength<0.95:
            predictionsMade[8] += 1
            if correct == True:
                correctPredictions[8] += 1
        else:
            print predictionStrength
            predictionsMade[9] += 1
            if correct == True:
                print "correctly predicted"
                correctPredictions[9] += 1
            DisplayResults(game)
    print predictionsMade,correctPredictions
    print "Accuracy for 50-55% confidence: " + str(float(correctPredictions[0])/predictionsMade[0]) + " (" + str(predictionsMade[0]) + " games in this category)"
    print "Accuracy for 55-60% confidence: " + str(float(correctPredictions[1])/predictionsMade[1]) + " (" + str(predictionsMade[1]) + " games in this category)"
    print "Accuracy for 60-65% confidence: " + str(float(correctPredictions[2])/predictionsMade[2]) + " (" + str(predictionsMade[2]) + " games in this category)"
    print "Accuracy for 65-70% confidence: " + str(float(correctPredictions[3])/predictionsMade[3]) + " (" + str(predictionsMade[3]) + " games in this category)"
    print "Accuracy for 70-75% confidence: " + str(float(correctPredictions[4])/predictionsMade[4]) + " (" + str(predictionsMade[4]) + " games in this category)"
    print "Accuracy for 75-80% confidence: " + str(float(correctPredictions[5])/predictionsMade[5]) + " (" + str(predictionsMade[5]) + " games in this category)"
    print "Accuracy for 80-85% confidence: " + str(float(correctPredictions[6])/predictionsMade[6]) + " (" + str(predictionsMade[6]) + " games in this category)"
    print "Accuracy for 85-90% confidence: " + str(float(correctPredictions[7])/predictionsMade[7]) + " (" + str(predictionsMade[7]) + " games in this category)"    
    print "Accuracy for 90-95% confidence: " + str(float(correctPredictions[8])/predictionsMade[8]) + " (" + str(predictionsMade[8]) + " games in this category)"
    print "Accuracy for 95-100% confidence: " + str(float(correctPredictions[9])/predictionsMade[9]) + " (" + str(predictionsMade[9]) + " games in this category)"

    
    print "Overall Accuracy: " + str(float(sum(correctPredictions))/sum(predictionsMade))

#The number of layers in the hidden layer must be manually input here. It is the second argument
#wizard = buildNetwork(len(championdictionary),20,1)
wizard = NetworkReader.readFrom("TrainedNetwork.xml")
print wizard
#print BiggestTheoreticalBlowoutGame(wizard)
#We will now apply the wizard to our 10,000 ARAM examples
#ErrorTest(wizard, "ARAMData.txt")
ErrorTest(wizard, "CrossValidationData.txt")










'''
Created on Sep 28, 2015

@author: Matthias
'''

from urllib import urlopen
import json
import time

#Calls to Riot API and returns a JSON dictionary of the Player's recent matches

def GetHistory(playerId):
    return json.loads(urlopen('https://na.api.pvp.net/api/lol/na/v1.3/game/by-summoner/' +
           str(playerId) + '/recent?api_key=13b54550-6ce2-43e0-8ffa-df3f73f92581').read())
    

#Takes a dict of Riot Recent games data and deletes the entries we aren't interested in
def UsableGames(playerhistory, gameIds = []):
    #keeps track of games that meet our requirements
    usablehistory = []
    #This is the epoch time (in ms) of the oldest data we are willing to use. The number is approx. 1 month
    cutoffdate = time.time() - 2592000000
    for game in playerhistory["games"]:
        #Checks to make sure the game is both unique and current and ARAM
        if (game["gameId"] not in gameIds) and (game["createDate"]>cutoffdate) and (game["gameMode"]=="ARAM") and (game["gameType"]=="MATCHED_GAME"):
            usablehistory.append(game)
    #overwrites the games data with only the usable ones and returns the data        
    playerhistory["games"] = usablehistory
    return playerhistory
        
def CreateChampionDictionary():
    championdict = {}
    championinfo = open("StaticChampionData.txt","r")
    for line in championinfo.readlines():
        champname,riotId,alphaId = line.strip().split("\t");
        championdict[riotId] = {"championName":champname, "Id":int(alphaId)};
    championinfo.close()
    return championdict

championdictionary = CreateChampionDictionary()

def GetResults(aramgame):
    #Figures out which champions were on which team
    hostteam = aramgame["teamId"]
    teamchampids = [player["championId"] for player in aramgame["fellowPlayers"] if player["teamId"] == hostteam]
    teamchampids.append(aramgame["championId"])
    opponentchampids = [player["championId"] for player in aramgame["fellowPlayers"] if player["teamId"] != hostteam]
    return teamchampids,opponentchampids,aramgame["stats"]["win"]
    
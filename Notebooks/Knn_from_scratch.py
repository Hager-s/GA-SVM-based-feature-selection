
import pandas as pd
import math
import operator

TrainData= pd.read_csv("Modified_TrainData.csv")
TrainData = TrainData.drop('Sample', axis=1)
df_Train = TrainData.values.tolist()


TestData= pd.read_csv("Modified_TestData.csv")
TestData = TestData.drop('Sample', axis=1)
df_Test = TestData.values.tolist()


def Euclideandistance(x,xi, length):
    d = 0.0
    for i in range(length):
        d += pow(float(x[i])- float(xi[i]),2)
    return math.sqrt(d)


def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = Euclideandistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def MajorityVotes(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]


def CalcAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0



predictions = []
counter = 0
coun = []
for x in range(len(df_Test)):
     neighbors = getNeighbors(df_Train, df_Test[x], 9)
     result = MajorityVotes(neighbors)
     predictions.append(result)

     if result == df_Test[x][-1]:
        counter += 1
        coun.append(counter)

print("Number of correctly classified instances :",counter)
print("Total number of instances :", len(df_Test))

accuracy = CalcAccuracy(df_Test, predictions)
print("Accuracy :", accuracy,"%", "\n")

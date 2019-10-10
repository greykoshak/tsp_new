
import numpy as np
import math
import copy

class City:

    def __init__(self, x, y):
        self.x = x
        self.y = y

def distance(c1, c2):
    return math.sqrt((c1.x - c2.x) ** 2 + (c1.y - c2.y) ** 2)

def TSP(distances, numberOfCity, startNode):

    assert distances.shape[0] - 1 == numberOfCity
    listOfDic = []
    for i in range(numberOfCity + 1):
        listOfDic.append({})

    S = [0] * numberOfCity
    S[0] = 1
    listOfDic[1][convert(S)] = 0

    for m in range(2, numberOfCity + 1, 1):
        potentialS = constructList(m, numberOfCity)
        for eachS in potentialS:

            selectedCities = []
            for index in range(1, len(eachS)):
                if eachS[index] == 1:
                    selectedCities.append(index + 1)


            currentKey = convert(eachS)
            for city in selectedCities:
                copyS = copy.copy(eachS)
                copyS[city - 1] = 0
                respectiveKey = convert(copyS)

                copyCities = copy.copy(selectedCities)
                copyCities.remove(city)

                if not copyCities:
                    listOfDic[city][currentKey] = distances[1][city]
                    continue

                minDistance = distances[copyCities[0]][city] + listOfDic[copyCities[0]][respectiveKey]
                for testCity in copyCities[1:]:
                    tempResult = distances[testCity][city] + listOfDic[testCity][respectiveKey]
                    if minDistance > tempResult:
                        minDistance = tempResult

                listOfDic[city][currentKey] = minDistance

    finalS = [1] * numberOfCity
    finalKey = convert(finalS)

    finalDistance = distances[1][2] + listOfDic[2][finalKey]
    for j in range(3, numberOfCity + 1, 1):
        tempDis = distances[1][j] + listOfDic[j][finalKey]
        if finalDistance > tempDis:
            finalDistance = tempDis

    return finalDistance


def constructList(m, numberOfCities):

    if m == 1:
        S = [0] * numberOfCities
        S[0] = 1
        temp = []
        temp.append(S)

        return temp

    candidates = constructList(m-1, numberOfCities)

    lst = []

    for c in candidates:
        copyC = copy.copy(c)

        for index in range(1, numberOfCities, 1):
            if copyC[index] == 0:
                anotherCopy = copy.copy(copyC)
                anotherCopy[index] = 1

                if anotherCopy not in lst:
                    lst.append(anotherCopy)


    return lst

def convert(lst):

    result = 0
    for item in lst:
        result = result * 2 + item
    return result

def restore(result):
    lst = []
    while result / 2 != 0:
        item = result % 2
        result /= 2
        lst.insert(0, item)
    lst.insert(0, item)

    return lst


if __name__ == "__main__":

    import time

    file = open("tsp.txt", "r").readlines()
    number = int(file[0][:-1])

    # Read in distances
    cities = []
    for line in file[1:]:
        line = line.split(" ")
        x = float(line[0])
        y = float(line[1][:-1])

        c = City(x, y)
        cities.append(c)

    distances = np.zeros((number + 1, number + 1))
    for i in range(1, number + 1, 1):
        for j in range(i, number + 1, 1):

            if i == j:
                continue

            distances[i][j] = distance(cities[i-1], cities[j-1])
            # print distances[i][j]
            distances[j][i] = distances[i][j]

    # Process TSP
    print time.ctime()
    result = TSP(distances, number, 1)
    print time.ctime()

    print result



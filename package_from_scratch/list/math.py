#   Returns the max value.
def findMax(inputList:list):
    index = 0
    for i in range(len(inputList)):
        if inputList[i]>inputList[index]:
            index = i
    return index ,inputList[index]

#   Returns the min value
def findMin(inputList:list):
    index = 0
    for i in range(len(inputList)):
        if inputList[i]<inputList[index]:
            index = i
    return index ,inputList[index]

#   Returns summation of all values in a list
def sumation(inputList:list):
    sum = 0
    for element in inputList:
        sum += element
    return sum

#   Returns the length of list.
def length(inputList:list):
    return len(inputList)

#   Computes the average
def mean(inputList:list):
    return sumation(inputList)/length(inputList)

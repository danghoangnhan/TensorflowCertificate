from package_from_scratch.list.math import length
from package_from_scratch.list.sort import selectionSort
from package_from_scratch.list.math import sumation,length,mean
from package_from_scratch.list.math import findMax, findMin
import math

#   Computes the median. 
#   Median is the middle value in a sorted list of values.
def median(inputList:list):
    sorted_list = selectionSort(inputList)
    n = length(sorted_list)
    middle_index = n // 2
    if n % 2 == 0:
        median = (sorted_list[middle_index - 1] + sorted_list[middle_index]) / 2
    else:
        median = sorted_list[middle_index]
    return median



#   Computes the mean deviation
def mean_deviation(inputList:list):
    meanValue = mean(inputList=inputList)
    deviations = [abs(x - meanValue) for x in inputList]
    return sumation(deviations) / length(deviations)

#   Computes the standard deviation.
#   Standard Deviation is useful to give an idea about range of normal values(i.e. location of most of values).
def standard_deviation(inputList:list):
    varianceValue = variance(inputList)
    std_dev = math.sqrt(varianceValue)
    return std_dev

#   Computes the variance. Variance is useful to see how the list of values varied 
#   against the average.
def occurrence(inputList:list,value):
    hashMap ={}
    for element in inputList:
        if element in hashMap.keys():
            hashMap[element]+=1
        else:
            hashMap[element]=1
    return hashMap[value]

#   Returns the length of list.
def length(inputList:list):
    return len(inputList)

#   Calculate the difference between max and min values using ptp() function
def rangeMaxMin(inputList:list):
    _,maxValue = findMax(inputList=inputList)
    _,minValue = findMin(inputList=inputList)
    return maxValue - minValue

#   Computes the variance.
#   Variance is useful to see how the list of values varied against the average.
def variance(inputList:list):
    meanValue = mean(inputList=inputList)
    return sumation((i - meanValue) ** 2 for i in inputList) / length(inputList)
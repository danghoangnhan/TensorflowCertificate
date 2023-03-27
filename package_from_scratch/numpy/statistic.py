import numpy as np
import collections

#   Computes the variance.
#   Variance is useful to see how the list of values varied against the average.
def variance(inputList:list):
    return  np.var(inputList)

#   Returns the max value.
def findMax(inputList:list):
    return np.amax(inputList)

#   Returns the min value
def findMin(inputList:list):
    return np.amin(inputList)

#   Computes the median. 
#   Median is the middle value in a sorted list of values.
def median(inputList:list):
    return np.median(inputList)

#   Computes the mean deviation
def mean_deviation(inputList:list):
    return np.mean(np.abs(inputList - np.mean(inputList)))

#   Computes the standard deviation.
#   Standard Deviation is useful to give an idea about range of normal values(i.e. location of most of values).
def standard_deviation(inputList:list):
    return np.std(inputList)

#   Computes the variance. Variance is useful to see how the list of values varied 
#   against the average.
def occurrence(inputList:list,value):
    arr = np.array(inputList)
    return np.count_nonzero(arr == value)

#Sorts the list.
def sort(inputList:list):
    return np.sort(inputList)

#   Computes the average
def mean(inputList:list):
    return np.mean(inputList)

#   Returns summation of all values in a list
def sumation(inputList:list):
    return np.sum(inputList)

#   Computes the variance. Variance is useful to see how the list of values varied 
#   against the average.
def occurrence(inputList:list,value):
    counter = collections.Counter(inputList)
    return counter[value]


#   Calculate the difference between max and min values using ptp() function
def rangeMaxMin(inputList:list):
    return np.ptp(inputList)

#   Returns the length of list.
def length(inputList:list):
    return len(inputList)
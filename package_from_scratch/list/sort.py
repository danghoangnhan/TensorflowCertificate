from package_from_scratch.list.math import findMin

#Sorts the list.
def bubbleSort(sample):
    i = len(sample)
    while i > 1:
        i-=1
        for j in range(i):
            if sample[j]>sample[j+1]:
                tmp = sample[j]
                sample[j] = sample[j+1]
                sample[j+1] = tmp
    return sample

def selectionSort(sample):
    result = []
    while len(sample)>0 :
        index ,minValue = findMin(sample)
        result.append(minValue)
        sample = sample[:index]+sample[index+1:]
    return result
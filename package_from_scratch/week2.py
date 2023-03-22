import numpy as np

def displayRow(matrix):
    for row in matrix:
        print(row,'\n')

def tranpose(maxtrix):
    new_result = []
    for i in range(len(maxtrix[0])):
        cols = []
        for j in range(len(maxtrix)):
            cols.append(maxtrix[j][i])
        new_result.append(cols)
    return new_result

def cloneAnTranpose(matrix,value):
    new_result = []
    for _ in range(len(matrix[0])):
        cols = []
        for _ in range(len(matrix)):
            cols.append(value)
        new_result.append(cols)
    return new_result

def cloneMatrix(matrix,value):
    new_result = []
    for _ in range(len(matrix)):
        cols = []
        for _ in range(len(matrix[0])):
            cols.append(value)
        new_result.append(cols)
    return new_result

def add2Matrix(A,B):
    result = cloneMatrix(A,0)
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j]+B[i][j]
    return result
def sub2Matrix(A,B):
    result = cloneMatrix(A,0)
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j]-B[i][j]
    return result

def Mutil2Matrix(A,B):
    result = cloneMatrix(A,0)
    for i in range(len(A)):
        for j in range(len(B[0])):
                for k in range(len(A[0])):
                    result[i][j]+=A[i][k]*B[k][j]
    return result



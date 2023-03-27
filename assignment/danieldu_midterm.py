def mulmatrix(X,Y):
    result = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*Y)] for X_row in X]
    return result

def identify(I_row,I_cols):
    I = []
    for i in range(I_row):
        row_temp = []
        for j in range(I_cols):
            if i == j:
                row_temp.append(1)
            else:
                row_temp.append(0)
        I.append(row_temp)
    return I
def transposeMatrix(X):
    return list(map(list,zip(*X)))

def getMatrixMinor(X,i,j):
    return [row[:j]+row[j+1:] for row in (X[:i]+X[i+1:])]

def getMatrixDeternminant(X):
    if len(X) == 2:
        return X[0][0] * X[1][1] - X[0][1] * X[1][0]
    determinant = 0
    for c in range(len(X)):
        determinant += ((-1)**c)*X[0][c]*getMatrixDeternminant(getMatrixMinor(X,0,c))
    return determinant

def getMatrixInverse(X):
    determinant = getMatrixDeternminant(X)
    if len(X) == 2:
        return [[X[1][1]/determinant,-1*X[0][1]/determinant],
                [-1*X[1][0]/determinant,X[0][0]/determinant]]
    cofactors = []
    for r in range(len(X)):
        cofactorRow = []
        for c in range(len(X)):
            minor = getMatrixMinor(X,r,c)
            cofactorRow.append(((-1)**(r+c))*getMatrixDeternminant(minor))
        cofactors.append(cofactorRow)
    cofactors = transposeMatrix(cofactors)
    for r in range(len(cofactors)):
        for c in range(len(cofactors)):
            cofactors[r][c] = cofactors[r][c]/determinant
    return cofactors

def getDx(A,b,cols):
    result = [[0 for _ in range(len(A[0]))] for _ in range(len(A))]
    
    for i in range(len(A)):
        for j in range(len(A[0])):
            if j == cols-1:
                result[i][j] = b[0][i]
            else:
                result[i][j] = A[i][j]
    return result
def sameMatrix(A,B)->bool:    
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] != B[i][j]:
                return False
    return True

def method1(A,b):
    
    DX = [getDx(A,b,i+1) for i in range(len(A))]
    X = [[getMatrixDeternminant(DX[i])/getMatrixDeternminant(A) for i in range(len(A))]]
    return X

def method2(A,b):
    
    x = mulmatrix(getMatrixInverse(A),transposeMatrix(b))
    return transposeMatrix(x)



a = [[2, -4, 4], [34, 3, -1], [1, 1, 1]]
b = [[8,30,108]]
print("A :",a)
print("B :",b)

print("solve by method 1")
print(method1(a,b))
print("solve by method 2")
print(method2(a,b))
X1 = method1(a,b)
X2 = method2(a,b)



def test(A,X,B):
    mul = mulmatrix(X,transposeMatrix(A))
    print("our result: ",mul,'actual result',B,'\n')

print("test method 1",'\n')
test(a,X1,b)

print("test method 2",'\n')
test(a,X2,b)


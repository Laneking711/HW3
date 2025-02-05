import copy
import matrixOperations as mo

def GaussSeidel(Aaug, x, Niter=15, epsilon=1e-5):
    '''
    This is Gauss-Seidel iterative solution to a set of equations in an augmented matrix.
    Step 1:  Ensure the matrix is diagonal dominant (i.e., put the largest coefficient for a diagonal term on the diagonal
    by interchanging rows as necessary.  Note:  start at top and work your way to the bottom.
    Step 2:  Solve each row for corresponding x[row] in sequencence, using most recent values on rhs
    Solve first equation for x[0]=(A[0][n]-(A[0][1]*x[1]+A[0][2]*x[2]...A[0][n-1]*x[n-1]))/A[0][0]
    Updated value for x[0] is used in solution for x[1], etc.
    Once you've solved all the way to x[n-1], this is one iteration.
    Step 3:  Keep iterating for Niter or until the maximum change in a row of x is < epsilon
    :param Aaug: the augmented matrix
    :param x: the initial guess vector
    :param Niter: number of iterations to get correct x
    :param epsilon: the precision for early escape from iteration.
    :return: x solution vector
    '''
    AA = copy.deepcopy(Aaug)  # deep copy Aaug so that we are not altering Aaug unintentionally
    # Step 1:
    AA = DiagDominant(AA)  # ensure matrix is diagonal dominant

    # Steps 2 & 3
    n = len(x)
    for j in range(Niter):  # main iteration loop
        maxErr = 0  # for calculating maximum change in one of the x values
        for r in range(n):  # update the x vector one element at a time
            xOld = x[r]
            rhs = AA[r][n]  # the value from last col of row r (i.e., from b vector of augmented matrix)
            for c in range(n):  # move all the non-diagonal terms to right-hand side (rhs)
                if c != r:
                    rhs -= AA[r][c] * x[c]
            x[r] = rhs / AA[r][r]  # divide rhs by coefficient of diagonal term and update x[r]
            maxErr = max(maxErr, abs(xOld - x[r]))
        if maxErr <= epsilon:
            break
    return x

def DiagDominant(A):
    """
    This function makes the matrix A diagonal dominant.
    :param A:  a matrix
    """
    AA = copy.deepcopy(A)  # deep copy A so that we don't unintentionally change it
    rows = len(AA)
    for i in range(rows):  # iterate through all rows of AA
        c = abs(AA[i][i])  # grab value along diagonal for current row
        for k in range(i + 1, rows):  # scan rows below i
            if abs(AA[k][i]) > c:  # if this row has a larger absolute value for diag, exchange row i and k
                row = AA.pop(k)
                AA.insert(i, row)
                c = abs(AA[i][i])  # set c to this larger absolute value
    return AA  # return the diagonal dominant AA

def separateAugmented(Aaug):
    """
    This function separates the last column from Aaug and returns a tuple with the A, b
    :param Aaug: the augmented matrix
    :return: (A,b)
    """
    A=copy.deepcopy(Aaug)
    b=[]
    n=len(A[0])-1
    for r in A:
        b.append(r.pop(n))
    b=mo.makeColumnVector(b)
    return (A,b)

def checkMatrixSoln(A, x, augmented=True):
    """
    I want to check to see if answer vector x is correct and matrix multiplication gives b
    :param Aaug: The augmented matrix
    :param x: The solution vector transpose (a row vector)
    :return: The b vector transpose (a row vector)
    """
    if augmented:  # If A is augmented, strip off the last column
        AA,b=separateAugmented(A)
    else:
        AA=A
    B=[]  # the result of multiplying AA*x
    for r in AA:
        s = 0
        rCntr = 0
        for c in r:
            s += c * x[rCntr]
            rCntr += 1
        B.append(s)
    return B

def matrixMult(A,B):
    """
    Multiplies matrix A by Matrix B.  Note:  you are responsible for making sure nxm*mxp gives nxp
    :param A: a matrix
    :param B: another matrix
    :return: the matrix with size nxp
    """
    ARows=len(A)
    BCols=len(B[0])
    C=[[0 for c in range(BCols)] for r in range(ARows)]
    for r in range(ARows):
        for c in range(BCols):
            C[r][c]=multVecs(A[r],getCol(B,c))
    return C

def getCol(A, c):
    """
    Gets a column from A matrix
    :param A: a matrix
    :param c: the index for column you want
    :return: the row vector corresponding to column c
    """
    vec=[]
    for r in A:
        vec.append(r[c])
    return vec

def multVecs(A,B):
    """
    simply multiplies the vectors.  Note:  they should both be row vectors of same length
    :param A: a row vector
    :param B: another row vector
    :return: the product of the multiplication, a scalar
    """
    s=0
    for a in range(len(A)):
        s+=A[a]*B[a]
    return s

def main():
    # from problem statement
    A1 = [[3, 1, -1, 2],
          [1, 4, 1, 12],
          [2, 1, 2, 10]]

    x1 = [0, 0, 0]

    A2 = [[1, -10, 2, 4, 2],
          [3, 1, 4, 12, 12],
          [9, 2, 3, 4, 21],
          [-1, 2, 7, 3, 37]]

    x2 = [1, 1, 1, 1]

    xSoln1 = GaussSeidel(A1, x1, Niter=22)
    xSoln2 = GaussSeidel(A2, x2, Niter=50)
    print("xSoln1=", [round(x, 4) for x in xSoln1])
    print("xSoln2=", [round(x, 4) for x in xSoln2])

# The following bit of code will run main() if we are executing this file in debut or run mode directly, but not if
# importing it as a module in another .py file.  This is useful for testing and debugging.
if __name__ == "__main__":
    main()

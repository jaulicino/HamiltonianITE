import numpy as np
import numpy.linalg as npla
import pennylane as qml
import math
import os

from itertools import product
from itertools import combinations
from matplotlib import pyplot as plt


pauli_dict = {"I": qml.Identity, "X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}

PauliX = np.array([[0.0,1.0],[1.0,0.0]],dtype=complex)
PauliY = np.array([[0.0,-1.j],[1.j,0.0]],dtype=complex)
PauliZ = np.array([[1.0,0.0],[0.0,-1.0]],dtype=complex)
I2 = np.array([[1.0,0.0],[0.0,1.0]],dtype=complex)
PauliDict= {}
PauliDict["I"] = I2
PauliDict["X"] = PauliX
PauliDict["Y"] = PauliY
PauliDict["Z"] = PauliZ

def chainKron(letters):
    if(len(letters) == 1):
        return PauliDict[letters[0]]
    elif(len(letters) == 2):
        return np.kron(PauliDict[letters[0]], PauliDict[letters[1]])
    else:
        K = np.kron(PauliDict[letters[0]], PauliDict[letters[1]])
        return np.kron(K,chainKron(letters[2:]))


def heisenberg_letters(J,U,B,n=4):
    letters = []
    coeffs = []
    for comb in list(combinations(list(range(n)),2)):
        for le in "XY":
            s1 = ["I"]*n
            s1[comb[0]] = le
            s1[comb[1]] = le
            s1 = "".join(s1)
            letters += [s1]
            coeffs += [J]
        s1 = ["I"]*n
        s1[comb[0]] = "Z"
        s1[comb[1]] = "Z"
        s1 = "".join(s1)
        letters += [s1]
        coeffs += [U]
    for i in range(n):
        s1 = ["I"]*n
        s1[i] = "Z"
        s1 = "".join(s1)
        letters += [s1]
        coeffs += [B]
    return letters, coeffs


def PQ_merge(P,Q):
    '''

    Multiply (or merge) two Pauli strings.

    Originally written by Dr. Bo Peng, PNNL Computational Chemistry

            two strings  ---->   single string

    '''
    L = len(P)
    nimag = 0 # numberf of 'i'
    nsign = 0 # number of '-1'
    I = ''
    for i in range(0,L):
        if P[i] == 'I':
            I = os.path.join(I+str(Q[i]))
        elif Q[i] == 'I':
            I = os.path.join(I+str(P[i]))
        elif P[i] == Q[i]:
            I = os.path.join(I+str('I'))
        elif P[i] == 'X':
            if Q[i] == 'Y':
                I = os.path.join(I+str('Z'))
                nimag += 1
            elif Q[i] == 'Z':
                I = os.path.join(I+str('Y'))
                nimag += 1
                nsign += 1
        elif P[i] == 'Y':
            if Q[i] == 'X':
                I = os.path.join(I+str('Z'))
                nimag += 1
                nsign += 1
            elif Q[i] == 'Z':
                I = os.path.join(I+str('X'))
                nimag += 1
        elif P[i] == 'Z':
            if Q[i] == 'X':
                I = os.path.join(I+str('Y'))
                nimag += 1
            elif Q[i] == 'Y':
                I = os.path.join(I+str('X'))
                nimag += 1
                nsign += 1
    #
    sgn = ((1j)**nimag)*((-1)**nsign)
    #
    return I,sgn


def mergn(letters):
    '''
    
    Merge multiple pauli words

        list of strings  ---->   single string

    '''
    
    if(len(letters) == 2):
        return PQ_merge(letters[0], letters[1])
    else:
        a, b = PQ_merge(letters[0], letters[1])
        c, d = mergn([a]+letters[2:])
        return c, d*b  


def poly(H,P):

    '''
    
    Taylor series expansion for operator H and coefficients P, P[0] = constant, P[n] -> P[n] * H**n

        np.array (N x N), np.array (N) ---> np.array (N x N)

    '''

    M = np.eye(len(A), dtype=complex)
    for i in range(len(P)):
        M+=npla.matrix_power(A, i+1)*P[i]
    return M

def PadeExpansion(TaylorOrder, m):
    '''

    Compute Pade(P,Q) for polynomial with coefficients (1,c1,c2,...,ck)
    here, Coeff contains the polynomial coeffcients (c1,c2,...,ck)
    P contains (a1,a2,...,am) and Q contains (b1,b2,...,bn)
    constant term is normalized
    m is given, k is the length of Coeff, n = k - m

    '''
    k = TaylorOrder
    
    # Here, the expansion we care about is exp(x), so we initialize the coefficient vector,
    # which represents the coefficient of the function's maclaurian series,
    # to the series of the exponential...
    
    Coeff = np.array([1/np.math.factorial(i+1) for i in range(k)], dtype="float")
    
    # Now, we set up the matrices to represent the system of equations we want to solve
    n=k-m
    M = np.zeros((k,k))
    M[:m, :m] = np.eye(m)
    row = 0
    for i in range(m,k):
        M[row,i] = -1.0
        row += 1
        M[row:,i] = -1.0*np.copy(Coeff[:(k-row)])
    PQ = npla.solve(M, Coeff)
    P = PQ[:m]
    Q = PQ[m:]
    
    return P, Q


'''

Hamiltonian Class

'''

class Hamiltonian():
    def __init__(self, letters, coeffs):
        '''        
        Initialize a Hamiltonian
        
        The Hamiltonian class has a few useful methods shown below. This should help streamline the Hamiltonian
        powers workflow
        
        letters = list of Hamiltonian elements i.e.
                        ["IXXI","YXZI"]
        
        coeffs = list of corresponding coefficients i.e.
                        [0.2, 0.7]
                        
                        -------->
                        
                        H = 0.2 * IXXI + 0.7 * YXZI
        
        '''
        self.letters = letters
        self.coeffs = coeffs
        
    def condense(self):
        
        '''
            Combine duplicate pauli words and their coefficients and remove terms with coefficient = 0
    
        '''
        
        dict_ = {}
        for i in range(len(self.letters)):
            if(self.letters[i] in dict_):
                dict_[self.letters[i]] += self.coeffs[i]
            else:
                dict_[self.letters[i]] = self.coeffs[i]
        kPop = []
        for key in dict_:
            if np.abs(dict_[key]) == 0:
                kPop += [key]
        for i in kPop: 
            dict_.pop(i)
        letters = list(dict_.keys())
        values = list(dict_.values())
        return Hamiltonian(letters, values)
    
    def multiply(self, HB):

        '''
            Multiply self by Hamiltonian HB, condenses and removes duplicates

                Hamiltonian, Hamiltonian ---> Hamiltonian

        '''
        
        prod = list(product(self.letters, HB.letters))
        coeffs = list(product(self.coeffs, HB.coeffs))
        assert len(coeffs) == len(prod), "Error in Hamiltonian! Coeffs length != Combinations Length!"
        
        lettersP = []
        coeffsP = []
        for i in range(len(prod)):
            l, b = mergn(list(prod[i]))
            c = np.prod(coeffs[i])
            c *= b
            lettersP += [l]
            coeffsP += [c]
            
        return_hamiltonian = Hamiltonian(lettersP, coeffsP)
        return return_hamiltonian.condense()
        
    def powerFast(self, n):

        '''
            Raise self Hamiltonian to power n. Calls itself recursively with condensing each round.

                Hamiltonian, int ---> Hamiltonian

        '''


        if(n == 0):
            m = len(self.letters[0])
            return Hamiltonian(["I"*m],[1.0])
        elif(n == 1):
            return self
        elif(n == 2):
            return self.multiply(self)
        else:
            HB_ = self.multiply(self)
            return HB_.multiply(self.powerFast(n-2))
        
        
    def power(self, n):
        
        '''
            Slow version of powerFast. Calculates cartesian product. Depracated

                Hamiltonian, int ---> Hamiltonian

        '''
        
        n = int(n)
        prod = list(product(*[self.letters] * n))
        coeffs = list(product(*[self.coeffs] * n))
        assert len(coeffs) == len(prod), "Error in Hamiltonian! Coeffs length != Combinations Length!"
       
        # now merge each block together and combine coefficients
        
        lettersP = []
        coeffsP = []
        for i in range(len(prod)):
            l, b = mergn(list(prod[i]))
            c = np.prod(coeffs[i])
            c *= b
            lettersP += [l]
            coeffsP += [c]
            
        return_hamiltonian = Hamiltonian(lettersP, coeffsP)
        return return_hamiltonian.condense()
    
    def multScalar(self, c):

        '''
            Multiply Hamiltonian by a scalar

                Hamiltonian, double ---> Hamiltonian

        '''


        cprime = [i*c for i in self.coeffs]
        return Hamiltonian(self.letters, cprime)
    
    def add(self, HB):
        
        '''
            Add two different Hamiltonians

                Hamiltonian, Hamiltonian ---> Hamiltonian
    
        '''
        letters = self.letters + HB.letters
        coeffs = self.coeffs + HB.coeffs
        return Hamiltonian(letters, coeffs).condense()
        
    def exp(self,n=3):

        '''
            Estimate the exponential of a Hamiltonian with an n order Taylor expansion

                Hamiltonian, int ---> Hamiltonian
    
        '''
        m = len(self.letters[0])
        Hret = Hamiltonian(["I"*m],[1.0])
        for i in range(1,n):
            Hexp = self.powerFast(i).multScalar(1/(math.factorial(i)))
            Hret = Hret.add(Hexp)
            Hret = Hret.condense()
            Hret = Hret.clean()
        return Hret

    def toMatrix(self):
        
        '''
            Convert Hamiltonian to a matrix

                Hamiltonian ---> np.array (N x N)
    
        '''


        n = int(len(self.letters[0]))
        arr = np.zeros((2**n, 2**n), dtype=complex)
        for i in range(len(self.letters)):
            A = chainKron(self.letters[i])
            arr += self.coeffs[i] * A 
        return arr

    def clean(self, tol=1e8):

        '''
            Clean hamiltonian based on a given tolerance

                Hamiltonian, **tolerance ---> Hamiltonian

        '''
        Hret = self.condense()
        max_ = np.max(np.abs(self.coeffs))
        non_dels = []
        for i in range(len(self.coeffs)):
            if(max_/np.abs(self.coeffs[i]) > tol):
                pass;
            else:
                non_dels += [i]
        cleanCoeffs = list(np.array(self.coeffs)[non_dels])
        cleanLetters = list(np.array(self.letters)[non_dels])
        return Hamiltonian(cleanLetters, cleanCoeffs)
    
    def toPennylaneGroup(self):

        '''
            Convert a Hamiltonian object to a pennylane compatible object
            
                Hamiltonian ---> list of observables (pennylane tensors)


        '''


        words = self.letters
        coeffs = self.coeffs
        group = []
        for i in range(len(words)):
            for j in range(len(words[i])):
                g = pauli_dict[words[i][j]]
                if(j==0):
                    A = g(0)
                else:
                    A = A @ g(j)
            group += [A]
        return group
    
    def grouping(self):

        '''
            Convert a Hamiltonian object to a list of grouped observables

                Hamiltonian ---> list of observables, list of floats
        '''


        group = self.toPennylaneGroup()
        coeffs = self.coeffs
        obs_grouping, coeffs_grouping = qml.grouping.group_observables(group, coeffs)
        return obs_grouping, coeffs_grouping
    


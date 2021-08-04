<h1> Hamiltonian ITE </h1>

The codes for this module were written during the DOE WDTS Science Undergraduate Laboratory and later compiled into a neat module. I would like to thank the DOE, SULI, and Dr. Bo Peng.

<h2> Hamiltonian Class </h2>

The Hamiltonian modules contains codes of a Hamiltonian object which the user can manipulate in a number of ways. Hamiltonians are defined as a linear combination of Pauli Strings and their respective coefficients. For instance,

`Hamil = Hamiltonian(["XZXZY, IIIIX"], [0.25,1.j])`


The codes provide functionality for operator multiplication, multiplication by a scalar, addition of operators, and grouping operators into functional qubit wise commuting groups (along with many others). 

`Hamil = Hamil.multScalar(0.238+1.j)`

Operator multiplications are conducted using the Pauli comutation rules. This allows for operator operations which may not be classically tractable if one were to represent their Hamiltonian as a matrix.

<h2> ITE </h2>

In order to run an ITE scheme, first build a hamiltonian. For instance,

`Hamil = Hamiltonian(["XZXZY, IIIIX"], [0.25,1.j])`



***Authors:*** Joe Aulicino, Dr. Bo Peng 

<img src=https://upload.wikimedia.org/wikipedia/en/thumb/1/17/Pacific_Northwest_National_Laboratory_logo.svg/1200px-Pacific_Northwest_National_Laboratory_logo.svg.png width="500px"/>



# Skew-Symmetric Matrices

A skew-symmetric (or antisymmetric or antimetric) matrix is a square matrix
whose transpose is also its negative; that is, it satisfies the condition $−A =
A^{T}$. If the entry in the i-th row and j-th column is $a_{ij}$, i.e. $A =
(a_{ij})$ then the skew symmetric condition is $a_{ij} = −a_{ji}$. For example,
the following matrix is skew-symmetric:

\begin{equation}
    A =
    \begin{bmatrix}
        0 & 2 & -1 \\
        -2 & 0 & -4 \\
        1 & 4 & 0
    \end{bmatrix}
\end{equation}

Imagine a column vector ${\bf A} = (A_1, A_2, A_3)$ and define the matrix

$$ A_\times = \left(\begin{array}{ccc} 0 & -A_3 & A_2 \\ A_3 & 0 & -A_1 \\
-A_2 & A_1 & 0 \end{array}\right) $$

Note that if ${\bf B}$ is another column vector, then:

$$ A_\times {\bf B} = {\bf A}\times {\bf B} $$

Moreover:

$$ {\rm Transpose}(A_\times) = -A_\times $$

The skew-symmetric product **generalizes the concept to arbitrary
dimensions**, not just 3 dimensions like the cross-product.

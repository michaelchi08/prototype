# Characteristic Polynomial

In linear algebra, the characteristic polynomial of a square matrix is a
polynomial which is invariant under matrix similarity and has the eigenvalues
as roots. It is a common technique used to calculate the eigenvalues of the
square matrix by setting the characteristic polynomial to zero.

Let $A$ by an $n \times n$ matrix, the characteristic polynomal is defined as:

\begin{equation}
    f(\lambda) = \det(A - \lambda I)
\end{equation}

Or

\begin{equation}
    f(\lambda) = \det(\lambda I - A)
\end{equation}

This is a polynomial in $\lambda$.

For instance, if

\begin{equation}
    A =
    \begin{bmatrix}
       a & b & c \\
       d & e & f \\
       g & h & i
    \end{bmatrix}
\end{equation}

then

\begin{equation}
    f(\lambda) = \det
    \begin{bmatrix}
       a & b & c \\
       d & e & f \\
       g & h & i
    \end{bmatrix} \\
    = (a - \lambda)((e - \lambda)(i - lambda) - fh) -
        b(d(i - \lambda) - gf) +
            c(dh - (e - \lambda) g)
\end{equation}

Which simplifies to some degree 3 polynomial in $\lambda$. The zeros of this
polynomial give the eigenvalues of $A$.

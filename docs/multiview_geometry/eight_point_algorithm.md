# Eight Point Algorithm

Eight Point Algorithm is an algorithm used in computer vision to estimate the
**essential matrix** $E$ or the **fundamental matrix** $F$. The algorithm's
name derives from the fact that it estimates the essential matrix or the
fundamental matrix from a set of eight (or more) corresponding image points.
However, variations of the algorithm can be used for fewer than eight points.

- Longuet-Higgins (1981): The 8-point algorithm is used to estimate the
  essential matrix $E$
- Hartley (1997): The normalized 8-point algorithm is used estimate the
  fundamental matrix $F$


## Normalized 8-Point Algorithm (Hartley 1997)

Algorithm Outline:

1. Normalize Point Correspondances: By translating and scaling the image points
to have the origin at the centroid of the image.
2. Linear Solution for the Fundamental Matrix: Approximate fundamental matrix
$F$ with SVD
3. Singularity Constraint: Minimize the Frobenius norm $||F - F^{\prime} ||$,
where $F$ and $F^{\prime}$ are the approximate fundamental matrix and
fundamental matrix respectively.


### Normalize Point Correspondances

To normalize data to be between 0 and 1, one can do so with:

\begin{equation}
    x^{\prime} = \dfrac{x - \text{min}(x)}{\text{max}(x) - \text{min}(x)}
\end{equation}

To normalize it to be between -1 and 1, one can use:

\begin{equation}
    x^{\prime} = 2 \dfrac{x - \text{min}(x)}{\text{max}(x) - \text{min}(x)} - 1
\end{equation}

For normalizing point correspondances we want to transform the coordinate
system to have the origin from the top left corner of the image to the centroid
of the image, this can be achieved with the latter of the two normalization
approaches outlined above, which results in the following formula:

\begin{align}
   x^{\prime} &= 2 \dfrac{x}{w} - 1 \\
   y^{\prime} &= 2 \dfrac{y}{h} - 1 \\
   z^{\prime} &= z
\end{align}

Where $x^{\prime}$, $y^{\prime}$ and $z^{\prime}$ are normalized pixel
coordinates of $x$, $y$ and $z$, $w$ and $h$ are the width and height of the
image in pixels. The normalization can be written in matrix form:

\begin{equation}
    N = \begin{bmatrix}
       2 / w & 0 & -1 \\
       0 & 2 / h & -1 \\
       0 & 0 & 1
    \end{bmatrix}
\end{equation}

Where $N$ is the normalization matrix, we assume that the correspondnace point
has $z$ as a constant of $1$.

\begin{equation}
    p^{\prime} = N \cdot p^{\top}
\end{equation}



### Linear Solution for the Fundamental Matrix
Epipolar Constraint:

\begin{equation}
    x_{1}^{\top} E x_{2} = 0
\end{equation}


Let $a$ be the Kronecker Product of $x_{1}$ and $x_{2}$:

\begin{equation}
    a = x_{1} \otimes x_{2}
\end{equation}

\begin{equation}
    a = (
        x_{1} x_{2},
        x_{1} y_{2},
        x_{1} z_{2},
        y_{1} x_{2},
        y_{1} y_{2},
        y_{1} z_{2},
        z_{1} x_{2},
        z_{1} y_{2},
        z_{1} z_{2}
    )^{\top} \quad{\in {\rm I\! R}^{9}}
\end{equation}

The Epipolar Constraint becomes where $E^{s}$ is $E$ stacked into a $1 \times
9$ vector:

\begin{equation}
    a E^{s} = 0
\end{equation}

For $n$ point pairs

\begin{equation}
    A = (a_{1}, ..., a_{n})^{\top}
\end{equation}

\begin{equation}
    A E^{s} = 0
\end{equation}

We can solve the above linear system of equations using Single Value
Decomposition (SVD) method.


    [U, S, V] = svd(A)
    F = reshape(V(:, 9), 3, 3);

We obtain the least eigen-vector in $V$ by extracting the 9-th column and build
the approximate fundamental matrix $F$.

**IMPORTANT**: In computing the approximate fundamental matrix $F$, we assume
that the SVD used to compute $A$ returns a decreasing order of singular
diagonal values in $S$, in which case the last eigen-value in $S$ corresponds
to the least sensitive eigen-vector in $V$ (i.e. the 9-th column).


### Singularity Constraint

The approximate fundamental matrix $F$ calculated with SVD does not guarantee
the matrix is of rank 2. The most convenient way is to correct the matrix $F$
by minimizing the Frobenius norm $|| F - F^{\prime} ||$, subject to the
condition $\text{det}(F^{\prime}) = 0$.

\begin{equation}
    F = U \Sigma V^{\top}
\end{equation}

Where $\Sigma$ is:

\begin{equation}
    S = \text{diag}(1, 1, 0)
\end{equation}

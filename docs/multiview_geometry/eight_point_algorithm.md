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


## Derivation

First we rewrite the epipolar constraint as a scalar product in the elements of
the matrix $E$ and the coordinates of hte points $x_{1}$ and $x_{2}$. Let

\begin{equation}
    E^{s} = (
        e_{11},
        e_{21},
        e_{31},
        e_{12},
        e_{22},
        e_{32},
        e_{13},
        e_{23},
        e_{33}
    )^{\top} \quad \in {\rm I\!R}^{9}
\end{equation}

be the vector of elements of $E$ ($E^{s}$ means it was a matrix, but is now a
stacked vector; aka vectorization) and

\begin{equation}
    a = \boldsymbol{x_{1}} \otimes \boldsymbol{x_{2}}
\end{equation}

the **Kronecker product** of the vectors $\boldsymbol{x_{i}} = (x_{i}, y_{i},
z_{i})$, defined as:

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
    )^{\top} \quad \in {\rm I\!R}^{9}
\end{equation}

then the epipolar constraint can be written as:

\begin{equation}
    \boldsymbol{x_{2}}^{\top} E \boldsymbol{x_{1}} = a^{\top} E^{s} = 0
\end{equation}

For $n$ point pairs, we can combine this into the linear system:

\begin{equation}
    \chi E^{s} = 0 \qquad
        \text{with } \chi = (a^{1}, a^{2}, \dots, a^{n})^{\top}
\end{equation}

We see that the vector of coefficients fo the essential matrix $E$ defines the
null space of the matrix $\chi$. In order for the above system to have a unique
solution (up to a scalaing factor and ruling out the trivial solution $E = 0$),
the rank of the matrix $\chi$ needs to be exactly 8. Therefore we need atleast
8 point pairs.

In certian degenerate cases, the solution for the essential matrix is not
unique even if we have 8 or more point pairs. One such example is the case that
all points lie on a line or on a plane.

Clearly, we will not be able to recover the sign of $E$. Since with each $E$,
there are two possible assignments of rotation $R$ and translation $T$, we
therefore end up with four possible solutions for rotation and translation.


### Projection onto Essential Space

The Numerically estimated coefficients $E^{s}$ will in general not correspond
to an essential matrix. One can resolve this problem by projecting it back to
the essential space.

Theorem (Projection onto essential space): Let $F \in {\rm I\!R}^{3 \times 3}$
be an arbitrary matrix with SVD

\begin{equation}
    F = U \text{ diag} \{ \lambda_{1}, \lambda_{2}, \lambda_{3} \} V^{\top} \\
        \text{with } \lambda_{1} \leq \lambda_{2} \leq \lambda_{3}
\end{equation}

Then the essential matrix $E$ which minimizes the Frobenius norm $|| F - E
||^{2}_{f}$ is given by

\begin{equation}
    E = U \text{ diag} \{ \sigma, \sigma, 0 \} V^{\top} \\
        \text{with } \sigma = \dfrac{\lambda_{1} + \lambda_{2}}{2}
\end{equation}


### Algorithm Outline

Given a set of $n = 8$ or more point pairs $\boldsymbol{x_{1}^{i}}$,
$\boldsymbol{x_{2}^{i}}$:

1. **Compute an approximation of the essential matrix**: Construct the matrix
$\chi = (a^{1}, \dots, a^{n})^{\top}$, where $a^{i} = \boldsymbol{x_{1}^{i}}
\otimes \boldsymbol{x_{2}^{i}}$. Find the vector $E^{s} \in {\rm I\!R}^{9}$
which minimizes $|| \chi E^{s} ||$ as the ninth column of $V_{\chi}$ in the SVD
$\chi = U_{\chi} \Sigma_{\chi} V_{\chi}^{\top}$. Unstack $E^{s}$ into $3 \times
3$ matrix E.

2. **Project onto essential space**: Compute the SVD $E = U \text{ diag} \{
\sigma_{1}, \sigma_{2}, \sigma_{3} \} V^{\top}$. Since in the reconstruction,
$E$ is only defined up to a scalar, we project $E$ onto the **normalized
essential space** by replacing the singular values $\sigma_{1}, \sigma_{2},
\sigma_{3}$ with $1, 1, 0$.

3. **Recover the displacement from the essential matrix**: The four possible
solutions for rotation and translation are:

\begin{equation}
    R = U R_{Z}^{\top} (\pm\frac{\pi}{2}) V^{\top} \\
    \hat{T} = U R_{Z} (\pm\frac{\pi}{2}) \Sigma U^{\top}
\end{equation}

with a rotation by $\pm\frac{\pi}{2}$ around $z$:

\begin{equation}
    R_{z}^{\top}(\pm\frac{\pi}{2}) =
    \left(
    \begin{array}{ccc}
       0 & \pm 1& 0 \\
       \mp 1& 0& 0 \\
       0 & 0 & 1
    \end{array}
    \right)
\end{equation}


## Normalized 8-Point Algorithm (Hartley 1997)

Algorithm Outline (again):

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

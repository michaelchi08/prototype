# Covariance Matrix

A covariance matrix is a matrix whose elements in the $i$ and $j$ position
is the covariance between the $i^{\text{th}}$ and $j^{\text{th}}$ elements
of a random vector. Each element of the vector is a scalar random
variable, either with a finite or infinite number of potential values
specified by a theoretical join probability distribution of all the random
variables.

Intuitively the covariance matrix generalizes the notion of variance to
multiple dimensions. As an example, the variation in a collection of
random points in a 2D space cannot be characterized fully by a single
number, nor would the variances in the $x$ and $y$ directions contain all
of the necessary information; a $2 \times 2$ matrix would be necessary to fully
characterize the 2D variation.

The variance and covariance are often displayed together in
a variance-covariance matrix (aka covariance matrix). The variances appear
along the diagonal and covariances appear int he off-diagnoal elements.


\begin{equation}
    V =
    \begin{bmatrix}
        \sum x_{1}^{2} / N
        & \sum x_{1} x_{2} / N
        & \dots
        & \sum x_{1} x_{c} / N \\
        \sum x_{2} x_{1} / N
        & \sum x_{2}^{2} / N
        & \dots
        & \sum x_{2} x_{c} / N \\
        \dots
        & \dots
        & \dots
        & \dots \\
        \sum x_{c} x_{1} / N
        & \sum x_{c} x_{2} / N
        & \dots
        & \sum x_{c}^{2} / N \\
    \end{bmatrix}
\end{equation}

Where:

- $V$: $c \times c$ variance-covariance matrix
- $N$: Number of elements in each of the $c$
- $x_{i}$: deviation from the $i^{\text{th}}$ element
- $\sum x_{i}^{2} / N$: variance of elements from the $i^{\text{th}}$ data
- $\sum x_{i} x_{j} / N$: covariance for elements from the $i^{\text{th}}$ and
  $j^{\text{th}}$ datasets



# Singular Value Decomposition

The Singular Value Decomposition (SVD) is a factorization of a real or complex
matrix. It is the generalization of the eigendecomposition of a positive
semidefinite normal matrix ( for example, a symmetric matrix with positive
eigenvalues) to any $m \times n$ matrix via an extension of polar
decomposition.

Formally the SVD of an $m \times n$ real or complex matrix $M$ is a
factorization of the form $U \Sigma V^{\ast}$, where:

- $U$ is an $m \times m$ real or complex unitary matrix
- $\Sigma$ is an $m \times n$ rectangular diagonal matrix with non-negative
  real numbers on the diagonal
- $V$ is an $n \times n$ real or complex unitary matrix

The diagonal entries $\sigma_{i}$ of $\Sigma$ are known as the sinular values
of $M$. The columns of $U$ and columns of $V$ are called the left-sinuglar
vectors and right-singular vectors of $M$ respectively.

Applications that employ the SVD include computing the pseudoinverse, least
squares fitting of data, multivariable control , matrix approximation and
determining the rank, range and null space of a matrix.

# Rank

The rank of a matrix $A$ (denoted as $\rho(A)$) is the maximum number of
independent rows (or, the maximum number of independent columns). The rank of
a square matrix $A_{n \times n}$ is non-singular (non-zero) only if its rank is
equal to $n$.


- **Full Rank** (Non-Singular)

\begin{equation}
    \rho(A) = \text{min}(n, m)
\end{equation}

- **Not Full Rank** (Singular)

\begin{equation}
    \rho(A) \lt \text{min}(n, m) \\
    \exists x \mid Ax = 0 \text{ (non-empty null space)}
\end{equation}

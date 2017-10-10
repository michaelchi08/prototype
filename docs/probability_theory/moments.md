# Moments

When working with mass distributions in dynamics, we often keep track of only
a few properties called the moments of mass. The same is true with PDFs. The
zeroth probability moment is always 1 since this is exactly the axiom of total
probability.

The first probability moment is known as the mean, $\mu$:

\begin{equation}
  \mu = E[\mathbf{x}] = \int \mathbf{x} p(\mathbf{x}) d\mathbf{x}
\end{equation}

Where $E[\cdot]$ denotes the expectation operator. For a general matrix
function $\mathbf{F(x)}$, the expectation is written as:

\begin{equation}
  E[\mathbf{F(x)}] = \int \mathbf{F(x)} p(\mathbf{x}) d\mathbf{x}
\end{equation}

but note that we must interpret this as:

\begin{equation}
  E[\mathbf{F(x)}]
    = [E[f_{ij}(x)]]
    = [\int f_{ij}(\mathbf{x}) p(\mathbf{x}) d\mathbf{x}
\end{equation}

The second moment is known as the covariance matrix $\sum$

\begin{equation}
  \sum = E[(x - \mu)(x - \mu)^{T}]
\end{equation}

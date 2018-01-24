# Cheat Sheet

**Contents**
- Probability Basics
- Gaussian Function
- Bayes Rule




## Probability Basics

**Expectation**:

\begin{align}
    \mu &= E[X] \\
    \mu &= \sum^{n}_{i = 1} x_{i} p(x_{i}) \big\} \text{discrete case} \\
    \mu &= \int x p(x) dx \big\} \text{continuous case}
\end{align}


**Variance**:

\begin{align}
    \text{Var}(X) &= E \big[ (X - \mu)^{2} \big] \\
    \text{Var}(X) &= \sum_{i = 1}^{n} (x_{i} - \mu)^{2} p(x_{i})
        \big\} \text{discrete case} \\
    \text{Var}(X) &= \int (x - \mu)^{2} p(x_{i}) dx
        \big\} \text{continuous case}
\end{align}


**Standard Deviation**:

\begin{equation}
    \sigma^{2} = \text{Var}(X)
\end{equation}


**Covariance**:

\begin{align}
    \text{Cov}(X_{i}, X_{j}) 
        &= E \big[ (X_{i} - \mu_{i}) (X_{j} - \mu_{j}) \big] \\
        &= E \big[ (X_{i} X_{j}) \big] - \mu_{i} \mu_{j}
\end{align}

- If $Cov(X, Y) > 0$, when $X$ is above its expected value, then $Y$ tends to
  be above its expected value
- If $Cov(X, Y) < 0$, when $X$ is above its expected value, then $Y$ tends to
  be below its expected value
- $Cov(X, Y) = 0$: $X, Y$ are independent, 



## Gaussian Function

**Single Variate Gausian Function**:

Mean $\mu$ and variance $\sigma^{2}$

\begin{equation}
    p(x) = (2 \pi \sigma^{2})^{-0.5}
        \exp
            \big\{
                -\frac{1}{2} \frac{(x - \mu)^{2}}{\sigma^{2}}
            \big\}
\end{equation}


**Multi-Variate Normal Distribution Gaussian Function**:

Mean vector $\mu$ and $\Sigma$ is a positive semidefinite symmetric matrix
called a covariance matrix.

\begin{equation}
    p(x) =
        \text{det}(2 \pi \Sigma)^{-0.5}
        \exp
            \big\{
                -\frac{1}{2} (x - \mu)^{T}
                \Sigma^{-2} (x - \mu)^{T}
            \big\}
\end{equation}






## Bayes Rule

\begin{equation}
    p(A | B) = \frac{p(B | A) p(A)}{p(B)}
\end{equation}

where:

- $A$ and $B$ are events
- $p(A)$ and $p(B)$ are probabilities of $A$ and $B$ without regard to each
  other
- $p(A | B)$ is a conditional probability, it is the probability of observing
  event $A$ given that $B$ is true
- $p(B | A)$ is the probability of observing event $B$ given that $A$ is true

# Linear Least Sqaures

## Simple Explanation
The method of least squares is a standard approach in regression analysis to
the approximate solution of overdetermined systems, i.e. set of equations in
which there are more equations than unknowns. Least squares means that the
overal solution minimizes the sum of the squares of the errors made in the
results of every single equation.

The most important application is in data fitting. The best fit in the
least-squares sense minimizes the sum of the squared residuals (aka errors),
a residual being the difference between an observed value and the fitted value
provided by a model. When the problem has substantial uncertainties in the
independent variable, then simple regression and least squares method have
problems; in such cases, the methodology required for fitting
errors-in-variables models maybe considered instead of that for least squares.

Least squares works if we have a problem in the form:

\begin{equation}
    Ax = b
\end{equation}

Where we wish to determine what $x$ is. If $A$ is a skinny matrix ($n > m$) the
problem is over-constrained (no solution exists). Instead, we can minimize the
square error between $Ax$ and $b$.

\begin{equation}
    \text{min}_{x} || \text{error} ||^{2}_{2}
\end{equation}

\begin{equation}
    \text{min}_{x} || Ax - b ||^{2}_{2}
\end{equation}

\begin{equation}
    \text{min}_{x} \left( (Ax - b)^{T} (Ax - b) \right)
\end{equation}

\begin{equation}
    \text{min}_{x} \left( x^{T} A^{T} A x - 2b^{T} A x + b^{T} b \right)
\end{equation}

The smallest error is when the derivative of the error is 0, and so:

\begin{equation}
    \frac{d}{dx} \left( x^{T} A^{T} A x - 2b^{T}Ax + b^{T}b \right) = 0
\end{equation}

\begin{equation}
    \left( 2x^{T} A^{T} A - 2b^{T}A \right) = 0
\end{equation}

\begin{equation}
    A^{T} A x = A^{T} b
\end{equation}

\begin{equation}
    x = (A^{T} A)^{-1} A^{T} b
\end{equation}

\begin{equation}
    x = A^{\dagger} b
\end{equation}

where $A^{\dagger}$ is the **Pseudo Inverse** $A^{\dagger} = (A^{T} A)^{-1}
A^{T}$


## Advanced Explanation
Ordinary least squares or linear least squares is a method for estimating a set
of parameters $x \in {\rm I\! R}^{d}$ in a linear regression model. Assume for
each input vector $b_{i} \in {\rm I\! R}^{d}$, $i \in \{ 1, ..., n \}$, we
observe a scalar response $a_{i} \in {\rm I\! R}$. Assume there is a linear
relationship of the form:

\begin{equation}
    a_{i} = b_{i}^{\top} x + \eta_{i}
\end{equation}

with an unknown vector $x \in {\rm I\! R}^{d}$ and zero-mean Gaussian noise
$\eta \approx N(0, \Sigma)$ with a diagonal covariance matrix of the form
$\Sigma = \omega^{2} I_{n}$. **Maximum likelihood estimation** of $x$ leads to
the ordinary least squares problem:

\begin{align}
    &\text{error} = (a_{i} - x^{\top} b_{i})^{2} \\
    &\min_{x} \sum_{i} \text{error} \\
    &\min_{x} \sum_{i} (a_{i} - x^{\top} b_{i})^{2} \\
    &\min_{x} \sum_{i} (a - Bx)^{\top} (a - Bx)
\end{align}

Linear least squares estimation was introduced by Lengendre (1805) and Gauss
(1795 / 1809). When asking for which noise distribution the optimal estimator
was the arithmetic mean, Gauss invented the normal distribution.


## Generalized Least Squares

For general $\Sigma$, we get the **generalized least squares** problem:

\begin{equation}
    \min_{x} (a - Bx)^{\top} \Sigma^{-1} (a - Bx)
\end{equation}

This is a quadratic cost function with positive definite $\Sigma^{-1}$. It has
the closed form solution:

\begin{align}
    \hat{x} &= \arg\min_{x} (a - Bx)^{\top} \Sigma^{-1} (a - Bx) \\
            &= (B^{\top} \Sigma^{-1} B)^{-1} B^{\top} \Sigma^{-1} a
\end{align}


## Weighted Least Squares

If there is no correlation among the observed variances, then the matrix
$\Sigma$ is diagonal. This case is referred to as **weighed least squares**: 

\begin{equation}
    \min_{x} \sum_{i} w_{i} (a - x^{\top} b_{i})^{2} \\
    \text{with} \quad w_{i} = \sigma_{i}^{-2}
\end{equation}

For the case of unknown matrix $\Sigma$, there exist iterative estimation
algorithms such as **fesible generalized least squares** or **iteratively
reweighted least squares**.

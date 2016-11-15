# Newtons Method

Newton's method (aka Netwon-Raphson method), named after Isssac Netwon and
Joseph Raphson, is a method for finding successively better approximations to
the roots of a real valued function.

Newton methods are **second order methods**. In contrast to first-order methods
like gradient descent, they also make use of second derivatives. Geometrically,
Newton method iteratively approximates the cost function $E(x)$ quadratically
and takes a step to the minimizer of this approximation.

Let $x_{t}$ be the estimated solution after $t$ iterations. Then then Taylor
approximation of $E(x)$ in the vicinity of this estimate is:

\begin{equation}
    E(x) \approx
        E(x_{t}) + g^{\top} (x - x_{t}) +
            \dfrac{1}{2} (x - x_{t})^{\top} H (x - x_{t})
\end{equation}

The frist and second derivative are denoted by the **Jacobian** $g =
dE / dx(x_{t})$ and the **Hessian matrix**
$d^{2}E / dx^{2}(x_{t})$. For this second order approximation, the
optimality condition is:

\begin{equation}
    \dfrac{dE}{dx} = g + H(x - x_{t}) = 0
\end{equation}

Setting the next iterate to the minimizer $x$ leads to:

\begin{equation}
    x_{t + 1} = x_{t} - H^{-1} g
\end{equation}

In practice, one often choses a more conservative step size $\gamma \in (0,
1)$:

\begin{equation}
    x_{t + 1} = x_{t} - \gamma H^{-1} g
\end{equation}

When applicable, second-order methods are often faster than first-order
methods, at least when measured in number of iterations. In particular, there
exists a local neighborhood around each optimum where the Newton method
converges quadratically for $\gamma = 1$.

For large optimization problems, computing and inverting the Hessian may be
challenging. Moreover, since this problem is often not parallelizable, some
second order methods do not profit from GPU acceleration. In such cases, one
can aim to iteratively solve the extrememality condition.

In case that $H$ is not positive definite, there exist **quasi-Newton methods**
which aim at approximating $H$ or $H^{-1}$ with a positive definite matrix.

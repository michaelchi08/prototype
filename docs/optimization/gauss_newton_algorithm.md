# Gauss Newton Algorithm

The Gauss-Newton Algorithm is a modification of Newton's method for finding a
minimium of a function. Unlike Newton's method howevere, the Gauss-Newton
algorithm can only be used to minimize a sum of squared function values, but it
has the advantage that second derivatives, which can be challenging to compute
are not required.

It can be derived as an **approximation to the Newton's method**:

\begin{equation}
    x_{t + 1} = x_{t} - H^{-1} g
\end{equation}

with the gradient $g$:

\begin{equation}
    g_{j} = 2 \sum_{i} r_{i} \dfrac{\partial r_{i}}{\partial x_{j}}
\end{equation}

and the Hessian $H$:

\begin{equation}
    H_{jk} = 2 \sum_{i} \left(
        \dfrac{\partial r_{i}}{\partial x_{j}}
        \dfrac{\partial r_{i}}{\partial x_{k}} +
        r_{i} \dfrac{\partial^{2} r_{i}}{\partial x_{j} \partial x_{k}}
    \right)
\end{equation}

Dropping the second order term leads to the **approximation**:

\begin{align}
    H_{jk} \approx 2 \sum_{i} J_{ij} J_{ik} \\
    \text{with} \quad J_{ij} = \dfrac{\partial r_{i}}{\partial x_{j}}
\end{align}

The approximation

\begin{align}
    H \approx 2 J^{\top} J \\
    \text{with} \quad J = \dfrac{dr}{dx}
\end{align}

together with $g = 2 J^{\top} r$, leads to the **Gauss-Newton Algorithm**:

\begin{align}
    x_{t + 1} = x_{t} + \Delta \\
    \text{with} \quad \Delta = -(J^{\top} J)^{-1} J^{\top} r
\end{align}

This approximation of the Hessian is valid only if

\begin{equation}
    \left|
        r_{i} \dfrac{\partial^{2} r_{i}}{\partial x_{j} \partial x_{k}}
    \right|
    \ll
    \left|
        \dfrac{\partial r_{i}}{\partial x_{j}}
        \dfrac{\partial r_{i}}{\partial x_{k}}
    \right|
\end{equation}

This is the case if the **residuum $r_{i}$ is small** or if it is **close to
linear** (in which case the second derivatives are small).


# Gradient Descent

Gradient descent or steepest descent is a first-order optimization method, it
aims at computing a local minimum of a generally non-convex cost function by
iteratively stepping in the direction in which the energy decreases the most.
This is given by the negative gradient.

To minimize a real-valued cost $E : {\rm I\!R}^{n} \rightarrow {\rm I\!R}$, the
gradient flow for $E(x)$ is defined by the differential equation:

\begin{align}
    x(0) &= x_{0} \\
    \dfrac{dx}{dt} &= -\dfrac{dE}{dx}(x)
\end{align}

Or in discrete form:

\begin{align}
    x_{k + 1} &= x_{k} - \epsilon \dfrac{dE}{dx}(x_{k}) \\
    k &= 0, 1, 2, ...
\end{align}

The gradient descent method in 1 variable is implemented as follows. The method
starts witha  function $f$ defined over the real numbers $x$, the function's
derivative $f^{\prime}$, and an initial guess $x_{0}$ for a root of the
function $f$. If the function satisfies the assumptions made in the derivation
of the formula and the initial guess is close, then a better approximation
$x_{1}$ is:

\begin{equation}
    x_{1} = x_{0} - \dfrac{f(x_{0})}{f^{\prime}(x_{0})}
\end{equation}

Geometrically, $(x_{1}, 0)$ is the intersection fo the $x$-axis and the tangent
of the graph of $f$ at $(x_{0}, f(x_{0}))$. The process is repeated as:

\begin{equation}
    x_{n + 1} = x_{n} - \dfrac{f(x_{n})}{f^{\prime}(x_{n})}
\end{equation}

until a sufficiently accurate value is reached.

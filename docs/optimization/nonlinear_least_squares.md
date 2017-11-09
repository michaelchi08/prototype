# Nonlinear Least Squares

Nonlinear Least Squares aims at fitting observations $(a_{i}, b_{i})$ with a
nonlienar model of the form $a_{i} \approx f(b_{i}, x)$ for some function $f$
parameterized with an unknown vector $x \in {\rm I\!R}^{d}$. Minimizing the sum
of squares error:

\begin{equation}
    \min_{x} \sum_{i} r_{i} (x)^{2} \\
    \text{with} \quad r_{i} = a_{i} - f(b_{i}, x)
\end{equation}

is generally a **non-convex optimization problem**.

The optimality condition is given by:

\begin{equation}
    \sum_{i} r_{i} \dfrac{\delta r_{i}}{\delta x_{j}} = 0 \\
    \forall j \in \{1, ..., d\}
\end{equation}

Typically one cannot directly solve these euqation. Yet, there exist iterative
algorithms for computing approximate solutions, including Newton methods, the
Gauss-Newton algorithm and the Levenberg-Marquardt algorithm.

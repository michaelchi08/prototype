# Levenberg-Marquardt Algorithm

The Levenberg-Marquardt Algorithm (LMA) also knonw as the damped least squares
(DLS) method, is used to solve non-linear least square problems. LMA
interpolates between the Gauss-Newton algorithm (GNA) and the method of
gradient descent. The LMA is more robust than the GNA, which means that in many
cases it finds a solution even if it starts very far off the final minimum. For
well behaved functions and reasonable starting parameters, the LMA tends to be
a bit slower than GNA. LMA can also be viewed as Gauss-Newton using a trust
region appraoch.

The algorithm was first published in 1944 by Kenneth Levenberg, while working
at the Framford Army Arsenal. It was rediscovered bin 1963 by Donald Marquardt
who worked as a statistician at DuPont and independently ba Girard, Wynne and
Morrison.

The Newton algorithm:

\begin{equation}
    x_{t + 1} = x_{t} - H^{-1} g
\end{equation}

can be modified (damped):

\begin{equation}
    x_{t + 1} = x_{t} - (H^{-1} + \lambda I_{n})^{-1} g
\end{equation}

to create a hybrid between the Newton method ($\lambda = 0$) and a gradient
descent with step size $1 / \lambda$ (for $\lambda \rightarrow \infty$). In the
same manner, Levenberg (1944) suggested to damp the Gauss-Newton algorithm for
nonlinear least squares:

\begin{equation}
    x_{t + 1} = x_{t} + \Delta \\
        \quad \text{with} \quad
            \Delta = -(J^{\top} J + \lambda I_{n})^{-1} J^{\top} r
\end{equation}

Marquardt (1963) suggested a more adaptive component-wise damping of the form:

\begin{equation}
        \Delta = -(
            J^{\top} J +
            \lambda \text{diag}{(J^{\top} J)}
        )^{-1} J^{\top} r
\end{equation}

which avoids slow convergence in the directions of small gradients.

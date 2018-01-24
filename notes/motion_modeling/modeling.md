# Motion Modeling

A motion model seeks to describe how system motion can occur given inputs at
time $T$, what will the system do? Define a set of constraints between states
and inputs, define unknown disturbances as distributions.





## Discrete Time Motion Model

\begin{equation}
    x_{t} = f(x_{t - 1}, u_{t}, \epsilon_{t})
\end{equation}

where:
- $x_{t}$ is the state vector at time $t$
- $u_{t}$ is the input vector at time $t$
- $\epsilon_{t}$ is the disturbance to the system at time $t$
- $f(x_{t - 1}, u_{t}, \epsilon_{t})$ is the motion model






## Types of Motion Models

**Linear models** with additive Gaussian disturbances

\begin{equation}
    x_{t} = Ax_{t - 1} + Bu_{t} + \epsilon_{t}
\end{equation}


**Nonlinear models** with additive Gaussian disturbances

\begin{equation}
    x_{t} = f(x_{t - 1} + u_{t}) + \epsilon_{t}
\end{equation}


**Nonlinear models** with nonlinear disturbances

\begin{equation}
    x_{t} = f(x_{t - 1} + u_{t}, \epsilon_{t})
\end{equation}


**Probabilistic model**

\begin{equation}
    p(x_{t} \mid x_{t - 1}, u_{t})
\end{equation}


**Continuous time model**

\begin{equation}
    \dot{x} = \bar{f}(x, u, \epsilon)
\end{equation}





## Holonomic vs Non-Holonomic Constraints

A system is **holonomic** if all constraints of the system are holonomic. For
a constraint to be holonomic it must be expressible as a function of the form:

\begin{equation}
    f(x_{1}, x_{2}, \dots, x_{N}, t) = 0
\end{equation}

in other words a **holonomic constraint** depends only on the coordinates
$x_{j}$ and time $t$ (states that define the degrees of freedom). It does not
depend on the velocities. 

A constraint that cannot be expressed in the form shown above is considered
a **non-holonomic constraint**. A **Non-holonomic constraint** is a system that
depends on all states of the system.

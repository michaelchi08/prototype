# Potential Fields

Creates and uses gradient information to guide the agent from origin to
goal state. The potential field attractors and repellers are generally
quadratic but it can be anything, it has a general form of:



## Potential Field Attractor

\begin{equation}
    V_{att}(q) = K_{att} \rho(q, q^{g})^{2}
\end{equation}



## Potential Field Repeller

**Strength based on shortest distance to obstacle $O^{i}$**

\begin{equation}
    V_{rep}(q) = K_{rep} \sum_{i = 1}^{n}
        \dfrac{1}{\rho(q, O^{i})^{2}}
\end{equation}

Often a maximum distance of influence is included

\begin{equation}
    V_{rep}(q) = K_{rep} \sum_{i = 1}^{n}
        \begin{cases}
            \left( \dfrac{1}{\rho(q, O^{i})^{2}} \right)^{2} &
            \rho(q, O^{i}) \lt \bar{\rho} \\
            0 &
            \text{otherwise}
        \end{cases}
\end{equation}

**Distance to obstacle function**

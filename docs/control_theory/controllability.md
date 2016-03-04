# Controllability

A system is controllable if for any set of initial and final states $x(0)$ and
$x(T)$, there exists a control input sequence $u(0)$ to $u(T)$ to get from
$x(0)$ to $x(T)$.

This can be calculated by determining the rank of $A$ and $B$.

\begin{equation}
    \text{Rank} \left(
        \begin{bmatrix}
            B & AB & A^{2}B & \dots & A^{n-1}B
        \end{bmatrix}
    \right) = n
\end{equation}

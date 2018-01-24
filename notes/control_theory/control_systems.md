# Control Systems

**Robust Control System**: The first step in the design of a control system is
to obtain a mathematical model of the plant or control object. In reality, any
model of a plant we want to control will include an error in the modeling
process. That is, the actual plant difference from the model to be used in the
design of the control system.

To ensure the controller designed based on a model will work satisfactorily
when this controller is used with the actual plant, one reasonable approach is
to assume from the start that there is an uncertainty or error between the
actual plant and its mathematical model and include such uncertainty or error
in th edesign process. This approach is called a robust control system.

Suppose that:

- $\tilde{G}(s$ is the actual plant model with uncertainty $\Delta(s)$
- $G(s)$ is the nominal plant model to be used for designing and control system

$\tilde{G}(s)$ and $G(s)$ may be related by a multiplicative factor such as:

\begin{equation}
    \tilde{G}(s) = G(s)[1 + \Delta(s)]
\end{equation}

or an additive factor

\begin{equation}
    \tilde{G}(s) = G(s) + \Delta(s)
\end{equation}

or in other forms. Since hte exact description of the uncertainty or error
$\Delta(s)$ is unknown, we use an estimate $W(s)$, it is a scalar transfer
function where:

\begin{equation}
    \| \Delta(s)_{\infty} \| \lt \| W(s)_{\infty} \| =
        \max_{0 \leq \omega \leq \infty} | W(j \omega) |
\end{equation}

where $\| W(s)_{\infty} \|$ is the maximumm value of $| W(j \omega) |$ for $0
\leq \omega \leq \infty$ and is called the $H$ infinity norm of $W(s)$.

Using the small gain theorem, the design procedure boils down to the
determination of the controller $K(s)$ such that the inequality:

\begin{equation}
    \big\| \frac{W(s)}{1 + K(s) G(s)} \big\|_{\infty} \lt 1
\end{equation}

is satisfied, where $G(s)$ is the transfer function of the model used in the
design process, $K(s)$ is the transfer function of the controller and $W(s)$ is
the chosen transfer function to approximate $\Delta(s)$. In most practical
case, we must satisfy more than one such inequality that involves $G(s)$,
$K(s)$ and $W(s)$. For example to guarantee robust stability and robust
performance we may require two inequalities such as:

\begin{align}
    \big\| \frac{W_m(s) K(s) G(s) }{1 + K(s) G(s)} \big\|_{\infty} &\lt 1
        &\text{(for robust stability)} \\
    \big\| \frac{W(s)}{1 + K(s) G(s)} \big\|_{\infty} &\lt 1
        &\text{(for robust performance)}
\end{align}

be satisfied. There are many such inequalitistes that need to be satisfied in
many different robust control systems. Robust stability means the specified
performance is satisfied in all systms that belong to the group.

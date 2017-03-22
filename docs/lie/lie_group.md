# Lie Group

A Lie group is a group whose group elements are specified by one or more
continuous parameters which vary smoothly. Let us consider a simple example,
the SO(2) group of rotation in two dimensions. The group is characterized by
a single parameter $\theta$ the angle of rotation.

\begin{equation}
  R(\theta) = \left(
    \begin{array}{c c}
      \cos(\theta) & -\sin(\theta) \\
      \sin(\theta) & \cos(\theta)
    \end{array}
  \right)
\end{equation}

The Lie algebra is a mathematical structure that underlies the infinitesimal
group structure. In this example there is a matrix generator of the SO(2) Lie
algebra associated to the SO(2) group and which is given by:

\begin{equation}
  X = \left(
    \begin{array}{c c}
      0 & i \\
      -i & 0
    \end{array}
  \right)
\end{equation}

Such that the SO(2) Lie group matrix element, and the SO(2) Lie algebra
generator $X$ are related as:

\begin{equation}
  R(\theta) = e^{i \theta X} = \cos(\theta) I + i X \sin(\theta)
\end{equation}

One can verify the above equation after performing a Taylor series expansion of
the exponential and using $X^{2} = I$.

In general, exponentiating the generators $X_{1}, X_{2}, \dots, X_{n}$ of the
$n$-dim Lie algebra $\mathfrak{g}$ will recreate the representation of the
group $\mathfrak{G}$ which demonstrates that the generators contain all of the
information of the group structure. For SO(3), we have the following three
Hermitian matrix generators:

\begin{align}
  X_{1} &= \left(
    \begin{array}{c c c}
      0 & 0 & 0 \\
      0 & 0 & i \\
      0 & -i & 0
    \end{array}
  \right)  \\
  X_{2} &= \left(
    \begin{array}{c c c}
      0 & 0 & i \\
      0 & 0 & 0 \\
      -i & 0 & 0
    \end{array}
  \right)  \\
  X_{3} &= \left(
    \begin{array}{c c c}
      0 & i & 0 \\
      -i & 0 & 0 \\
      0 & 0 & 0
    \end{array}
  \right)
\end{align}

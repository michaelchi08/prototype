# Affine Transformation

\begin{equation}
  \begin{bmatrix}
    x' \\
    y' \\
    1
  \end{bmatrix} =
  \begin{bmatrix}
    a_{1} & a_{2} & t_{x} \\
    a_{3} & a_{4} & t_{y} \\
    0 & 0 & 1
  \end{bmatrix}
  \begin{bmatrix}
    x \\
    y \\
    1
  \end{bmatrix}
\end{equation}

or

\begin{equation}
  x' = \begin{bmatrix}
    A & t \\
    0 & 1 \\
  \end{bmatrix}
  x
\end{equation}

preserves $w = 1$ and cannot represent as strong deformations as a full
projective transformation. The affine transformation contains an invertible
matrix $A$ and a atransformation vector $t = [t_x, t_y]$. Affine
transofmrations are used for example in warping.

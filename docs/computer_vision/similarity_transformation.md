# Similarity Transformation

\begin{equation}
  \begin{bmatrix}
    x' \\
    y' \\
    1
  \end{bmatrix} =
  \begin{bmatrix}
    s \cos(\theta) & -s \sin(\theta) & t_x \\
    s \sin(\theta) & s \cos(\theta) & t_y \\
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
    sR & t \\
    0 & 1 \\
  \end{bmatrix}
  x
\end{equation}

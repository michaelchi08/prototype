# Linear Triangulation

Th linear triangulation method is the direct analogue of the DLT method. It
should be noted that the estimated point does not exactly satisfy the geometric
relations and is not an optimal estimate.

In each image we have a measurement:

\begin{align}
    x = P X \\
    x^{\prime} = P^{\prime} X
\end{align}

Where:

- $x$ and $x^{\prime}$ are the point correspondances (2D: x, y, 1) of the same
  feature in the first and second image respectively.
- $P$ and $P^{\prime}$ are camera matrices ($3 \times 4$) of the first and
  second image respectively.
- $X$ is the image feature in 3D coordinates.

The transformation from point correspondances $x$ and $x^{\prime}$ to 3D
coordinates can be written as:

\begin{equation}
    x \times (P H) = 0
\end{equation}

\begin{align}
    x (p^{3 \top} X) - (p^{1 \top} X) = 0 \\
    y (p^{3 \top} X) - (p^{2 \top} X) = 0 \\
    x (p^{2 \top} X) - y (p^{1 \top} X) = 0
\end{align}

Formulating it of the form $A X = 0$ where $A$ is

\begin{align}
    A = \begin{bmatrix}
        x p^{3 \top} - p^{1 \top} \\
        y p^{3 \top} - p^{2 \top} \\
        x^{\prime} p^{\prime 3 \top} - p^{\prime 1 \top} \\
        y^{\prime} p^{\prime 3 \top} - p^{\prime 2 \top}
    \end{bmatrix}
\end{align}


## Homogeneous Method (DLT)

The homogeneous method uses SVD to find the solution as the unit singular
vector corresponding to the smallest singular value of A.

    #!/usr/bin/octave
    [U, S, V] = svd(A);
    X = V(:, end);  // last column of V
    X = X / X(4);  // normalize 3D point


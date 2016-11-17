# Visual Odometry

## Feature Detection

## Optical Flow

## Bundle Adjustment

Let $x_{1}$ **(in homogeneous coordinates)** be the projection of a 3D point
$X$.  Given known camera parameters $(K = 1)$ and no rotation or translation of
the first camera, we merely have a projection with unknown depth $\lambda_{1}$.
From the first to the second frame we additionally have a camera rotation $R$
and translation $T$ followed by a projection. This gives the equations:

\begin{equation}
    \lambda_{1} x_{1} = X
\end{equation}

\begin{equation}
    \lambda_{2} x_{2} = RX + T
\end{equation}

Inserting the first equation into the second, we get:

\begin{equation}
    \lambda_{2} x_{2} = R(\lambda_{1}, x_{1}) + T
\end{equation}

Now we remove the translation by multiplying with $\hat{T}$ where $(\hat{T} v =
T \times v)$:

\begin{equation}
    \lambda_{2} \hat{T} x_{2} = \lambda_{1} \hat{T} R x_{1}
\end{equation}

And projection onto $x_{2}$ gives the **Epipolar Constraint**:

\begin{align}
    x_{2}^{\top} \hat{T} R x_{1} &= 0 \\
    x_{2}^{\top} E x_{1} &= 0
\end{align}

Where $E = \hat{T} R \in {\rm I\! R}^{3 \times 3}$ is the Essential Matrix.

Given that we have observed points $x_1$ and $x_2$, we can find the
corresponding estimated points $\tilde{x_{2}}$ and $\tilde{x_{1}}$.

\begin{align}
    \tilde{x_{2}} = R(x_{1}) + T \\
    \tilde{x_{1}} = R(x_{2} - T)^{-1}
\end{align}

and then compare the euclidean distance between observed and estimated

\begin{align}
    | x_{1} - \tilde{x_1} |^{2} \\
    | x_{2} - \tilde{x_2} |^{2}
\end{align}

The cost function for Bundle Adjustment becomes:

\begin{equation}
    | x_{1} - \tilde{x_1} |^{2} + | x_{2} - \tilde{x_2} |^{2}
\end{equation}

# Homography

A homography is a 2D projective transformation that maps points in one
plane to anther. In our case the planes are images or planar surfaces in
3D. Homographices have many practical uses such as registering images,
rectifying images, texture warping and creating panormamas. We will make
frequent use of them. In essence a homography $H$ maps 2D points (in
homogeneous coordinates) according to

\begin{equation}
  \begin{bmatrix}
    x' \\
    y' \\
    z'
  \end{bmatrix} =
  \begin{bmatrix}
    h_{1} & h_{2} & h_{3} \\
    h_{4} & h_{5} & h_{6} \\
    h_{7} & h_{8} & h_{9}
  \end{bmatrix}
\end{equation}

or

\begin{equation}
  x' = H x
\end{equation}

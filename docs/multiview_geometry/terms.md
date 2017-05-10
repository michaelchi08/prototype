# Difference between Essential and Fundamental Matrix

Both matrices relate corresponding points in two images. The difference is
that in the case of the Fundamental matrix, the points are in pixel
coordinates, while in the case of the Essential matrix the points are in
normalized homogeneous image coordinates. The normalized image coordinates
have the origin at the optical center of the image, and the x and
y coordinates are normalized by focal lengths $f_x$ and $f_y$
respectively.

The two matrices are related as follows:

\begin{equation}
  E = K^{\top} F K
\end{equation}

Where $K$ is the intrinsic matrix of the camera.

Further differences include $F$ has 7 degrees of freedom, while $E$
has 5 degrees of freedom, because it takes the camera parameters
into account. That's why there is an 8-point algorithm for computing
the fundamental matrix and a 5-point algorithm for computing the
essential matrix.

One way to get a 3D position from a pair of matching points from two
images is to take the fundamental matrix, compute the essential
matrix, and then to get the rotation and translation between the
cameras from the essential matrix. This, of course, assumes that you
know the intrinsics of your camera. Also, this would give you
up-to-scale reconstruction, what the translation being a unit vector.

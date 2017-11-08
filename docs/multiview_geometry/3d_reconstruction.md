# 3D Reconstruction

## Origins

The goal to reconstruct the 3D structure of the world from a set of 2D views
has a long history in computer vision. It is classical ill-posed problem
because the reconstruction consistent with a geven set of observations / images
is still typically not unique. Therefore, one will need to impose additional
assumptions. Mathematically, the study of geometric relations between a 3D
scene and the observed 2D projections is based on two types of transformations,
namely:

- Euclidean motion or rigid body motion representing the motion of the camera
  from one frame to the next.

- Perspective projection to account for the image formation process (see
  pinhole camera, etc).

The first work on the problem of multiple view geometry was that of Erwin
Kruppa (1913) who showed that two views of five points are sufficient to
determine both the relative transformation (motion) between the two views and
the 3D location (structure) of the points up to finitely many solutions.

A linear algorithm to recover structure and motion from two views based on the
epipolar constraint was proposed by Longuet-Higgins in 1981. An entire series
of works along these lines was summarized in several text books (Faugeras 1993,
Kanatani 1993, Maybank 1993, Weng et al. 1993).

Extensions to three views were developed by Spetsakis and Aloimonos (1987,
1990) and  by Shshua (1994) and Hartley (1995). Factorization techniques for
multiple views and orthogonal projection were developed by Tomasi and Kanade
1992.

The joint estimation of camera motion and 3D location is called structure and
motion of visual SLAM.

# Bundle Adjustment And Nonlinear Optimization

In the eight-point algorithm a linear approach was used to solve the structure
and motion problem. In particular, the eight-point algorithm provides a closed
form solutions to estimate the camera parameters and the 3D structure, based on
SVD.

However, if we have noisy data $x_{1}$ and $x_{2}$ where the correspondences
are not exact or correct then we have **no guarantee** that:

1. $R$ and $T$ are as close as possible to the true solution
2. Consistent recontruction can be obtained

In order to take noise and statistical fluctuation into account, one can revert
to a Bayesian formulation and determine the most likely camera transformation
$R$, $T$ and 'true' 2D coordinates $x$ given the measured coordinates
$\tilde{x}$, by performing a **maximum aposteriori estimate**:

\begin{equation}
    \arg\max_{x, R, T} P(x, R, T \mid \tilde{x}) =
        \arg\max_{x, R, T} P(\tilde{x} \mid x, R, T) P (x, R, T)
\end{equation}

This approach will however involve modeling probability densities $P$ on the
fairly compicated space $SO(3) \times \mathbb{S}^{3}$ of rotation and
translation parameters, as $R \in SO(3)$ and $T \in \mathbb{S}^{2}$ (3D
translation with unit length).



## What is Bundle Adjustment
Given a set of images depicting a number of 3D points from different
viewpoints, bundle adjustment can be defined as the problem of simultaneously
refining the 3D coordinates describing the scene geometry, the parameters of
the relative motion, and optical characteristics of the cameras employed to
acquire the images according to an optimality criterion involing the
corresponding image projections of all points.

Bundle adjustment is almost used as the last step of every feature-based 3D
reconstruction algorithm. It amounts to an optimization problem on the 3D
structure and viewing parameters (i.e. camera pose and possibly intrinsic
calibration and radial distorition), to obtain a reconstruction which is
optimal under certain assumptions regarding the noise pertaining to the
observed image features: if the image error is zero-mean Gaussian, then bundle
adjustment is the Maximum Likelihood Estimator (MLE).

The name bundle adjustment refers to the the bundles of light rays originating
from each 3D feature and converging on each camera's optical center, which are
adjusted optimally with respect to both the structure and viewing parameters
(similarity in meaning to categorical bundle seems purely coincidental).

Bundle adjustment boils down to minimizing the reprojection error between the
image locations of observed and predicted image points, which is expressed as
the sum of squares of a large number of nonlinear, real-valued functions. The
minimization is achieved using nonlinear least-squares algorithms. Of these,
Levenberg-Marquardt is commonly used due to its ease of implementation and its
use of an effective damping strategy that lends it the ability to converge
quickly from a wide range of initial guesses.

Under the assumption that the observed 2D point coordinates $\tilde{x}$ are
corrupted by zero-mean Gaussian noise, maximum likelihood estimation leads to:

\begin{equation}
    E(R, T, X_{1}, \dots, X_{N}) =
        \sum_{j = 1}^{N}
            |\tilde{x}_{1}^{j} - \pi(X_{j})|^{2} +
                |\tilde{x}_{2}^{j} - \pi(R, T, X_{j})|^{2}
\end{equation}

It aims at minimizing the **reprojection error** between the observed 2D
coordinates $\tilde{x}_{i}^{j}$ and the projected 3D coordinate $X_{j}$ (w.r.t
camera 1). Here $\pi(R, T, X_{j})$ denotes the perspective projection of
$X_{j}$ after rotation and translation.

For the general case of $m$ images, we get:

\begin{equation}
    E(\{R_{i}, T_{i}\}_{i = 1..m}, \{X_{j}\}_{j = 1..N}) =
        \sum_{i = 1}^{m} \sum_{j = 1}^{N}
            \theta_{ij} |\tilde{x}_{i}^{j} - \pi(R_{i}, T_{i}, X_{j})|^{2}
\end{equation}

with $T_{1} = 0$ and $R_{1} = 1$. $\theta_{ij} = 1$ if point $j$ is visible in
image $i$, else $\theta_{ij} = 0$. The above problem are non-convex.

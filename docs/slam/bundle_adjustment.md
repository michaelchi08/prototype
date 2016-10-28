# Bundle Adjustment

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

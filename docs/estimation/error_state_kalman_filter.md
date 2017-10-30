# Error-State Kalman Filter

There are two main Kalman Filter (KF) formulations, the direct (standard)
Kalman Filter and the indirect Kalman Filter. The indirect Kalman Filter
is also known as the error-state Kalman filter, where the formulation has
a notion of true, nominal and error state values, the true-state being
expressed as a suitable composition of the nominal and error states. The
idea is to consider the nominal state as a large-signal (integrable in
a non-linear fashion) and the error-state as a small signal (thus linearly
integrable and suitable for linear-Guassian filtering).

The error-state filter can be explained as follows. On one side,
high-frequency IMU data $\mathbf{u_m}$ is integrated into the
nominal-state $\mathbf{x}$. This nominal state does not take into account
the noise terms $\mathbf{w}$ and other possible model impoerfections. As
a consequence, it will accumulate errors. These errors are collected in
the error-state $\delta \mathbf{x}$ and estimated with the Error-State
Kalman Filter (ESKF), this time incorporating all the noise and
perturbations.

The error state consists of small signal magnitudes, and its evolution
function is correctly defined by a (time-variant) linear dynamic system,
with its dynamic, control and measurment matrices computed from the values
of the nominal-state. In parallel with integration of the nominal state,
the ESKF predicts a Gaussian estimate of the error-state. It only
predicts, because by now no other measurement is available to correct
these estimates. The filter correction is performed at the arrival of
information other than the IMU (e.g. GPS, vision, etc), which is able to
render the errors observable and which happens generally at much lower
rate than the intergration pahse. This correction provides a posterior
Gaussian estimate of the error-state. After this, the error-state's mean
is injected into the nominal-state, then reset to zero. The error-state's
covariance matrix is conveniently updated to reflect this reset. The
system goes on like this forever.

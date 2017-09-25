# Gimbl Lock

Gimbal lock is the loss of one degree of freedom in a three dimensional,
three gimbal mechanism that occurs when the axes of two of the three
gimbals are driven into a parallel configuration, "locking" the system
into rotation in degenerate two-dimensional space.

The word lock is misleading: no gimbal is restrained. All three gimbals can
still rotate freely about their respective axes of suspension. Nevertheless,
because of the parallel orientation of two of the gimbal axes there is no
gimbal available to accomodate rotation along one axis.

The problem of gimbal lock appears when one uses Euler angles in applied
mathematics; developers of 3D computer programs, such as 3D modeling, embedded
navigation systems, and video games must take care to avoid it.

In formal language, gimbal lock occurs because the map from Euler angles to
rotations (topologically, from the 3-torus T3 to the real projective space RP3)
is not a covering map - it is not a local homeomorphism at every point, and
thus at some points the rank (degrees of freedom) must drop below 3, at which
point gimbal lock occurs. Euler angles provide a means for giving a numerical
description of any rotation in three-dimensional space using three numbers, but
not only is this description not unique, but there are some points where not
every change in the target space (rotations) can be realized by a change in the
source space (Euler angles). This is a topological constraint – there is no
covering map from the 3-torus to the 3-dimensional real projective space; the
only (non-trivial) covering map is from the 3-sphere, as in the use of
quaternions.


## Example

A rotation in 3D space can be represented numerically with matrices in several
ways. One of these representations is:

\begin{equation}
    R =
        \begin{bmatrix}
            1 & 0 & 0 \\
            0 & \cos  \alpha & -sin \alpha \\
            0 & \sin  \alpha & \cos \alpha
        \end{bmatrix}
        \begin{bmatrix}
            \cos \beta & 0 & \sin \beta \\
            0 & 1 & 0 \\
            -\sin \beta & 0 & \cos \beta
        \end{bmatrix}
        \begin{bmatrix}
            \cos \gamma & -\sin \gamma & 0 \\
            \sin \gamma & \cos \gamma & 0 \\
            0 & 0 & 1
        \end{bmatrix}
\end{equation}

Let $\beta = \frac{\pi}{2}$, the above becomes

\begin{align}
    R &=
        \begin{bmatrix}
            1 & 0 & 0 \\
            0 & \cos \alpha & -\sin \alpha \\
            0 & \sin \alpha & \cos \alpha
        \end{bmatrix}
        \begin{bmatrix}
            0 & 0 & 1 \\
            0 & 1 & 0 \\
            -1 & 0 & 0
        \end{bmatrix}
        \begin{bmatrix}
            \cos \gamma & -\sin \gamma & 0 \\
            \sin \gamma & \cos \gamma & 0 \\
            0 & 0 & 1
        \end{bmatrix} \\
    R &=
        \begin{bmatrix}
            0 & 0 & 1 \\
            \sin \alpha & \cos \alpha & 0 \\
            -\cos \alpha & \sin \alpha & 0
        \end{bmatrix}
        \begin{bmatrix}
            \cos \gamma & -\sin \gamma & 0 \\
            \sin \gamma & \cos \gamma & 0 \\
            0 & 0 & 1
        \end{bmatrix} \\
    R &=
        \begin{bmatrix}
            0 & 0 & 1 \\
            \sin \alpha \cos \gamma + \cos \alpha \sin \gamma &
            -\sin \alpha \sin \gamma + \cos \alpha \cos \gamma &
            0 \\
            -\cos \alpha \cos \gamma + \sin \alpha \sin \gamma &
            \cos \alpha \sin \gamma + \sin \alpha \cos \gamma &
            0
        \end{bmatrix} \\
    R &=
        \begin{bmatrix}
            0 & 0 & 1 \\
            \sin(\alpha + \gamma) & \cos(\alpha  + \gamma) & 0 \\
            -\cos(\alpha + \gamma) & \sin(\alpha  + \gamma) & 0
        \end{bmatrix} \\
\end{align}

Changing the values of $\alpha$ and $\gamma$ in the above matrix has the same
effects: the rotation angle $\alpha + \gamma$ changes, but the rotation axis
remains in the Z direction: the last column and the first row in the matrix
won't change. The only solution for $\alpha$  and $\gamma$ to recover different
roles is to change $\beta$.

It is possible to imagine an airplane rotated by the above-mentioned Euler
angles using the X-Y-Z convention. In this case, the first angle - $\alpha$ is
the pitch. Yaw is then set to $\frac{\pi}{2}$ and the final rotation - by
$\gamma$ - is again the airplane's pitch. Because of gimbal lock, it has lost
one of the degrees of freedom - in this case the ability to roll.

It is also possible to choose another convention for representing a rotation
with a matrix using Euler angles than the X-Y-Z convention above, and also
choose other variation intervals for the angles, but in the end there is always
at least one value for which a degree of freedom is lost.

The gimbal lock problem does not make Euler angles "invalid" (they always serve
as a well-defined coordinate system), but it makes them unsuited for some
practical applications.


## Alternatve orientation representation

The cause of gimbal lock is representing an orientation as 3 axial rotations
with Euler angles. A potential solution therefore is to represent the
orientation in some other way. This could be as a rotation matrix, a quaternion
(see quaternions and spatial rotation), or a similar orientation representation
that treats the orientation as a value rather than 3 separate and related
values. Given such a representation, the user stores the orientation as a
value. To apply angular changes, the orientation is modified by a delta
angle/axis rotation. The resulting orientation must be re-normalized to prevent
floating-point error from successive transformations from accumulating. For
matrices, re-normalizing the result requires converting the matrix into its
nearest orthonormal representation. For quaternions, re-normalization requires
performing quaternion normalization.

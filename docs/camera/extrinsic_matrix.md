# Extrinsic Matrix

The camera's extrinsic matrix describes the camera's location in the world, and
what direction it is pointing. It has two components: a rotation matrix $R$ and
a translation vector $t$, but as we will soon see, these don't exactly
correspond to the camera's rotation and translation.

First we will examine parts of the extrinsic matrix and later we will look for
alternative ways of describing the camera's pose more intuitively.

The extrinsic matrix takes the form of a rigid transformation matrix: a $3
\times 3$ rotation matrix in the left-block, and a $3 \times 1$ translation
column-vector on the right:

\begin{equation}
    [ R_{I}^{B} \mid t_{I}^{B} ] =
        \left[
            \begin{array}{c c c | c}
                r_{1, 1} & r_{1, 2} & r_{1, 3} & t_{1} \\
                r_{2, 1} & r_{2, 2} & r_{2, 3} & t_{2} \\
                r_{3, 1} & r_{3, 2} & r_{3, 3} & t_{3}
            \end{array}
        \right]
\end{equation}

It is common to see a version of the matrix with extra row of $(0, 0, 0, 1)$
added to the bottom, this makes the matrix square in turn allows us to further
decompose this matrix into a rotation followed by translation:

\begin{align}
    \left[
        \begin{array}{c | c}
            R_{I}^{B} & t_{I}^{B} \\
            \hline
            0 & 1
        \end{array}
    \right]
        &=
        \left[
            \begin{array}{c | c}
                I & t_{I}^{B} \\
                \hline
                0 & 1
            \end{array}
        \right]
        \left[
            \begin{array}{c | c}
                R_{I}^{B} & 0 \\
                \hline
                0 & 1
            \end{array}
        \right] \\
        &= \left[
            \begin{array}{c c c | c}
                1 & 0 & 0 & t_{1} \\
                0 & 1 & 0 & t_{2} \\
                0 & 0 & 1 & t_{3} \\
                \hline
                0 & 0 & 0 & 1
            \end{array}
        \right]
        \left[
            \begin{array}{c c c | c}
                r_{1, 1} & r_{1, 2} & r_{1, 3} & 0 \\
                r_{2, 1} & r_{2, 2} & r_{2, 3} & 0 \\
                r_{3, 1} & r_{3, 2} & r_{3, 3} & 0 \\
                \hline
                0 & 0 & 0 & 1
            \end{array}
        \right]
\end{align}

This matrix describes how to transform points in the world coordinates to
camera coordinates. The vector $t_{I}^{B}$ can be interpreted as the position of
the world origin in camera coordinates, and the columns of $R_{I}^{B}$
represent the directions of the world-axes in camera coordinates.

The important thing to remember about the extrinsic matrix is that **the
extrinsic matrix describes how the world is transformed relative to the
camera**.  This is often counter-intuitive, because we often want to specify how
the camera is transformed relative to the world.

Next we will examine two alternative ways to describe the camera extrinsic
parameters that are more intuitive and how to convert them into the form of an
extrinsic matrix.


### Building the Extrinsic Matrix from Camera Pose

It is often more natural to specify the camera's pose directly rather than
specifying how world points should transform to camera coordinates. Luckily,
building an extrinsic camera matrix this way is easy: just build a rigid
transformation matrix that describes the camera's pose and then take its
inverse.

Let $C$ be a column vector describing the location of the camera-center in
world coordinates, and let $R_{B}^{I}$ be the rotation matrix describing the
camera's orientation with respect to the world coordinate axes. The
transformation matrix that describes the camera's pose is then $[R_{B}^{I} \mid
C]$. Like before, we make the matrix square by adding an extra row of $(0, 0,
0, 1)$. Then the extrinsic matrix is obtained by inverting the camera's pose
matrix:

\begin{align}
    \left[
        \begin{array}{c | c}
            R_{I}^{B} & t_{I}^{B} \\
            \hline
            0 & 1
        \end{array}
    \right]
        &=
        \left[
            \begin{array}{c | c}
                R_{B}^{I} & C \\
                \hline
                0 & 1
            \end{array}
        \right]^{-1} \\
        &=
        \left[
            \left[
                \begin{array}{c | c}
                    I & C \\
                    \hline
                    0 & 1
                \end{array}
            \right]
            \left[
                \begin{array}{c | c}
                    R_{B}^{I} & 0 \\
                    \hline
                    0 & 1
                \end{array}
            \right]
        \right]^{-1} \quad \text{(decomposing rigid transform)} \\
        &=
        \left[
            \begin{array}{c | c}
                R_{B}^{I} & 0 \\
                \hline
                0 & 1
            \end{array}
        \right]^{-1}
        \left[
            \begin{array}{c | c}
                I & C \\
                \hline
                0 & 1
            \end{array}
        \right]^{-1} \quad \text{(distributing the inverse)} \\
        &=
        \left[
            \begin{array}{c | c}
                (R_{B}^{I})^{\top} & 0 \\
                \hline
                0 & 1
            \end{array}
        \right]
        \left[
            \begin{array}{c | c}
                I & -C \\
                \hline
                0 & 1
            \end{array}
        \right] \quad \text{(applying the inverse)} \\
        &=
        \left[
            \begin{array}{c | c}
                (R_{B}^{I})^{\top} & -(R_{B}^{I})^{\top} C \\
                \hline
                0 & 1
            \end{array}
        \right] \quad \text{(matrix multiplication)} \\
\end{align}

When applying the inverse, we use the fact that the inverse of a rotation
matrix is its transpose, and inverting a translation matrix simply negates the
translation vector. Thus we see that the relationship between extrinsic
parameters and camera's pose is straight forward:

\begin{align}
    R_{I}^{B} &= (R_{B}^{I})^{\top} \\
    t_{I}^{B} &= -R_{I}^{B} C
\end{align}

Some texts write the extrinsic matrix substituting $-RC$ for $t$, which mixes a
world transform $R$ and camera transform notation $C$.


### The "Look-At" Camera

Readers familiar with OpenGL might prefer a third way of specifying the
camera's pose using

1. Camera's position
2. What it is looking at
3. Up direction

In legacy OpenGL, this is accomplished by the `gluLookAt()` function, so we'll
call this the "look-at" camera. Let $C$ be the camera center, $p$ be the
target point and $u$ be the "up"-direction. The algorithm for computing the
rotation matrix is:

1. Compute $L = p - C$
2. Normalize $L$
3. Compute $s = L \times u$ (cross product)
4. Normalize $s$
3. Compute $u^{\prime} = s \times L$ (cross product)

The extrinsic rotation matrix is then given by:

\begin{equation}
    R = \left[
        \begin{array}{c c c}
            s_{1} & s_{2} & s_{3} \\
            u_{1}^{\prime} & u_{2}^{\prime} & u_{3}^{\prime} \\
            -L_{1} & -L_{2} & -L_{3}
        \end{array}
    \right]
\end{equation}

See [this](http://www.opengl.org/sdk/docs/man2/xhtml/gluLookAt) for more
info.

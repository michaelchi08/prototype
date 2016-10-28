# Point Correspondence

In practice, in computer vision the images that we observe are not points or
lines, but rather brightness or colour values at the pixel level. In order to
transfer from this photometric representation to a geometric representation of
the scene, one can idenitify points with **characteristic image features** and
try to associate theses points with corresponding points in other frames.
Usually in the form of **features** in the image.

The matching of corresponding points will allow us to infer 3D structure.
Nevertheless one should keep in mind that this approach is **suboptimal**: By
selecting a small number of feature points from each image, we end up throwing
away a large amount of potentially useful information contained in each iage.
Yet retaining all image information is computationally challenging. The
selection and matching of a small number of feature points, on the other hand,
allows tracking of 3D objects from a moving camera in real time - even with
limited processing power.

Most feature based techniques assume that objects move rigidly. However, in
general, objects can and will deform non-rigidly, moreover there may be
**partial occlusions**.

In point matching one distinguishes two cases:

- **Small deformation**: The deformation from one frame to the other is assumed
  to be (infinitesimally) small. In this case the displaceent from one frome to
  the ohter can be estimated by classical **optical flow estimation**, for
  example using the methods of **Lucas/Kanade** or **Horn/Schunck**. In
  particular these methods allows one to model dense deformation fields (giving
  a displacement for every pixel in the image). But one can also track the
  displacement of a few feature points which is typically faster.

- **Wide baseline stereo**: In this case the displacement is assumed to be
  large. A dense matching of all points to all is generally computationally
  infeasible. Therefore, one typically selects a small number of feature points
  in each of the images and develops efficient methods to find an appropriate
  pairing of points.


## Small Deformation

The transformation of all points of a rigidly moving object is given by:

\begin{align}
    x_{2} &= h(x_{1}) \\
        &= \dfrac{1}{\lambda_{2}(X)}(R \lambda_{1}(X) x_{1} + T)
\end{align}

Locally this motion can be approximated in several ways

### Translation model

\begin{equation}
    h(x) = x + b
\end{equation}

### Affine model

\begin{equation}
    h(x) = Ax + b
\end{equation}

  The 2D Affine model can also be written as

\begin{equation}
    h(x) = x + u(x)
\end{equation}

  where

\begin{align}
    u(x) &= S(x) p \\
        &=
        \left(
            \begin{array}{cccccc}
                x & y & 1 & 0 & 0 & 0 \\
                0 & 0 & 0 & x & y & 1
            \end{array}
        \right)
        \left(
            \begin{array}{cccccc}
                p_{1} & p_{2} & p_{3} & p_{4} & p_{5} & p_{6}
            \end{array}
        \right)^{T}
\end{align}


## Wide Baseline Matching

In the case of **wide baseline matching**, large parts of the image plane will
not match at all bcause they are not visible in the other image. In other
words, while a given point may have many potential matches, quite possibly it
does not have a corresponding point in the other image.

One of the limitations of tracking features frame by frame is that small errors
in the motion accumulate over time and the window gradually moves away from teh
point that was originally tracked, this is know as **drift**. A remedy is to
match a given point back to the frist frame. This generally implies larger
displacements between frames.

Two aspects matter when extending the simple feature tracking methods to larger
displacements:

- Since the motion of the window between frames is (in general) no longer
  translational, one needs to generalize the motion model for the window
  $W(x)$, for example by using an affine motion model.

- Since the illumination will change over time (especially when comparing more
  distant frames), one can replace the sum of squared differences by the
  **normalized cross correlation** which is more robust to illumination changes.


### Normalized Cross Correlation

The **normalized cross correlation** is defined as:

\begin{equation}
    \text{NCC}(h) =
        \dfrac{
            \int_{W(x)}
            \left( I_{1}(x') - \bar{I}_{1} \right)
            \left( I_{2}(h(x')) - \bar{I}_{2} \right)
            dx'
        }{
            \sqrt{
                \int_{W(x)}
                \left(
                    I_{1}(x') - \bar{I}_{1}
                \right)^{2}
                dx'
                \int_{W(x)}
                \left(
                    I_{2}(h(x')) - \bar{I}_{2}
                \right)^{2}
                dx'
            }
        }
\end{equation}

where $\bar{I_{1}}$ and $\bar{I_{2}}$ are the average intensity over the window
$W(x)$. By subtracting this average intensity, the measure becomes invariant to
additive intensity changes $I \rightarrow I + \gamma$

Dividing by the intensity variances of each window makes the measure invariant
to multiplicative changes $I \rightarrow \gamma I$.

If we stack the normalized intensity values of respective windows into one
vector, $v_{i} = vec(I_i - \bar{I}_{i})$ then the normalized cross correlation
is the cosine of the angle between them:

\begin{equation}
    \text{NCC}(h) = \cos\angle(v_{1}, v_{2})
\end{equation}

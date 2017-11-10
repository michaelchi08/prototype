# Eigenvectors and Eigenvalues


## Short Answer

Eigenvectors make understanding linear transformations easy. They are the
"axes" (directions) along which a linear transformation acts simply by
"stretching/compressing" and/or "flipping"; eigenvalues give you the
factors by which this compression occurs.

The more directions you have along which you understand the behavior of a
linear transformation, the easier it is to understand the linear
transformation; so you want to have as many linearly independent
eigenvectors as possible associated to a single linear transformation.



## Slightly Longer Answer

There are a lot of problems that can be modeled with linear
transformations, and the eigenvectors give very simple solutions. For
example, consider the system of linear differential equations

\begin{align*}
  \frac{dx}{dt} &= ax + by \\\
  \frac{dy}{dt} &= cx + dy.
\end{align*}

This kind of system arises when you describe, for example, the growth of
population of two species that affect one another. For example, you might
have that species $x$ is a predator on species $y$; the more $x$ you have,
the fewer $y$ will be around to reproduce; but the fewer $y$ that are
around, the less food there is for $x$, so fewer $x$s will reproduce; but
then fewer $x$s are around so that takes pressure off $y$, which
increases; but then there is more food for $x$, so $x$ increases; and so
on and so forth. It also arises when you have certain physical phenomena,
such a particle on a moving fluid, where the velocity vector depends on
the position along the fluid.

Solving this system directly is complicated. But suppose that you could do
a change of variable so that instead of working with $x$ and $y$, you
could work with $z$ and $w$ (which depend linearly on $x$; that is,
$z=\alpha x+\beta y$ for some constants $\alpha$ and $\beta$, and
$w=\gamma x + \delta y$, for some constants $\gamma$ and $\delta$) and the
system transformed into something like

\begin{align*}
  \frac{dz}{dt} &= \kappa z\\\
  \frac{dw}{dt} &= \lambda w
\end{align*}

that is, you can "decouple" the system, so that now you are dealing with
two *independent* functions. Then solving this problem becomes rather
easy: $z=Ae^{\kappa t}$, and $w=Be^{\lambda t}$. Then you can use the
formulas for $z$ and $w$ to find expressions for $x$ and $y$.

Can this be done? Well, it amounts *precisely* to finding two linearly
independent eigenvectors for the matrix $\left(\begin{array}{cc}a & b\\c &
d\end{array}\right)$! $z$ and $w$ correspond to the eigenvectors, and
$\kappa$ and $\lambda$ to the eigenvalues.  By taking an expression that
"mixes" $x$ and $y$, and "decoupling it" into one that acts independently
on two different functions, the problem becomes a lot easier.

That is the essence of what one hopes to do with the eigenvectors and
eigenvalues: "decouple" the ways in which the linear transformation acts
into a number of independent actions along separate "directions", that can
be dealt with independently. A lot of problems come down to figuring out
these "lines of independent action", and understanding them can really
help you figure out what the matrix/linear transformation is "really"
doing.

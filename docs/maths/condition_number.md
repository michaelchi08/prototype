# Condition Number

In numerical analysis, the condition number of a function with respect to
an argument measures how much the output value of the function can change
for a small change in the input argument, one can also think of how
sensitive the function is to the changes or errors in the output. and how
much error int he output results from an error from the input.

Very frequently one is solving the inverse problem; given $f(x) = y$, one
is solving for $x$ and thus the condition number of the local inverse must
be used.

The condition number is an application of the derivative, and is formally
defined as the value of the asymptotic worst-case relative change in
output for a relative change in input. The function in question is the
solution of a problem and the arguments are the data in the problem. The
condition number is frequently applied to questions in linear algebra, in
which case the derivative is straightforward but the error could be in
many different directions, and is thus computed from the geometry of the
matrix. More generally, condition numbers can be defined for non-linear
functions in several variables.

A problem with a low condition number is said to be **well-conditioned**,
while a problem with a high condition number is said to be
**ill-conditioned**. Some algorithms have a property called **backward
stability**. In general, a backward stable algorithm can be expected to
accurately solve well-conditioned problems. Numerical analysis textbooks
give formulas for the condition numbers of problems and identify known
backward stable algorithms.



## Example: Condition Number for Matrices

For example, the condition number associated with the linear equation $Ax
= b$ gives a bound on how inaccurate the solution $x$ will be after
approximation. Note that this is before the effects of round-off error are
taken into account; conditioning is a property of the matrix, not the
algorithm or floating point accuracy of the computer used to solve the
corresponding system. In particular, one should think of the condition
number as being (very roughly) the rate at which the solution, $x$, will
change with respect to a change in $b$. Thus, if the condition number is
large, even a small error in $b$ may cause a large error in $x$. On the other
hand, if the condition number is small then the error in $x$ will not be
much bigger than the error in $b$.

The condition number is defined more precisely to be the maximum ration of
the relative error in $x$ divided by the relative error in $b$.

Let $e$ be the error in $b$. Assuming that $A$ is a nonsingular matrix,
the error in the solution $A^{-1} b$ is $A^{-1} e$. The ration of the
relative error in the solution to the relative error in $b$ is:

\begin{equation}
  \dfrac{|| A^{-1} e ||}{|| A^{-1} b ||} / dfrac{|| e ||}{|| b ||}
\end{equation}

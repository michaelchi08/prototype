# Constrained Optimization

Constrained optimization is the process of optimizing an objective function
with respect to some variables in the presence of constraints on those
variables. The objective function is either a cost function or an energy
function whihc is to minimized, or a reward function or utility function to be
maximized.

Constraints can either be **hard constraints** which set conditions for
the variables that are required to be satisfieda, or **soft constraints**
which have some variable values that are penalized in the objective
function if, and based on the extent that, the conditions on the variables
are not satisified.


## General Form

A general constrained minimization problem may be written as follows:

    min f(x)

    subject to:

        g(x) = c_i  for i .. n   (Equality constraints)
        h(x) >= d_j  for j .. m  (Inequality constraints)

where $g_{i}(x)$ and $h_{j}(x)$ are constraints that are required to be
satisifed, theses are called **hard constraints**.


### Example

    f(x) = x^{2} y

    s.t

        g(x) = x^{2} + y^{2} = 1

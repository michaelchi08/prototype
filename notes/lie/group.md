# Group

A mathematical group $G$ is a structure consisting of a finite or infinite set
of elements plus some binary operation (the group operation), which for any
two group elements $A, B \in G$ is denoted as the multiplication $AB$.

A group is said to be a group _under_ some given operation if it fulfills
the following conditions:

1. **Closure**: The group operation is a function $G \times G \mapsto G$,
   that is, for any $A, B \in G$ we have $AB \in G$.
2. **Associativity**: For $A, B, C \in G, (AB)C = A(BC)$.
3. **Identity element**: There must exists an identity element $I \in G$
   such that $IA = AI = A$ for any $A \in G$.
4. **Inverse**: For any $A \in G$ there must exist an inverse element
   $A^{-1}$ such as $AA^{-1} = A^{-1} A = I$.


## In Layman terms

A group is the set of symmetries of an object: the ways you can rearrange
something without damaging its structure. For instance, we can think about the
symmetry group of a square.

We can move the vertex 1 to any of the four points here.  Once we've decided
where to move it, we then have to move the vertex 2 to one of the two adjacent
positions.  The vertex 4 is then forced to go to the other adjacent position,
and the vertex 3 goes to the point diagonally opposite the point we chose for
one.  So the symmetry group of the square has 4 x 2 = 8 elements.

This is an example of a discrete group -- our symmetries are isolated in some
sense.  On the other hand, consider the symmetries of a circle.  We can rotate
the circle by 90 degrees, or by 45 degrees, or by 12 degrees, or by any amount
we want.  The symmetry group of the circle is an example of a continuous group
-- any symmetry of the circle has "nearby" symmetries, and we can think about
things like continuous paths through the symmetry group.

The symmetry group of a circle is a topological group -- a group which is also
a topological space.  Of course, if you know anything about topology, you know
that there are more structured versions of topological spaces -- differentiable
manifolds, for instance, which are essentially spaces that we can do calculus
on.  A Lie group (named after the mathematician Sophus Lie) is a group which is
a differentiable manifold.  The symmetry group of the circle is one example.

To summarize: in the case of the square, we had to rotate by 90 degrees at a
minimum, so we got a discrete symmetry group.  In the case of the circle, we
could rotate by an arbitrarily small amount, so we got a Lie group of
symmetries.

Why is this relevant?  Well, if you have a system of linear equations, the set
of symmetries (i.e., the coordinate changes you can make without altering the
equations) generally form a Lie group.  For instance, if you take Maxwell's
equations and calculate the symmetry group, you learn that they're not
invariant under Galilean changes of reference frame, but under things like
Lorentz boosts. Lorentz[1] discovered this property, and if people had taken
the implications seriously we would have discovered special relativity in the
19th century.


[1] And FitzGerald, and Heaviside, and several others.

# Lie Algebra

A Lie algebra is a vector space $\mathfrak{g}$ over some field $F$ together
with a binary operation $[\cdot, \cdot]: \mathfrak{g} \times \mathfrak{g}
\mapsto \mathfrak{g}$ called the Lie bracket that satisfies the following
axioms:

- **Bilinearity**
\begin{align}
  [ax + by, z] = a[x, z] + b[y, z] \\
  [x, ay + bz] = a[x, z] + b[x, z]
\end{align}
For all $a, b \in F$ and all elements $x, y, z, \in \mathfrak{g}$

- **Alternativity**
\begin{align}
  [x, x] = 0
\end{align}
For all $x \in \mathfrak{g}$

- **Anticommutativity**
\begin{align}
  [x, y] = -[y, x]
\end{align}

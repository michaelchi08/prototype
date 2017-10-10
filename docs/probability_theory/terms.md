# Probability Theory - Terms

- **Random Variables**: denoted using either upper case $X$ or lower case
  $x$ and a set of variables will typically be denoted as ${a, B, c}$

- **Domain**: of a variable dom($x$) denotes the states $x$ can take.

- **Statistical Independence**: The best way to think about statistical
independence is to ask whether or not knowing the state of variable $y$
tells you something more than you knew before about variable $x$, where
"knew before" means working with the joint probability $p(x, y)$ to figure
out what we can know about $x$ namely $p(x)$.

- **Singly-connected** if there is only one path from a vertex to another
vertex, other wise the graph is **multiply-connected**. This definition
applies regardless of whether or not the edges in the graph are directed.
An alternative name for a singly-connected graph is a **tree**,
a multiply-connected graph is also called a **loopy**.

- **Spanning tree**: of an undirected graph $G$ is a singly-connected
subset of the existing edges such that the resulting singly-connected
graph covers all vertices of $G$. On the right is a graph and an
associated spanning tree. A maximum weight spanning tree is a spanning
tree such that the sum of all weights on the edges of the tree is larger
than for any other spanning tree of $G$.

The central paradigm of probabilistic reasoning is to identify all
relevant variables $x_1, \dots, x_{N}$ in the environment, and make
a probabilistic model $p(x_1, \dots, x_{N}$ of their interaction.
Reasoning (inference) is then performed by introducing evidence that
sets variables in known states, and subsequently computing probabilities
of interest, conditioned on this evidence.

- **Probability density function (PDF)**: is of a continuous random variable,
whose integral across an interval gives the probability that the value of the
variable lies within the same interval.

\begin{equation}
  \int^{b}_{a} p(x) dx = 1
\end{equation}

# Span of a Set of Vectors

Consider a set of vectors $\{v_{1}, v_{2}, \dots, v_{k}\}$. The set of all
linear combinations of this set is called the span. Recall a linear combination
has the form  $\{c_{1} v_{1} + \dots + c_{k} v_{k}\}$ where each $c_{i}$ is
a real number scalar.

Let $V$ be a vector space and  let $S = \{v_1, v_2, \dots, v_n\}$ be a subset
of $V$. We say that $S$ spans $V$ if every vector $v$ in $V$ can be written as
a linear combination of vector in $S$.


### Example 1

Show that the set $S = \{(0, 1, 1), (1, 0, 1), (1, 1, 0)\}$ spans $\mathbb{R}^{3}$
and show that vector $(2, 4, 8)$ belongs to the span $S$ as a linear
combination of vectors in $S$.

---

A vector in $\mathbb{R}^{3}$ has the form $v = (x, y, z)$, hence we need to
show that every $v$ can be written as:

\begin{align}
    (x, y, z) &= c_1 (0, 0, 1) + c_2 (1, 0, 1) + c_3 (1, 1, 0) \\
        &= (c_2 + c_3, c_1 + c_3, c_1 + c_2)
\end{align}

This corresponds to the system of equations:

\begin{equation}
    c_2 + c_3 = x \\
    c_1 + c_3 = y \\
    c_1 + c_2 = z
\end{equation}

Which can be written in matrix form:

\begin{align}
    \begin{bmatrix}
        0 & 1 & 1 \\
        1 & 0 & 1 \\
        1 & 1 & 0
    \end{bmatrix}
    \begin{bmatrix}
        c_1 \\ c_2 \\ c_3
    \end{bmatrix}
    =
    \begin{bmatrix}
        x \\
        y \\
        z
    \end{bmatrix}
\end{align}

We can write this as:

\begin{equation}
    Ac = b
\end{equation}

Notice that $\text{det}(A) = 2$, hence $A$ is nonsingular and $c = A^{-1}b$. So
that a nontrivial solution exists. To write $(2, 4, 8)$ as a linear combination
of vectors in $S$, we find that.

\begin{align}
    A^{-1} =
    \begin{bmatrix}
        -0.5 & 0.5 & 0.5 \\
        0.5 & -0.5 & 0.5 \\
        0.5 & 0.5 & -0.5
    \end{bmatrix}
\end{align}

So that:

\begin{align}
    c =
    \begin{bmatrix}
        -0.5 & 0.5 & 0.5 \\
        0.5 & -0.5 & 0.5 \\
        0.5 & 0.5 & -0.5
    \end{bmatrix}
    \begin{bmatrix}
        2 \\ 4 \\ 8
    \end{bmatrix}
    = 
    \begin{bmatrix}
        5 \\ 3 \\ -1
    \end{bmatrix}
\end{align}

<!--- Suppose we start with $\{v_{1}, v_{2}\}$, where $v_{1} = \begin{bmatrix} 1 --->
<!--- & 3 \end{bmatrix}$ and $v_{2} = \begin{bmatrix} 2 & 5 \end{bmatrix}$. --->
<!---  --->
<!--- Then the span of $\{v_{1}, v_{2}\}$ is: --->
<!---  --->
<!--- \begin{align} --->
<!---     \{v_{1}, v_{2}\} = --->
<!---         c_{1} \begin{bmatrix} 1 \\ 5 \\ \end{bmatrix} --->
<!---         + --->
<!---         c_{2} \begin{bmatrix} 2 \\ 3 \\ \end{bmatrix} --->
<!--- \end{align} --->
<!---  --->
<!--- Let $c_{1} = 1$, $c_{2} = 4$. --->

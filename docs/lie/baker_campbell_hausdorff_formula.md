# Baker-Campbell-Hausdorff Formula

The structure of the Lie group near the identity element is displayed
explicitly by the Baker-Campbell-Hausdorff formula, an expansion in Lie algebra
elements $X$, $Y$ and their Lie brackets, all nested together within a single
exponent, it is the solution to $Z = log(e^{X} e^{Y})$ for possibly
non-commutative $X$ and $Y$ in the Lie algebra of a Lie group.

\begin{align}
  Z(X, Y) \\
    &= log(e^{X} e^{Y}) \\
    &= X + Y + \dfrac{1}{2} [X, Y] + \dfrac{1}{12}([X, [X, Y]] + [Y, [Y, X]])
    - \dfrac{1}{24} [Y, [X, [X, Y]]] + \dots
\end{align}

\begin{align}
  e^{tX} e^{tY} = \text{exp}(tX + tY + \dfrac{1}{2} t^{2} [X, Y] + O(t^{3}))
\end{align}

# Finite Difference

\begin{equation}
    \dfrac{\partial^{2} u}{\partial x \partial y} =
    \dfrac{\partial}{\partial x}\left(\dfrac{\partial u}{\partial y}\right) =
    \dfrac{\partial}{\partial y}\left(\dfrac{\partial u}{\partial x}\right)
\end{equation}

\begin{equation}
    \left(\dfrac{\partial^{2} u}{\partial x \partial y}\right)_{i, j} =
    \dfrac{
        \left(\dfrac{\partial u}{\partial y}\right)_{i + 1, j} -
        \left(\dfrac{\partial u}{\partial y}\right)_{i - 1, j}
    }{
        2 \Delta x
    } + \mathcal{O}(\Delta x)^{2}
\end{equation}

\begin{equation}
    \left(\dfrac{\partial u}{\partial y}\right)_{i + 1, j} =
    \dfrac{
        u_{i + 1, j + 1} - u_{i + 1, j - 1}
    }{
        2 \Delta y
    } + \mathcal{O}(\Delta y)^{2}
\end{equation}

\begin{equation}
    \left(\dfrac{\partial u}{\partial y}\right)_{i - 1, j} =
    \dfrac{
        u_{i - 1, j + 1} - u_{i - 1, j - 1}
    }{
        2 \Delta y
    } + \mathcal{O}(\Delta y)^{2}
\end{equation}

\begin{equation}
    \left(\dfrac{\partial^{2} u}{\partial x \partial y}\right)_{i, j} =
    \dfrac{
        (u_{i + 1, j + 1} - u_{i + 1, j - 1}) -
            (u_{i - 1, j + 1} - u_{i - 1, j - 1})
    }{
        4 \Delta x \Delta y
    } + \mathcal{O}((\Delta x)^{2}, (\Delta y)^{2})
\end{equation}

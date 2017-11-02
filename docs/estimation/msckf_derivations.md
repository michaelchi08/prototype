\[
  % General
  \newcommand{\state}{\mathbf{X}}
  \newcommand{\frame}{\mathcal{F}}
  \newcommand{\quat}{\mathbf{q}}
  \newcommand{\accel}{\mathbf{a}}
  \newcommand{\vel}{\mathbf{v}}
  \newcommand{\angvel}{\boldsymbol{\omega}}
  \newcommand{\pos}{\mathbf{r}}
  \newcommand{\p}{\mathbf{p}}
  \newcommand{\transform}{\mathbf{T}}
  \newcommand{\rot}[2]{{}^{#1}_{#2}\mathbf{R}}
  \newcommand{\rottilde}[2]{{}^{#1}_{#2}\mathbf{\tilde R}}
  \newcommand{\rothat}[2]{{}^{#1}_{#2}\mathbf{\hat R}}
  \newcommand{\bias}{\mathbf{b}}
  \newcommand{\noise}{\mathbf{n}}
  \newcommand{\I}{\mathbf{I}}
  % Frames
  \newcommand{\global}{\text{G}}
  \newcommand{\imu}{\text{I}}
  \newcommand{\cam}{\text{C}}
  \newcommand{\body}{\text{B}}
  \newcommand{\BtoC}{{}^{C}_{B}}
  \newcommand{\BtoG}{{}^{G}_{B}}
  \newcommand{\GtoB}{{}^{B}_{G}}
  \newcommand{\inG}{{}^{G}}
  \newcommand{\inB}{{}^{B}}
  \newcommand{\inC}{{}^{C}}
  % Custom
  \newcommand{\skew}[1]{\lfloor #1 \enspace \times \rfloor}
  \newcommand{\dtheta}{\boldsymbol{\tilde{\theta}}}
  \newcommand{\feaTruGpos}{{}^{G}\p_{f}}
  \newcommand{\feaErrGpos}{{}^{G}\tilde{\p}_{f}}
  \newcommand{\feaEstGpos}{{}^{G}\hat{\p}_{f}}
  \newcommand{\feaTruCpos}{{}^{C}\p_{f}}
  \newcommand{\feaErrCpos}{{}^{C}\tilde{\p}_{f}}
  \newcommand{\feaEstCpos}{{}^{C}\hat{\p}_{f}}
  \newcommand{\bodyTruGpos}{{}^{G}\p_{B}}
  \newcommand{\bodyErrGpos}{{}^{G}\tilde{\p}_{B}}
  \newcommand{\bodyEstGpos}{{}^{G}\hat{\p}_{B}}
  \newcommand{\bodyTruCpos}{{}^{C}\p_{B}}
  \newcommand{\bodyErrCpos}{{}^{C}\tilde{\p}_{B}}
  \newcommand{\bodyEstCpos}{{}^{C}\hat{\p}_{B}}
  \newcommand{\bodyTruGvel}{{}^{G}\vel_{B}}
  \newcommand{\bodyErrGvel}{{}^{G}\tilde{\vel}_{B}}
  \newcommand{\bodyEstGvel}{{}^{G}\hat{\vel}_{B}}
  \newcommand{\bodyTruGaccel}{{}^{G}\accel_{B}}
  \newcommand{\bodyErrGaccel}{{}^{G}\tilde{\accel}_{B}}
  \newcommand{\bodyEstGaccel}{{}^{G}\hat{\accel}_{B}}
\]

\begin{align}
  \feaErrCpos &= \feaTruCpos - \feaEstCpos \\
  &=
  \left(
    \rot{\cam}{\body}
    \rot{\body}{\global}(t_{n})
    (\feaTruGpos - \bodyTruGpos(t_{n})) + \bodyTruCpos
  \right) -
  \left(
    \rot{\cam}{\body}
    \rothat{\body}{\global}(t_{n})
    (\feaTruGpos - \bodyEstGpos(t_{n})) + \bodyEstCpos
  \right)
  \label{eq:fpos_err_1} \\
  &=
  \rot{\cam}{\body}
    \left(
      \rot{\body}{\global}(t_{n})
      (\feaTruGpos - \bodyTruGpos(t_{n})) -
      \rothat{\body}{\global}(\hat{t}_{n})
      (\feaTruGpos - \bodyEstGpos(t_{n}))
    \right) + \bodyTruCpos - \bodyEstCpos
  \label{eq:fpos_err_2} \\
  &=
  \rot{\cam}{\body}
    \left(
      \rot{\body}{\global}(t_{n})
      (\feaTruGpos - \bodyTruGpos(t_{n})) -
      \rothat{\body}{\global}(\hat{t}_{n})
      (\feaTruGpos - \bodyEstGpos(t_{n}))
    \right) + \bodyErrCpos
  \label{eq:fpos_err_3} \\
\end{align}

\begin{align}
  \rot{\body}{\global} \simeq
    & \rot{\body}{\global}(\hat{t}_{n}) +
      \dfrac{\partial \rot{\body}{\global}}{\partial t_{n}} (\hat{t}_{n}) \tilde{t}_{n} \\
    & \rothat{\body}{\global}(\hat{t}_{n})
        (\I_{3} - \skew{\inG{\dtheta}(\hat{t}_{n})}) -
        \skew{\inB{\angvel}(\hat{t}_{n})}
        \rothat{\body}{\global} \tilde{t}_{n}
        \label{eq:rot_approx} \\[1em]
  \bodyTruGpos(t_{n}) \simeq
    & \bodyTruGpos(\hat{t}_{n}) +
      \inG{\vel}_{B}(\hat{t}_{n}) \tilde{t}_{n}
      \label{eq:fpos_approx} \\[2em]
\end{align}

\begin{align}
  \feaErrCpos
  =&
    \rot{\cam}{\body} \left(
      \rot{\body}{\global}(t_{n})
      (
        \feaTruGpos -
        {
          \color{red}
          \bodyTruGpos(\hat{t}_{n}) -
          \inG{\vel}_{B}(\hat{t}_{n}) \tilde{t}_{n}
        }
      ) -
      \rothat{\body}{\global}(t_{n})
      (\feaTruGpos - \bodyEstGpos(t_{n})) \right) +
    \bodyErrCpos
    & \text{Substituting \eqref{eq:fpos_approx} in \eqref{eq:fpos_err_3}}
    \label{eq:fpos_err_4} \\
  =&
    \rot{\cam}{\body} \left[
    {
      \color{red}
      \left\{
        \rothat{\body}{\global}(\I_{3} - \skew{\inG{\dtheta}(\hat{t}_{n})}) -
        \skew{\inB{\angvel}(\hat{t}_{n})}
        \rothat{\body}{\global} \tilde{t}_{n}
      \right\}
    }
    (
      \feaTruGpos -
      \bodyTruGpos(\hat{t}_{n}) -
      \inG{\vel}_{B}(\hat{t}_{n}) \tilde{t}_{n}
    ) -
    \rothat{\body}{\global}(t_{n})
    (\feaTruGpos - \bodyEstGpos(t_{n})) \right] + \bodyErrCpos
    & \text{Substituting \eqref{eq:rot_approx} in \eqref{eq:fpos_err_4}}
    \label{eq:fpos_err_5} \\
  =&
    \rot{\cam}{\body} \left[
    {
      \color{red}
      \rothat{\body}{\global}
        \left\{
          (
            \feaTruGpos -
            \bodyTruGpos(\hat{t}_{n}) -
            \inG{\vel}_{B}(\hat{t}_{n}) \tilde{t}_{n}
          ) -
          \skew{\inG{\dtheta}(\hat{t}_{n})}
          (
            \feaTruGpos -
            \bodyTruGpos(\hat{t}_{n}) -
            \inG{\vel}_{B}(\hat{t}_{n}) \tilde{t}_{n}
          )
        \right\}  -
        \skew{\inB{\angvel}(\hat{t}_{n})} \rothat{\body}{\global}
        \tilde{t}_{n}
        (
          \feaTruGpos -
          \bodyTruGpos(\hat{t}_{n}) -
          \inG{\vel}_{B}(\hat{t}_{n}) \tilde{t}_{n}
        )
    } -
    \rothat{\body}{\global}(t_{n})
    (
      \feaTruGpos -
      \bodyEstGpos(\hat{t}_{n})
    )) \right] + \bodyErrCpos
    & \text{
      Multiply \eqref{eq:rot_approx} in \eqref{eq:fpos_err_5} with
      $(
        \feaTruGpos -
        \bodyTruGpos(\hat{t}_{n}) +
        \inG{\vel}_{B}(\hat{t}_{n}) \tilde{t}_{n}
      )$
    }
    \label{eq:fpos_err_6} \\
  =&
    \rot{\cam}{\body} \left[
      \rothat{\body}{\global}
        \left\{
          (
            \feaTruGpos -
            \bodyTruGpos(\hat{t}_{n}) -
            \inG{\vel}_{B}(\hat{t}_{n}) \tilde{t}_{n}
          ) -
          \skew{\inG{\dtheta}(\hat{t}_{n})} (
            \feaTruGpos -
            \bodyTruGpos(\hat{t}_{n}) -
            \inG{\vel}_{B}(\hat{t}_{n}) \tilde{t}_{n})
        \right\}
        {
          \color{red}
          {- \rothat{\body}{\global}(t_{n})}
          (\feaTruGpos - \bodyEstGpos(t_{n}))
        } -
        \skew{\inB{\angvel}(\hat{t}_{n})} \rothat{\body}{\global}
          \tilde{t}_{n}
          (
            \feaTruGpos -
            \bodyTruGpos(\hat{t}_{n}) -
            \inG{\vel}_{B}(\hat{t}_{n}) \tilde{t}_{n}
          )
    \right] + \bodyErrCpos
    & \text{Rearrange \eqref{eq:fpos_err_6} so that
            $\rothat{\body}{\global}(t_{n})$ terms are closer together}
    \label{eq:fpos_err_7} \\
  =&
    \rot{\cam}{\body} \left[
      \rothat{\body}{\global}
        \left\{
            (\feaTruGpos -
            \bodyTruGpos(\hat{t}_{n}) -
            \inG{\vel}_{B}(\hat{t}_{n}) \tilde{t}_{n}) -
            (\feaTruGpos - \bodyEstGpos(t_{n})) -
          \skew{\inG{\dtheta}(\hat{t}_{n})} (\feaTruGpos - \bodyTruGpos(t_{n}))
        \right\} -
          \skew{\inB{\angvel}(\hat{t}_{n})} \rothat{\body}{\global}
            \tilde{t}_{n} (\feaTruGpos - \bodyTruGpos(t_{n}))
    \right] + \bodyErrCpos
    & \text{Combine $\rothat{\body}{\global}(t_{n})$ terms together}
    \label{eq:fpos_err_8} \\
  =&
    \rot{\cam}{\body} \left[
      \rothat{\body}{\global}
        \left\{
          (
            {
              \color{red}
              \inG{\tilde{\p}}_{f} -
              \inG{\tilde{\p}}_{B}(\hat{t}_{n})
            }
          ) -
          \skew{\inG{\dtheta}(\hat{t}_{n})}
          (\feaTruGpos - \bodyTruGpos(t_{n})
        \right\} -
          \skew{\inB{\angvel}(\hat{t}_{n})} \rothat{\body}{\global}
          \tilde{t}_{n} (\feaTruGpos - \bodyTruGpos(t_{n}))
    \right] + \bodyErrCpos
    & \text{Combine $\p$ and $\hat\p$ to form $\tilde\p$ where $\tilde\p = \p - \hat\p$}
    \label{eq:fpos_err_9} \\
  =&
    \rot{\cam}{\body} \left[
      \rothat{\body}{\global}
        \left\{
          (\inG{\tilde{\p}}_{f} - \inG{\tilde{\p}}_{B}(\hat{t}_{n})) -
          \skew{\inG{\dtheta}(\hat{t}_{n})}
          (\feaTruGpos -
            {
              \color{red}
              \bodyTruGpos(\hat{t}_{n}) -
              \inG{\vel}_{B}(\hat{t}_{n}) \tilde{t}_{n}
            }
          )
        \right\} -
          \skew{\inB{\angvel}(\hat{t}_{n})} \rothat{\body}{\global}
          \tilde{t}_{n}
          (\feaTruGpos -
            {
              \color{red}
              \bodyTruGpos(\hat{t}_{n}) -
              \inG{\vel}_{B}(\hat{t}_{n}) \tilde{t}_{n}
            }
          )
    \right] + \bodyErrCpos
    & \text{Substitute \eqref{eq:fpos_approx} in \eqref{eq:fpos_err_6}}
    \label{eq:fpos_err_10} \\
\end{align}

\begin{align}
  \bodyErrGpos(\hat{t}_{n}) &=
    \bodyErrGpos(t - \hat{t}_{d}) +
    \dfrac{k \hat{t}_{r}}{N} \bodyErrGvel(t - \hat{t}_{d}) +
    \dfrac{(k \hat{t}_{r})^{2}}{N} \bodyErrGaccel(t - \hat{t}_{d}) +
    \cdots
    \label{eq:body_err_approx} \\
  \dtheta(\hat{t}_{n}) &=
    \dtheta(t - \hat{t}_{d}) +
    \dfrac{k \hat{t}_{r}}{N} \tilde{\angvel}_{B} (t - \hat{t}_{d}) + \cdots
    \label{eq:dtheta_approx}
\end{align}

<!-- \begin{align} -->
<!--   \feaErrCpos -->
<!--   =& -->
<!--     \rot{\cam}{\body} \left[ -->
<!--       \rothat{\body}{\global} -->
<!--         \left\{ -->
<!--           (\inG{\tilde{\p}}_{f} - \inG{\tilde{\p}}_{B}(\hat{t}_{n})) - -->
<!--           \skew{ -->
<!--             { -->
<!--               \color{red} -->
<!--               \inG{\dtheta}(t - \hat{t}_{d}) -->
<!--             } -->
<!--           } -->
<!--           ( -->
<!--             \feaTruGpos - -->
<!--             \bodyTruGpos(\hat{t}_{n}) - -->
<!--             \inG{\vel}_{B}(\hat{t}_{n}) \tilde{t}_{n} -->
<!--           ) -->
<!--         \right\} - -->
<!--           \skew{\inB{\angvel}(\hat{t}_{n})} \rothat{\body}{\global} -->
<!--           \tilde{t}_{n} -->
<!--           ( -->
<!--             \feaTruGpos - -->
<!--             \bodyTruGpos(\hat{t}_{n}) - -->
<!--             \inG{\vel}_{B}(\hat{t}_{n}) \tilde{t}_{n} -->
<!--           ) -->
<!--     \right] + \bodyErrCpos -->
<!--     & \text{Substitute \eqref{eq:dtheta_approx} (1st term only) in \eqref{eq:fpos_err_7}} -->
<!--     \label{eq:fpos_err_12} \\ -->
<!--   =& -->
<!--     \rot{\cam}{\body} \left[ -->
<!--       \rothat{\body}{\global} -->
<!--         \left\{ -->
<!--           ( -->
<!--             \inG{\tilde{\p}}_{f} - -->
<!--             { -->
<!--               \color{red} -->
<!--               \bodyErrGpos(t - \hat{t}_{d}) - -->
<!--               \dfrac{k \hat{t}_{r}}{N} \bodyErrGvel(t - \hat{t}_{d}) -->
<!--             } -->
<!--           ) - -->
<!--           \skew{\inG{\dtheta}(t - \hat{t}_{d})} -->
<!--           ( -->
<!--             \feaTruGpos - -->
<!--             \bodyTruGpos(\hat{t}_{n}) - -->
<!--             \inG{\vel}_{B}(\hat{t}_{n}) \tilde{t}_{n} -->
<!--           ) -->
<!--         \right\} - -->
<!--           \skew{\inB{\angvel}(\hat{t}_{n})} \rothat{\body}{\global} -->
<!--           \tilde{t}_{n} -->
<!--           ( -->
<!--             \feaTruGpos - -->
<!--             \bodyTruGpos(\hat{t}_{n}) - -->
<!--             \inG{\vel}_{B}(\hat{t}_{n}) \tilde{t}_{n} -->
<!--           ) -->
<!--     \right] + \bodyErrCpos -->
<!--     & \text{Substitute \eqref{eq:body_err_approx} (1st and 2nd term only) in -->
<!--             \eqref{eq:fpos_err_8}} -->
<!--     \label{eq:fpos_err_13} \\ -->
<!--   =& -->
<!--       { -->
<!--         \color{red} -->
<!--         \rot{\cam}{\body} -->
<!--         \rothat{\body}{\global} -->
<!--       } -->
<!--         \left\{ -->
<!--           \inG{\tilde{\p}}_{f} - -->
<!--           \bodyErrGpos(t - \hat{t}_{d}) - -->
<!--           \dfrac{k \hat{t}_{r}}{N} \bodyErrGvel(t - \hat{t}_{d}) -->
<!--         \right\} \\ \nonumber &- -->
<!--       { -->
<!--         \color{red} -->
<!--         \rot{\cam}{\body} -->
<!--         \rothat{\body}{\global} -->
<!--       } -->
<!--         \skew{\inG{\dtheta}(t - \hat{t}_{d})} -->
<!--         \left[ -->
<!--             \feaTruGpos - -->
<!--             \bodyTruGpos(\hat{t}_{n}) - -->
<!--             \inG{\vel}_{B}(\hat{t}_{n}) \tilde{t}_{n} -->
<!--         \right] \\ \nonumber -->
<!--       & - { -->
<!--         \color{red} -->
<!--         \rot{\cam}{\body} -->
<!--       } -->
<!--       \skew{\inB{\angvel}(\hat{t}_{n})} \rothat{\body}{\global} -->
<!--       \tilde{t}_{n} -->
<!--       ( -->
<!--         \feaTruGpos - -->
<!--         \bodyTruGpos(\hat{t}_{n}) - -->
<!--         \inG{\vel}_{B}(\hat{t}_{n}) \tilde{t}_{n} -->
<!--       ) \\ \nonumber -->
<!--       & + \bodyErrCpos -->
<!--     & \text{Expand terms to include $\rot{\cam}{\body}$ and -->
<!--             $\rothat{\body}{\global}$} -->
<!--     \label{eq:fpos_err_14} \\ -->
<!-- \end{align} -->

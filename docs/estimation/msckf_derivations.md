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
  \right) \\
  &=
  \rot{\cam}{\body}
    \left(
      \rot{\body}{\global}(t_{n})
      (\feaTruGpos - \bodyTruGpos(t_{n})) -
      \rothat{\body}{\global}(\hat{t}_{n})
      (\feaTruGpos - \bodyEstGpos(t_{n}))
    \right) +
    \bodyTruCpos - \bodyEstCpos \\
  &=
  \rot{\cam}{\body}
    \left(
      \rot{\body}{\global}(t_{n})
      (\feaTruGpos - \bodyTruGpos(t_{n})) -
      \rothat{\body}{\global}(\hat{t}_{n})
      (\feaTruGpos - \bodyEstGpos(t_{n}))
    \right) +
  \bodyErrGpos
\end{align}

\begin{align}
  \rot{\body}{\global} \simeq
    & \rottilde{\body}{\global} +
      \dfrac{\partial \rot{\body}{\global}}{\partial t_{n}} (\hat{t}_{n}) \tilde{t}_{n} \\
    & \rothat{\body}{\global}(\I_{3} - \skew{\inG{\dtheta}} -
      \skew{\inB{\angvel}(\hat{t}_{n})}
      \rothat{\body}{\global} \tilde{t}_{n}
      \label{eq:rot_approx} \\[1em]
  \bodyTruGpos(t_{n}) \simeq
    & \bodyTruGpos(\hat{t}_{n}) +
      \inG{\vel}_{B}(\hat{t}_{n}) \hat{t}_{n}
      \label{eq:fpos_approx} \\[2em]
\end{align}

\begin{align}
  \feaErrCpos
  =&
    \rot{\cam}{\body} \left(
      \rot{\body}{\global}(t_{n})
      (\feaTruGpos - \bodyTruGpos(t_{n})) -
      \rothat{\body}{\global}(t_{n})
      (\feaTruGpos - \bodyEstGpos(t_{n})) \right) +
    \bodyErrGpos
    \label{eq:fpos_err_1} \\
  =&
    \rot{\cam}{\body} \left[
    {
      \color{red}
      \left(
        \rothat{\body}{\global}(\I_{3} - \skew{\inG{\dtheta}} -
        \skew{\inB{\angvel}(\hat{t}_{n})}
        \rothat{\body}{\global} \tilde{t}_{n}
      \right)
    }
    (\feaTruGpos - \bodyTruGpos(t_{n})) -
    \rothat{\body}{\global}(t_{n})
    (\feaTruGpos - \bodyEstGpos(t_{n})) \right] + \bodyErrGpos
    & \text{Substituting \eqref{eq:rot_approx} into \eqref{eq:fpos_err_1}}
    \label{eq:fpos_err_2} \\
  =&
    \rot{\cam}{\body} \left[
    {
      \color{red}
      \rothat{\body}{\global}
        \left(
          (\feaTruGpos - \bodyTruGpos(t_{n})) -
          \skew{\inG{\dtheta}} (\feaTruGpos - \bodyTruGpos(t_{n}))
        \right)  -
        \skew{\inB{\angvel}(\hat{t}_{n})} \rothat{\body}{\global}
          \tilde{t}_{n} (\feaTruGpos - \bodyTruGpos(t_{n}))
    } -
    \rothat{\body}{\global}(t_{n})
    (\feaTruGpos - \bodyEstGpos(t_{n})) \right] + \bodyErrGpos
    & \text{Multiply \eqref{eq:rot_approx} in \eqref{eq:fpos_err_2} with
            $(\feaTruGpos - \bodyTruGpos(t_{n})$)}
    \label{eq:fpos_err_3} \\
  =&
    \rot{\cam}{\body} \left[
      \rothat{\body}{\global}
        \left(
          (\feaTruGpos - \bodyTruGpos(t_{n})) -
          \skew{\inG{\dtheta}} (\feaTruGpos - \bodyTruGpos(t_{n}))
        \right)
        {
          \color{red}
          {- \rothat{\body}{\global}(t_{n})}
          (\feaTruGpos - \bodyEstGpos(t_{n})
        } ) -
        \skew{\inB{\angvel}(\hat{t}_{n})} \rothat{\body}{\global}
          \tilde{t}_{n} (\feaTruGpos - \bodyTruGpos(t_{n}))
    \right] + \bodyErrGpos
    & \text{Rearrange \eqref{eq:fpos_err_1} so that
            $\rothat{\body}{\global}(t_{n})$ terms are closer together}
    \label{eq:fpos_err_4} \\
  =&
    \rot{\cam}{\body} \left[
      \rothat{\body}{\global}
        \left(
          {
            \color{red}
            (\feaTruGpos -
              \bodyTruGpos(t_{n})) -
              (\feaTruGpos - \bodyEstGpos(t_{n}))
          } -
          \skew{\inG{\dtheta}} (\feaTruGpos - \bodyTruGpos(t_{n})) \right) -
          \skew{\inB{\angvel}(\hat{t}_{n})} \rothat{\body}{\global}
            \tilde{t}_{n} (\feaTruGpos - \bodyTruGpos(t_{n}))
    \right] + \bodyErrGpos
    & \text{Combine $\rothat{\body}{\global}(t_{n})$ terms together}
    \label{eq:fpos_err_5} \\
  =&
    \rot{\cam}{\body} \left[
      \rothat{\body}{\global}
        \left(
          (\feaTruGpos -
            {
              \color{red}
              \bodyTruGpos(\hat{t}_{n}) -
              \inG{\vel}_{B}(\hat{t}_{n}) \tilde{t}_{n}
            }
          ) -
          (\feaTruGpos - \bodyEstGpos(t_{n})) -
          \skew{\inG{\dtheta}}
          (\feaTruGpos -
            {
              \color{red}
              \bodyTruGpos(\hat{t}_{n}) -
              \inG{\vel}_{B}(\hat{t}_{n}) \tilde{t}_{n}
            }
          )
        \right) -
          \skew{\inB{\angvel}(\hat{t}_{n})} \rothat{\body}{\global}
          \tilde{t}_{n}
          (\feaTruGpos -
            {
              \color{red}
              \bodyTruGpos(\hat{t}_{n}) -
              \inG{\vel}_{B}(\hat{t}_{n}) \tilde{t}_{n}
            }
          )
    \right] + \bodyErrGpos
    & \text{Substitute \eqref{eq:fpos_approx} into \eqref{eq:fpos_err_5}}
    \label{eq:fpos_err_6} \\
  =&
    \rot{\cam}{\body} \left[
      \rothat{\body}{\global}
        \left(
          (
            {
              \color{red}
              \inG{\tilde{\p}}_{f} -
              \inG{\tilde{\p}}_{B}(\hat{t}_{n})
            } -
            \inG{\vel}_{B}(\hat{t}_{n}) \tilde{t}_{n}) -
          \skew{\inG{\dtheta}}
          (\feaTruGpos -
            \bodyTruGpos(\hat{t}_{n}) -
            \inG{\vel}_{B}(\hat{t}_{n}) \tilde{t}_{n})
        \right) -
          \skew{\inB{\angvel}(\hat{t}_{n})} \rothat{\body}{\global}
          \tilde{t}_{n} (\feaTruGpos -
                         \bodyTruGpos(\hat{t}_{n}) -
                         \inG{\vel}_{B}(\hat{t}_{n}) \tilde{t}_{n})
    \right] + \bodyErrGpos
    & \text{Combine $\p$ and $\hat\p$ to form $\tilde\p$ where $\tilde\p = \p - \hat\p$}
    \label{eq:fpos_err_7} \\
\end{align}

# Multi-State Constraint Kalman Filter

$$
  % General
  \newcommand{\state}{\mathbf{X}}
  \newcommand{\frame}{\mathcal{F}}
  \newcommand{\quat}{\mathbf{q}}
  \newcommand{\accel}{\mathbf{a}}
  \newcommand{\vel}{\mathbf{v}}
  \newcommand{\angvel}{\boldsymbol{\omega}}
  \newcommand{\pos}{\mathbf{r}}
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
  \newcommand{\inG}{{}^{G}}
  \newcommand{\inB}{{}^{B}}
  \newcommand{\inC}{{}^{C}}
  % Custom
  \newcommand{\dtheta}{\boldsymbol{\delta\theta}}
  \newcommand{\skew}[1]{\lfloor #1 \enspace \times \rfloor}
  \newcommand{\feaTruGpos}{{}^{G}p_{f}}
  \newcommand{\feaErrGpos}{{}^{G}\tilde{p}_{f}}
  \newcommand{\feaEstGpos}{{}^{G}\hat{p}_{f}}
  \newcommand{\feaTruCpos}{{}^{C}p_{f}}
  \newcommand{\feaErrCpos}{{}^{C}\tilde{p}_{f}}
  \newcommand{\feaEstCpos}{{}^{C}\hat{p}_{f}}
  \newcommand{\bodyTruGpos}{{}^{G}p_{B}}
  \newcommand{\bodyErrGpos}{{}^{G}\tilde{p}_{B}}
  \newcommand{\bodyEstGpos}{{}^{G}\hat{p}_{B}}
  \newcommand{\bodyTruCpos}{{}^{C}p_{B}}
  \newcommand{\bodyErrCpos}{{}^{C}\tilde{p}_{B}}
  \newcommand{\bodyEstCpos}{{}^{C}\hat{p}_{B}} $$



## Notations

We employ the following notation throughout this work: A vector in the Global
frame $\frame_{\global}$ can be expressed as $\pos_{\global}$, or more
precisely if the vector describes the position of the IMU frame $\frame_\imu$
expressed in $\frame_{\global}$, the vector can be written as
$\pos_{\global}^{{\global\imu}}$ with $\global$ and $\imu$ as start and end
points, or for brevity as $\pos_{\global}^{\imu}$. Similarly a transformation
between $\frame_{\global}$ to $\frame_\imu$ can be represented by a homogeneous
transform matrix $\transform_{\imu\global}$, where its rotation matrix
component can be written as $\rot_{\imu\global}$.



## State Vector

The IMU state vector is described by the vector:

\begin{equation}
  \state_{\imu} = \begin{bmatrix}
    \quat_{\global\imu}^{T}
    & \bias_{g}^{T}
    & \vel_{\global}^{{\global\imu}^{T}}
    & \bias_{a}^{T}
    & \pos_{\global}^{{\global\imu}^{T}}
  \end{bmatrix}^{T}
\end{equation}

The IMU **error**-state vector is described by the vector:

\begin{equation}
  \tilde{\state}_{\imu} = \begin{bmatrix}
    \boldsymbol{\delta\theta}_\imu^{T}
    & \tilde\bias_{g}^{T}
    & \tilde\vel_{\global}^{{\global\imu}^{T}}
    & \tilde\bias_{a}^{T}
    & \tilde\pos_{\global}^{{\global\imu}^{T}}
  \end{bmatrix}^{T}
\end{equation}

The $i$-th camera state vector is:

\begin{equation}
  \hat \state_{\cam_{i}} = \begin{bmatrix}
    \quat_{\global\cam_{i}}^{T}
    & \pos_{\global}^{{\global\cam_{i}}^{T}}
  \end{bmatrix}
\end{equation}

Since MSCKF is a sliding window Kalman Filter, the camera poses are augmented
to the end of the state vector at time step $k$, and has the following form:

\begin{equation}
  \hat \state = \begin{bmatrix}
    \hat \state_{\imu}^{T}
    & \hat \state_{\cam_{i}}^{T}
    & \dots
    & \hat \state_{\cam_{N}}^{T}
  \end{bmatrix}
\end{equation}

For brevity we have denoted

\begin{equation}
  \hat \angvel = \angvel_{m} - \hat \bias_{g} - \rot_{\hat \quat} \angvel_{\global}
\end{equation}

\begin{equation}
  \hat \accel = \accel_{m} - \hat \bias_{a}
\end{equation}

Linearized continuous time model for the IMU error state is:

\begin{equation}
  \dot{\tilde\state}_{\imu} =
    \mathbf{F} \tilde\state_{\imu} +
    \mathbf{G} \noise_{\imu}
\end{equation}

where

\begin{equation}
  \noise_{\imu} = \begin{bmatrix}
    \noise_{g}^{T}
    & \noise_{wg}^{T}
    & \noise_{a}^{T}
    & \noise_{wa}^{T}
  \end{bmatrix}
\end{equation}


## Feature Position Estimation




## Measurement model

The residual of the $i$-th camera pose and $j$-th feature track
measurement is:

\begin{equation} \label{eq:residual}
  r^{j}_{i} = z^{(j)}_{i} - \hat{z}^{(j)}_{i}
\end{equation}

i.e. it is the difference between the measured pixel location $z^{(j)}_{i}$ and
the predicted pixel location $\hat{z}^{(j)}_{i}$ this is also known as the
**reprojection error**.

Linearizing about the estimates for the camera pose and for the feature
position, the residual $\eqref{eq:residual}$ can be approximated as

\begin{equation} \label{eq:residual_single_linearize}
  r^{(j)}_{i} \simeq
    \mathbf{H}^{(j)}_{\mathbf{X}_{i}} \tilde{\state} +
    \mathbf{H}^{(j)}_{f_{i}} ~^{G}\!\tilde{P}_{f_{j}} + \noise^{(j)}_{i}
\end{equation}

where $\mathbf{H}^{(j)}_{\mathbf{X}_{i}}$ and $\mathbf{H}^{(j)}_{f_{i}}$
represent the Jacobians of the measurement $z^{(j)}_{i}$ with respect to the
state $\mathbf{X}_{i}$ and feature position $f_{i}$ and
$~^{G}\!\tilde{P}_{f_{j}}$ is the error in the position estimate of $f_{j}$.

\begin{align}
  \mathbf{H}^{(j)}_{\mathbf{X}_{i}} = \begin{bmatrix}
    \dfrac{\partial{z}}{\partial{\state_{\imu}}}
    & \dfrac{\partial{z}}{\partial{\state_{\cam_{0}}}}
    & \cdots
    & \dfrac{\partial{z}}{\partial{\state_{\cam_{N}}}}
  \end{bmatrix}
\end{align}

To obtain residuals over all camera poses, however, we stack the residuals
over all measurements $M_{J}$, and so $\eqref{eq:residual_single_linearize}$
can be rewritten as:

\begin{equation} \label{eq:residual_stacked_linearize}
  r^{(j)} \simeq
    \mathbf{H}^{(j)}_{\mathbf{X}} \tilde{\state} +
    \mathbf{H}^{(j)}_{f} ~^{G}\!\tilde{P}_{f_{j}} + \noise^{(j)}
\end{equation}

where $r^{(j)}$, $\mathbf{H}^{(j)}_{\mathbf{X}}$, $\mathbf{H}^{(j)}_{f}$ and
$n^{(j)}$ are block vectors or matrices.


## Measurement Model Jacobian Derivation

\begin{align}
  \mathbf{H}^{(j)}_{\mathbf{X}_{i}} &= \begin{bmatrix}
    \dfrac{\partial{z}}{\partial{\state_{\imu}}}
    & \dfrac{\partial{z}}{\partial{\state_{\cam_{0}}}}
    & \cdots
    & \dfrac{\partial{z}}{\partial{\state_{\cam_{N}}}}
  \end{bmatrix} \\
  \mathbf{H}^{(j)}_{\mathbf{X}_{i}} &= \begin{bmatrix}
    \underbrace{0_{2 \times 15}}_\text{w.r.t IMU state}
    & \underbrace{0_{2 \times 6}}_\text{w.r.t Camera state}
    & \cdots &
    & \mathbf{J}^{(j)}_{i}
      \lfloor\! ~^{C_{i}}\!\hat{\state}_{f_{j}} \times \rfloor
    & - \mathbf{J}^{(j)}_{i} C(~^{C_{i}}_{G}\!\hat{\bar{q}})
  \end{bmatrix} \\[2em]
  % --- z
  z &= h(g(C_{Ci:G}, P_{f}, P_{C})) \\
  % --- g
  g &= C_{Ci:G} (P_{f} - P_{C}) \\
  % --- h
  h &= \begin{bmatrix}
    X \mathbin{/} Z \\
    Y \mathbin{/} Z \\
  \end{bmatrix} \\[2em]
  % --- dz / dX_{IMU}
  \dfrac{\partial{z}}{\partial{\state_{\imu}}} &= 0_{2 \times 15} \\
  % --- dz / dP_{CAM}
  \dfrac{\partial{z}}{\partial{\state_{\cam}}} &= \begin{bmatrix}
    \dfrac{\partial{z}}{\partial{\dtheta}}
    & \dfrac{\partial{z}}{\partial{\pos_{\global}^{{\global\cam_{i}}^{T}}}}
  \end{bmatrix} \\
  & = \begin{bmatrix}
    \dfrac{\partial{h}}{\partial{g}}
    \dfrac{\partial{g}}{\partial{P_{f}}}
    & \dfrac{\partial{h}}{\partial{g}}
    \dfrac{\partial{g}}{\partial{P_{f}}} \\[2em]
  \end{bmatrix} \\[2em]
  % --- dh / dg
  \dfrac{\partial{h}}{\partial{g}} &= \begin{bmatrix}
    \dfrac{\partial{h}}{\partial{X}} &
    \dfrac{\partial{h}}{\partial{Y}} &
    \dfrac{\partial{h}}{\partial{Z}}
  \end{bmatrix} \\
  &= \begin{bmatrix}
    1 \mathbin{/} Z & 0 & X \mathbin{/} Z^{2} \\
    0 & 1 \mathbin{/} Z & Y \mathbin{/} Z^{2}
  \end{bmatrix} \\
  &= \dfrac{1}{Z} \begin{bmatrix}
    1 & 0 & X \mathbin{/} Z \\
    0 & 1 & Y \mathbin{/} Z
  \end{bmatrix} \\[2em]
  % --- dg / dP_f
  \dfrac{\partial{g}}{\partial{P_{f}}} &= \begin{bmatrix}
    \dfrac{\partial{g}}{\partial{P_{f_{x}}}} &
    \dfrac{\partial{g}}{\partial{P_{f_{y}}}} &
    \dfrac{\partial{g}}{\partial{P_{f_{z}}}}
  \end{bmatrix} \\
  &= \begin{bmatrix}
    C_{Ci:G} \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix},
    & C_{Ci:G} \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix},
    & C_{Ci:G} \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}
  \end{bmatrix} \\
  &= \begin{bmatrix}
    1 & 0 & 0 \\
    0 & 1 & 0 \\
    0 & 0 & 1
  \end{bmatrix} C_{Ci:G}
\end{align}

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

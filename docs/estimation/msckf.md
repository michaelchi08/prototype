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
  \newcommand{\residual}{\mathbf{r}}
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
  \newcommand{\bodyEstCpos}{{}^{C}\hat{p}_{B}}
$$



## Notations

We employ the following notation throughout this work: A vector in the Global
frame $\frame_{\global}$ can be expressed as $\pos_{\global}$, or more
precisely if the vector describes the position of the IMU frame $\frame_\imu$
expressed in $\frame_{\global}$, the vector can be written as
$\pos_{\global}^{{\global\imu}}$ with $\global$ and $\imu$ as start and end
points, or for brevity as $\pos_{\global}^{\imu}$. Similarly a transformation
between $\frame_{\global}$ to $\frame_\imu$ can be represented by a homogeneous
transform matrix $\transform_{\imu\global}$, where its rotation matrix
component can be written as $\rot{\imu}{\global}$.



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
  \hat \angvel =
    \angvel_{m} - \hat \bias_{g} - \rot{}{}_{\hat \quat} \angvel_{\global}
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
    \mathbf{H}^{(j)}_{f_{i}} \inG{\tilde{\mathbf{P}}}_{f_{j}} +
    \noise^{(j)}_{i}
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

\begin{equation} \label{eq:residual_approx}
  \residual^{(j)} \simeq
    \mathbf{H}^{(j)}_{\state} \tilde{\state} +
    \mathbf{H}^{(j)}_{f} \inG{\tilde{\mathbf{P}}_{f_{j}}} +
    \mathbf{n}^{(j)}
\end{equation}

where $r^{(j)}$, $\mathbf{H}^{(j)}_{\mathbf{X}}$, $\mathbf{H}^{(j)}_{f}$ and
$n^{(j)}$ are block vectors or matrices.

The problem with $\eqref{eq:residual_approx}$ is that the state estimate,
$\state$, was used to compute the feature position estimate, therefore the
feature position error $\inG{\tilde{\mathbf{P}}_{f_{j}}}$ correlates with the
state errors $\tilde{\state}$, and thus we cannot use
$\eqref{eq:residual_approx}$ directly. To remedy this problem [Mourikis07] used
the null space trick to remove the correlation, this was achieved by creating
the left null space $\mathbf{A}^{T}$, a unitary matrix whose columns form the
basis of the left null space of the matrix $\mathbf{H}^{(j)}_{f}$. Once we have
the left null space $\mathbf{A}^{T}$ we can project $\mathbf{r}^{(j)}$ on the
left null space of $\mathbf{H}^{(j)}_{f}$ to form a new residual $r^{(j)}_{o}$.

\begin{equation}
  \mathbf{A}^{T} = \text{Null}(\text{Transpose}(\mathbf{H}^{(j)}_{f}))
\end{equation}

\begin{align}
  \mathbf{A}^{T}
  \underbrace{(z^{(j)} - \hat{z}^{(j)})}_{\residual^{(j)}}
  \simeq&
    \mathbf{A}^{T} \mathbf{H}^{(j)}_{\state} \tilde{\state} +
      \underbrace{\mathbf{A}^{T} \mathbf{H}^{(j)}_{f}}_{
        \mathbf{A}^{T} \mathbf{H}^{(j)}_{f} = \mathbf{0}
      }
      \inG{\tilde{\mathbf{P}}_{f_{j}}} + \mathbf{A}^{T} \mathbf{n}^{(j)} \\
  \residual^{(j)}_{o} = \mathbf{A}^{T} \residual^{(j)}
  \simeq&
    \mathbf{A}^{T} \mathbf{H}^{(j)}_{o} \tilde{\state} +
    \mathbf{0} +
    \mathbf{n}^{(j)}_{o} \\
  \residual^{(j)}_{o} = \mathbf{A}^{T} \residual^{(j)}
  \simeq&
    \mathbf{A}^{T} \mathbf{H}^{(j)}_{o} \tilde{\state} +
    \mathbf{n}^{(j)}_{o}
\end{align}

\begin{align}
  \mathbf{H}^{(j)}_{\mathbf{X}_{i}} &= \begin{bmatrix}
    \dfrac{\partial{z}}{\partial{\state_{\imu}}}
    & \dfrac{\partial{z}}{\partial{\state_{\cam_{0}}}}
    & \cdots
    & \dfrac{\partial{z}}{\partial{\state_{\cam_{N}}}}
  \end{bmatrix} \\
  \mathbf{H}^{(j)}_{\mathbf{X}_{i}} &= \begin{bmatrix}
    \underbrace{0_{2 \times 15}}_\text{w.r.t IMU state}
    & \underbrace{0_{2 \times 6}}_\text{w.r.t 1st Camera state}
    & \cdots &
    & \mathbf{J}^{(j)}_{i}
      \lfloor\! ~^{C_{i}}\!\hat{\state}_{f_{j}} \times \rfloor
    & - \mathbf{J}^{(j)}_{i} C(~^{C_{i}}_{G}\!\hat{\bar{q}})
  \end{bmatrix} \\[2em]
\end{align}


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
    & \underbrace{0_{2 \times 6}}_\text{w.r.t 1st Camera state}
    & \cdots &
    & \mathbf{J}^{(j)}_{i}
      \lfloor\! ~^{C_{i}}\!\hat{\state}_{f_{j}} \times \rfloor
    & - \mathbf{J}^{(j)}_{i} C(~^{C_{i}}_{G}\!\hat{\bar{q}})
  \end{bmatrix} \\[2em]
  % --- z
  z &= h(g(\rot{\cam_{i}}{\global}, \inG{P_{f}}, \inG{P_{C})}) \\
  % --- g
  g &= \rot{\cam_{i}}{\global} (\inG{P_{f}} - \inG{P_{C}}) \\
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
    \rot{\cam_{i}}{\global} \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix},
    & \rot{\cam_{i}}{\global} \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix},
    & \rot{\cam_{i}}{\global} \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}
  \end{bmatrix} \\
  &= \begin{bmatrix}
    1 & 0 & 0 \\
    0 & 1 & 0 \\
    0 & 0 & 1
  \end{bmatrix} \rot{\cam_{i}}{\global}
  = \rot{\cam_{i}}{\global} \\
  % --- dg / d delta theta
  \dfrac{\partial{g}}{\partial{\delta\theta}}
  &= \rothat{\cam_{i}}{\global} (\inG{\hat{P}_{f}} - \inG{\hat{P}_{C}}) \\
  &= \rot{\cam_{i}}{\global} (\I_{3} - \skew{\delta\theta}) (\inG{P_{f}}, \inG{P_{C}}) \\
  &= (\I_{3} - \skew{\delta\theta}) \rot{\cam_{i}}{\global} (\inG{P_{f}}, \inG{P_{C}}) \\
  &= (\I_{3} - \skew{\delta\theta}) \inC{P_{f}} \\
  &= \inC{P_{f}} - \skew{\delta\theta} \inC{P_{f}} \\
  &= \inC{P_{f}} + \skew{\inC{P_{f}}} \delta\theta \\
  &= \skew{\inC{P_{f}}}
\end{align}


[Mourikis07]: A.I. Mourikis, S.I. Roumeliotis: "A Multi-state Constraint Kalman
Filter for Vision-Aided Inertial Navigation," Proceedings of the IEEE
International Conference on Robotics and Automation (ICRA), Rome, Italy, April
10-14 2007, pp. 3565-3572.

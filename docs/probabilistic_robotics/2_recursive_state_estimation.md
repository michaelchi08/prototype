# Recursive State Estimation

At the core of probabilistic robotics is the idea of esitmating state from
sensor data. State estimation addresses the problem of estimating
quantities from sensor data. that are not directly observable, but can be
inferred. In most robotic aplications determining what to do is relatively
easy if one only knew certain quantities. Fro example, moving a mobile
robot is relatively easy if the exact location of the robot and all nearby
obstacles are knonw. State estimation seeks to recover state variables
from data. Probabilistic state estimation algorithms compute belief
distributions over possible world states.








## Robot Environment Interaction

The environment or world of a robot is a dynamical system that possessses
internal state. The robot can acquire information about its environment using
its sensors. However, sensors are noisy, and there are usually many thigns that
cannto be sensed directly. As a consequence, the robot maintainsan internal
belief with regards to the state of its environmen. The robot can also
influence its environment throught its actuators. However, the effect of doing
so is often somewhat unpredictable.



### State

Environments are characterized by **state**, it is convienint to think of them
as the collection of all aspects of the robot and its environment that can
impact the future. State may change over time, such as the location of peopl.
State that changes will be called **dynamic state**, which distinguishes from
**static**, or non-changing state. The state also include variables regarding
the robot itself, such as its pose, velocity, whether or not its sensors are
functioning correctly, and so on.

- **Pose**: comprises location and orientation relative to a global coordinate
  frame.
- **Configuration**: such as the joints of a robot's manuipulators.
- **Velocity**: also commonly referred to as dynamic state.
- **Location** where the robot is in the global environment.
- And more...: there can be a huge number of other state variales, for example,
  whether or not a sensor is broken is a state variable, as is the level of
  battery chrage for a battery powered robot.





## Probabilistic Generative Laws

The evolution of state and measurements is governed by probabilistic laws. In
general, the state at time $x_{t}$ is generated stochastically. Thus it makes
sense to specify the probability distribution from which $x_{t}$ is generated.
At first glance, the emergence of state $x_{t}$ might be conditioned on all
past states, measurements, and controls. Hence, the probabilistic law
characterizing the evolution of state might be given by a probability
distribution of the following form:

\begin{equation}
    p(x_{t} \mid x_{0: t - 1}, z_{1: t - 1}, u_{1: t})
\end{equation}

(Notice that through no particular motivation we assume here that the robot
executes a control action $u_{1}$ first, and then takes a measurement $z_{1}$.)

However, if the state $x$ is complete then it is sufficient summary of all that
happened in previous time steps. In particular, $x_{t -1}$ is a sufficient
statistic of all previous controls and measurementsup to this point, that is
$u_{1: t - 1}$ and $z_{1: t- 1}$. From all the variables in the expression
above, only the control $u_{t}$ matters if we know the state $x_{t - 1}$. In
probabilistic terms, this insight is expressed by the following equality:

\begin{equation}
    p(x_{t} \mid x_{0: t - 1}, z_{1: t - 1}, u_{1: t})
        = p(x_{t} \mid x_{t - 1}, u_{t})
\end{equation}

The property expressed by this equality is an example of **conditional
independence**. It states that certain variables are independent of others if
one knows the values of a third group of variables, the conditioning variables.

Simlarly, one might want to model the process by which measurements are being
generated. Again, if $x_{t}$ is complete, we have an important conditional
independence:

\begin{equation}
    p(x_{t} \mid x_{0: t - 1}, z_{1: t - 1}, u_{1: t})
        = p(z_{t} \mid x_{t})
\end{equation}

In other words, the state $x_{t}$ is sufficient to predict the (potentially
noisy) measurement $z_{t}$. Knowledge of any other variable, such as past
measurements, controls or even past states, is irrelevant if $x_{t}$ is
complete.

This discussion leaves open as to what the two resulting conditional
probabilities are:

- $p(x_{t} \mid x_{t - 1}, u_{t})$
- $p(z_{t} \mid x_{t})$

The probability $p(x_{t} \mid x_{t - 1}, u_{t})$ is the **state transition
probability**. It specifies how environmental state evolves over time as
a function of robot controls $u_{t}$.

The probability $p(z_{t} \mid x_{t})$ is called the **measurement
probability**. It also may not depend on the time index $t$, in which case it
shall be written as $p(z \mid x)$. The measurement probability specifies the
probabilistic law according to which measurements $z$ are generated from the
environment state $x$. Measurements are usually noisy projections of the state.

The state transition probability and the measurement probability together
describe the dynamical stocahstic system of the robot and its environment.




## Belief Distributions

Another key concept in probabilistic robotics is that of a **belief**. A belief
refelcts the robot's internal knowledge about the state of the environment. We
already discussed that state cannot be measured directly.  For example,
a robot's pose might be $x = <14.12, 12.7, 0.755>$ in some global coordinate
system, but it usually cannot know its pose, since poses are not measurable
directly. Instead, the robot must infer its pose from data. We therefore
distinguish the true state from its internal belief, or state of knowledge with
regards to that state.

Probabilistic robotics represents beliefs through conditional probability
distributions. A belief distribution assigns a probability (or density value)
to each possible hypothesis with regards to the true state. Belief
distributions are posterior probabilities over state variables conditioned on
the available data. we will denote belief over a state variable $x_{t}$ by
$\text{bel}(x_{t})$, which is an abbreviation for the posterior:

\begin{equation}
    \text{bel}(x_{t}) = p(x_{t} \mid z_{1: t}, u_{1: t})
\end{equation}

This posterior is the probability distribution over the state $x_{t}$ at time
$t$ conditioned on all past measurements $z_{1:t}$ and all past controls
$u_{t}$. Such a posterior will be denoted as follows:

\begin{equation}
    \overline{\text{bel}}(x_{t}) = p(x_{t} \mid z_{1: t - 1}, u_{1: t})
\end{equation}

This probability distribution is often referred to as the **prediction** in the
context of probabilistic filtering. This terminology refects the fact that
$\overline{\text{bel}}(x_{t})$ predicts the state at time $t$ based on the
previous state posterior, before incorporating the measurement at time $t$.
Calculating $\text{bel}(x_{t})$ from $\overline{\text{bel}}(x_{t})$ is called
**correction** or the **measurement update**.





## The Bayes Filter Algorithm

This is the most general algorithm for calculating belief distribution
$\text{bel}$ from measurement and control data. The Bayes filter is recursive,
that is, the belief $\text{bel}(x_{t})$ at time $t$ is calculated from the
belief $\text{bel}(x_{t - 1})$ at time $t - 1$. Its input is the belief
$\text{bel}$ at time $t - 1$, along with the most recent control $u_{t}$ and
the most recent measurement $z_{t}$. Its output is the belief
$\text{bel}(x_{t})$ at time $t$.

![bayes filter](images/bayes_filter.png)

Bayes filter allow robots to continuously update their most likely position
within a coordinate system, based onthe most recently acquired sensor data.
This is a recursive algorith, it consists of two parts:

- **Prediction**: It processes the control $u_{t}$ by calculating a belief over the
  state $x_{t}$ based on the prior belief over state $x_{t - 1}$ and the
  control $u_{t}$.
- **Innovation**: Also called the measurement update, the algorithm multiplies the
  belief prediction by the probability that the measurement $z_{t}$ may have
  been observed. It does so for each hypothetical posterior state $x_{t}$.

To compute the posterior belief recursively, the algorithm's initial belief
$\text{bel}(x_{0})$ at time $t = 0$ as boundary condition. If one knows the
value of $x_{0}$ with certainty, $\text{bel}(x_{0})$ should be intitialized
with a point mass distribution that centers all probability mass on the correct
value of $x_{0}$, and assigns zero probability anywhere else.

If the variables are linear and normally distributed the Bayes filter becomes
equal to the **Kalman filter**.



### Example

To make this example simple, let us assume that a robot is estimating the state
of a door using its camera. The door can be in one of two possible states open
or closed, and that only the robot can change the state of the door. Let us
furthermore asume that the robot does not know the state of the door intially.
Instead it assigns equal prior probability to the two possible door states:

\begin{align}
    \text{bel}(X_{0} = \textbf{open}) &= 0.5 \\
    \text{bel}(X_{0} = \textbf{closed}) &= 0.5
\end{align}

Let us furthermore assume the robot's sensors are noisy.

\begin{align}
    p(Z_{t} = \textbf{sense_open} \mid X_{t} = \textbf{is_open}) &= 0.6 \\
    p(Z_{t} = \textbf{sense_closed} \mid X_{t} = \textbf{is_open}) &= 0.4 \\
    p(Z_{t} = \textbf{sense_open} \mid X_{t} = \textbf{is_closed}) &= 0.2 \\
    p(Z_{t} = \textbf{sense_closed} \mid X_{t} = \textbf{is_closed}) &= 0.8
\end{align}

These probabilities suggest that teh robot's sensors are relatively reliable in
detecting a **closed door**, in that the error probability is 0.2. However, when
the door is open, it has a 0.4 probability of a false measurement.

Lets assume that the robot has an arm to push the door open. If the door is
already open, it will remain open. If it is closed the robot has 0.8 chance
that it will open afterwards:


\begin{align}
    p(X_{t} = \textbf{is_open} \mid
        U_{t} = \textbf{push},
        X_{t - 1} = \textbf{is_open})
            &= 1.0 \\
    p(X_{t} = \textbf{is_closed} \mid
        U_{t} = \textbf{push},
        X_{t - 1} = \textbf{is_open})
            &= 0.0 \\
    p(X_{t} = \textbf{is_open} \mid
        U_{t} = \textbf{push},
        X_{t - 1} = \textbf{is_closed})
            &= 0.8 \\
    p(X_{t} = \textbf{is_closed} \mid
        U_{t} = \textbf{push},
        X_{t - 1} = \textbf{is_closed})
            &= 0.2
\end{align}

It can also choose not to use its arm, in which case the state of the world
does not change:


\begin{align}
    p(X_{t} = \textbf{is_open} \mid
        U_{t} = \textbf{do_nothing},
        X_{t - 1} = \textbf{is_open})
            &= 1.0 \\
    p(X_{t} = \textbf{is_closed} \mid
        U_{t} = \textbf{do_nothing},
        X_{t - 1} = \textbf{is_open})
            &= 0.0 \\
    p(X_{t} = \textbf{is_open} \mid
        U_{t} = \textbf{do_nothing},
        X_{t - 1} = \textbf{is_closed})
            &= 0.0 \\
    p(X_{t} = \textbf{is_closed} \mid
        U_{t} = \textbf{do_nothing},
        X_{t - 1} = \textbf{is_closed})
            &= 1.0
\end{align}


Suppose at time $t$, the robot takes no control action but it sense an open
door. The resulting posterior belief is calcuated by the Bayes filter using the
prior belief $\text{bel}(X_{0})$, the control $u_{1} = \textbf{do_nothing}$,
and the measurement $\textbf{sense_open}$ as input. Since hte state space is
finite, the integral turns into a finite sum.

\begin{align}
    \overline{\text{bel}}(x_{1})
        &= \int p(x_{1} \mid u_{1}, x_{0}) \text{bel}(x_{0}) dx_{0} \\
        &= \sum_{x_{0}} p(x_{1} \mid u_{1}, x_{0}) \text{bel}(x_{0}) \\
        &= p(x_{1} \mid U_{1} = \textbf{do_nothing}, X_{0} = \textbf{is_open})
            \text{bel}(X_{0} = \textbf{is_open}) \\
        &+ p(x_{1} \mid U_{1} = \textbf{do_nothing}, X_{0} = \textbf{is_closed})
            \text{bel}(X_{0} = \textbf{is_closed})
\end{align}

We can now substitute the two possible values for hte state variable $X_{1}$
for the hypothesis $X_{1} = \textbf{is_open}$, we obtain:

\begin{align}
    \overline{\text{bel}}(X_{1} = \textbf{is_open})
        &= 1 \cdot 0.5 + 0.0 \cdot 0.5 \\
        &= 0.5
\end{align}

The fact that the belief $\overline{\text{bel}}(x_{1})$ equals our prior belief
$\text{bel}(x_{0})$ should not be surprising, as the action
$\textbf{do_nothing}$ does not affect the state of the world; neither does the
world change over time by itself in our example.

Incorporating measurement, however, changes the belief.

\begin{equation}
    \text{bel}(x_{1}) =
        \eta p(Z_{1} = \textbf{sense_open} \mid x_{1})
        \overline{\text{bel}}(x_{1})
\end{equation}

For the two possible case, $X_{1} = \textbf{is_open}$ and $X_{1}
= \textbf{is_closed}$, we get

\begin{align}
    \text{bel}(X_{1} = \textbf{is_open}) &=
        \eta p(Z_{1} = \textbf{sense_open} \mid X_{1} = \textbf{is_closed})
        \overline{\text{bel}}(X_{1}) \\
        &= \eta 0.6 \cdot 0.5 = \eta 0.3
\end{align}

and

\begin{align}
    \text{bel}(X_{1} = \textbf{is_closed}) &=
        \eta p(Z_{1} = \textbf{sense_open} \mid X_{1} = \textbf{is_closed})
        \overline{\text{bel}}(X_{1}) \\
        &= \eta 0.2 \cdot 0.5 = \eta 0.1
\end{align}

The normalizer $\eta$ is now easily calculated:

\begin{equation}
    \eta = (0.3 + 0.1)^{-1} = 2.5
\end{equation}

Hence, we have

\begin{align}
    \text{bel}(X_{1} = \textbf{is_open}) = 0.75 \\
    \text{bel}(X_{1} = \textbf{is_closed}) = 0.25
\end{align}

This calculation is now easily iterated for the next time step. As the reader
readily verifies, for $u_{2} = \textbf{push}$, $z_{2} = \textbf{sense_open}$ we
get:

\begin{align}
    \overline{\text{bel}}(X_{2}) = 1 \cdot 0.75 + 0.8 \cdot 0.25 = 0.95 \\
    \overline{\text{bel}}(X_{2}) = 1 \cdot 0.75 + 0.2 \cdot 0.25 = 0.05
\end{align}

and

\begin{align}
    \text{bel}(X_{2}) = \eta 0.6 \cdot 0.95 \approx 0.983 \\
    \text{bel}(X_{2}) = \eta 0.2 \cdot 0.05 \approx 0.017
\end{align}

At this point, the robot believes that with 0.983 probability that the door is
open, hence both its measurements were correct. At first glance, this
probability may appear to be sufficiently high to simply accept this
hypothesis. However, such an approach may result in unnecessarily high costs.
If mistaking a closed door for an open one incurs cost, considering both
hypotheses in the decision making will be essential. Image flying an aircraft
on auto pilot with perceived chance of 0.983 not crashing.



### Mathematical Derivation of the Bayes Filter

The following is a proof by induction. First we show it correctly calculates
the posterior distribution $p(x_{t} \mid z_{1: t}, u_{1: t})$ from the
corresponding posterior one time step earlier, $p(x_{t - 1} \mid z_{1: t - 1},
u_{1: t - 1})$. Lets assume we correctly initialized the prior belief
$\text{bel}(x_{0})$ at time $t = 0$.

Our derivation requires that the state $x_{t}$ is complete, and it requires
that the controls are chosen at random. The first step of our derivation
involves the application of Bayes rule to the target posterior.

\begin{align}
    \label{eq:bayes_posterior distribution}
    p(x_{t} \mid z_{1: t}, u_{1: t})
            &= \frac{
                p(z_{t} \mid x_{t}, z_{1: t - 1}, u_{1: t})
                p(x_{t} \mid z_{1: t - 1}, u_{1: t})
            }{
                p(z_{t} \mid z_{1: t - 1}, u_{1: t})
            } \\
            &= \eta p(z_{t} \mid x_{t}, z{1: t - 1}, u_{1: t})
                p(x_{t} \mid z_{1: t - 1}, u_{1: t})
\end{align}

We now exploit the assumption that our state is complete. Previously we defined
a state $x_{t}$ to be complete if no variables prior to $x_{t}$ may influence
the stochastic evolution of future states. In particular, if we
(hypothetically) knew the state $x_{t}$ and were interested in predicting the
measurement $z_{t}$, no past measurement or control would provide us additional
information. In mathematical terms, this is expressed by the following
conditional independence:

\begin{equation}
    p(z_{t} \mid x_{t}, z_{1: t - 1}, u_{1: t}) = p(z_{t} \mid x_{t})
\end{equation}

Such a statement is another example of conditional independence. It allows us
to simplify (\ref{eq:bayes_posterior distribution}) as:

\begin{equation}
    p(x_{t} \mid z_{1: t}, u_{1: t}) =
        \eta p(z_{t} | x_{t} \mid z_{1: t - 1}, u_{1: t})
\end{equation}

and hence the update of the Bayes filter is:

\begin{equation}
    \text{bel}(x_{t}) = \eta p(z_{t} \mid x_{t}) \overline{\text{bel}}(x_{t})
\end{equation}

The term $\overline{\text{bel}}(x_{t})$ can be expanded using the continuous
case for the total probability:

\begin{align}
    \overline{\text{bel}}
        &= p(x_{t} \mid z_{1: t - 1}, u_{1: t}) \\
        &= \int p(x_{t} \mid x_{t - 1}, z_{1: t - 1}, u_{1: t})
            p(x_{t} \mid z_{1: t - 1}, u_{1: t}) dx_{t - 1}
\end{align}

# Uncertainty in Robotics

Robotics is the science of perceiving and manipulating the physical world
through computer-controlled mechanical devices. Examples of successful
robotic systems include mobile platforms for planetary exploration,
robotic arms in assembly lines and more.

Uncertainty arises if the robot lacks critical information for carrying out its
task. It can arise from five different factors:

1. Environment
2. Sensors
3. Robots
4. Models
5. Computation

Traditionally such uncertainty has mostly been ignored in robotics.
However, as robots are moving away from factory floors to increasingly
unstructured environments the ability to cope is critical.

Probabilistic robotics is a new approach to robotics the pays tribute to
uncertainty in robot perception and action. The key idea of probabilitic
robotics is to represent uncertainty explicitly, using the calculus of
probability theory. Put differently, instead of relying on a single "best
guess" as to what might be the case in the world, probabilistic algorithms
represent information by probability distribution over a whole space of
possible hypothesis. By doing so, they can represent ambiguity and degree
of belief in a mathematically sound way.

An example is a localization problem called global localization. When
a robot is placed in an environment and has to localize itself from
scratch, the robot's momentary estimate (aka belief) is represented by
a probability dense function over the space of all locations. The robot
then based on the information from sensors deals with conflicting
hypothesis that arises from ambigous situations.

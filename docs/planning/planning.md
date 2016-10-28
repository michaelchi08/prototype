# Planning

## Constraints

- **Occupancy**
- **Dyanmics**

## Approaches

- **Reactive** (Local approach): decide a direction to go based on goal
  and obstacles whilst ignoring vehicle dynamics. These approaches are
  usually deterministic.
- **Graph Based** (Global approach): Graph extraced from workspace
  definition. Graph generate by ranom sampling of nodes and random
  connections between nodes.
- **Optimal** (Global approach): Find complete path to goal whilst
  incorporating constraints, but may need to model a certain way.

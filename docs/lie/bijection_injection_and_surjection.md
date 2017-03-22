# Injection, Surjection and Bijection

Injections, surjections and bijections are classes of functions distinguished
by the manner in which arguments (input expressions from the domain) and images
(output expressions from the codomain) are related or mapped to each other.

A function maps elements from its domain to elements in its codomain. Given a
function $f: X \mapsto Y$, the function is :

- **Injective (one-to-one)**: if every element of the codomain is mapped to by
  at most one element of the domain. An injective function is an **injection**.
  Notationally: $\forall x, \prime{x} \in X, f(x) = f(\prime{x}) \rightarrow x
  = \prime{x}$
- **Surjective (onto)**: if every element of the codomain is mapped to by at
  least one element of the domain. (That is, the image and the codomain of the
  function are equal). A surjective function is **surjection**. Notationally:
  $\forall y \in Y, \exists x \in X \text{s.t.} y = f(x)$.
- **Bijective (one-to-one and onto or one-to-one correspondance)**: if every
  element of the codomain is mapped to by exactly one element of the domain.
  (That is the function is both injective and surjective) A bijective function
  is **bijection**.

An injective function need not be surjective (not all elements of the codomain
may be associated with arguments), and a surjective function need not be
injective (some images may be associated with more than one argument). The four
possible combinations of injective and surjective features are illustrated in
the diagrams to the right.

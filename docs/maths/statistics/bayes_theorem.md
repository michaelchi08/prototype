# Bayes Theorem

Bayes' Theorem describes the probability of an event based on conditions that
might be related to the event. When applied, the probabilities involve in
Bayes' theorem may have different probability interpretations. In one of these
interpretations, the theorem is used directly as part of a particular approach
to statistical inference. With the Bayesian probability interpretation the
theorem expresses how a subjective degree of belief should rationally change to
account for evidence: this is Bayesian inference, which is fundamental to
Bayesian statistics.

The Bayes' theorem is stated mathematically as follows:

\begin{equation}
    p(A \mid B) = \frac{p(B \mid A) p(A)}{p(B)}
\end{equation}

where:

- $A$ and $B$ are events
- $p(A)$ and $p(B)$ are probabilities of $A$ and $B$ without regard to each
  other
- $p(A \mid B)$ is a conditional probability, it is the probability of observing
  event $A$ given that $B$ is true
- $p(B \mid A)$ is the probability of observing event $B$ given that $A$ is true


**Discrete Bayes Rule**

\begin{equation}
    p(x \mid y) = \frac{p(y \mid x) p(x)}{p(y)}
        = \frac{p(y \mid x) p(x)}{p(y)}
        = \frac{p(y \mid x) p(x)}{\sum_{x'} p(y \mid x') p(x')}
\end{equation}

**Continuous Bayes Rule**

\begin{equation}
    p(x \mid y) = \frac{p(y \mid x) p(x)}{p(y)}
        = \frac{p(y \mid x) p(x)}{p(y)}
        = \frac{p(y \mid x) p(x)}{\int p(y \mid x') p(x') dx'}
\end{equation}

The Bayes rule plays a predominant role in probabilistic robotics. If $x$
is a quantity that we would like to infer from $y$, the probability of
$p(x)$ will be referred to as the **prior probability distribution**, and
y is called the **data** (e.g. sensor measurement). The distribution
$p(x)$ summarizes the knowledge we have regarding $X$ prior to
incorporating the data $y$.

The probability $p(x \mid y)$ is called the **posterior probability
distribution** over $X$. If we look at the continuous Bayes rule it
provides a convenient way to compute a posterior $p(x \mid y)$ using the
**inverse** conditional probability $p(y \mid x)$ along with prior
probability $p(x)$. In other words, if we are interested in inferring
a quantity $x$ from sensor data $y$, Bayes rule allows us to do so through
the inverse proabbility, whic hspecifies the probability of data $y$
assuming that $x$ was the case.

In robotics this inverse probaility $p(y \mid x)$ is often coined "generative
model", since it describes, at some level of abstraction how state
variables $X$ cause sensor measurements $Y$.





## Example: Drug Testing

Suppose a drug test is $99\%$ sensitive and $99\%$ specific. That is the
test will produce $99\%$ true positive results for drug users and $99\%$
true negative results for non-drug users. Suppose that $0.5\%$ of people
are users of the drug. If a randomly selected individual tests positive,
what is the probability he or she is a user?

---

From the above information we can deduce that

- Employees are either users or non-users

\begin{equation}
    X = \{ \text{users}, \text{non-users} \}
\end{equation}

- Test is either postive or negative

\begin{equation}
    Y = \{ +, - \}
\end{equation}

We want to find the probability that an employee is a user given the test is
positive. Using Bayes' Theorem:

\begin{align}
    p(\text{user} \mid \text{+}) &=
        \frac
        {
            \text{likelihood} \cdot \text{prior}
        }
        {
            \text{evidence}
        } \\
    &=
        \frac
        {
            p(\text{+} \mid \text{user}) p(\text{user})
        }
        {
            p(\text{+})
        } \\
\end{align}

This is where things get interesting because we know:

- $p(\text{+} \mid \text{user})$ is $0.99$.
- $p(\text{user})$ is $0.005$.
- **but** $p(\text{+})$ is a little more involving, we need to find that out.

To find $p(\text{+})$ we have to find its **true positive and false positive
probabilities**:

\begin{align}
    p(\text{+})
        &=
            \text{true positive %} + \text{false positive %} \\
        &=
            p(\text{+} \mid \text{user}) p(\text{user}) +
            p(\text{+} \mid \text{non-user}) p(\text{non-user})
\end{align}
 
putting that back into the Bayes' theorem:

\begin{align}
    p(\text{user} \mid \text{+}) &=
        \frac
        {
            p(\text{+} \mid \text{user}) p(\text{user})
        }
        {
            p(\text{+} \mid \text{user}) p(\text{user}) +
            p(\text{+} \mid \text{non-user}) p(\text{non-user})
        } \\
    &=
        \frac
        {
            0.99 \times 0.005
        }
        {
            0.99 \times 0.005 + 0.01 \times 0.995
        } \\
    &\approx 33.2 \%
\end{align}


Despite the apparent accuracy of the test, if an individual tests
positive, it is more likely that they do not use the drug than that they
do. This again illustrates the importance of base rates, and how the
formation of policy can be egregiously misguided if base rates are
neglected.

This surprising result arises because the number of non-users is very
large compared to the number of users; thus the number of false positives
($0.995\%$) outweighs the number of true positives ($0.495\%$). To use
concrete numbers, if 1000 individuals are tested, there are expected to be
995 non-users and 5 users. From the 995 non-users, $0.01 \times 995 = 10$
false positives are expected. From the $5$ users, $0.99 \times 5 = 5$ true
positives are expected. Out of %15% positive results, only %5%, about
$33\%$ are genuine.

In summary:

- $33\%$ chance of that positive test result has cuahg a drug user. That is not
  a great test!
- Difficulty lies in the large number of non-drug users that are tested

# Basic Concepts in Probability

Let $X$ denote a random variable and $x$ denote a specific event that $X$
might take on.

\begin{equation}
    p(X = x)
\end{equation}

the above denotes the probability that the random variable $X$ has value
$x$. For example, a fair coin is characterized by $p(X = \text{head})
= p(X = \text{tail}) = 0.5$. All discrete probabilities should sum to 1.

\begin{equation}
    \sum_{x} p(X = x) = 1
\end{equation}

A common density function is that of the one-dimensional and multi-dimensional
Normal Distribution.


### **Single Variate Gausian Function**:

Mean $\mu$ and variance $\sigma^{2}$

\begin{equation}
    p(x) = (2 \pi \sigma^{2})^{-0.5}
        \exp
            \big\{
                -\frac{1}{2} \frac{(x - \mu)^{2}}{\sigma^{2}}
            \big\}
\end{equation}


### **Multi-Variate Normal Distribution Gaussian Function**:

Mean vector $\mu$ and $\Sigma$ is a positive semidefinite symmetric matrix
called a covariance matrix.

\begin{equation}
    p(x) = \text{det}(2 \pi \Sigma)^{-0.5}
        \exp
            \big\{
                -\frac{1}{2} (x - \mu)^{T}
                \Sigma^{-2} (x - \mu)^{T}
            \big\}
\end{equation}








### Normalizer Variable

An important observation is that the denominator of Bayes rule, $p(y)$,
does nto depend on $x$. This, the factor $p(y)^{-1}$ will be the same for
any value of $x$ in teh posterior $p(x \mid y)$. For thsi reason $p(y)^{-1}$
is often written as a **normalizer variable** denoted with $\eta$:

\begin{equation}
    p(x \mid y) = \eta p(y \mid x) p(x)
\end{equation}

Or if $X$ is discrete, equations of this type can be computed as:

\begin{equation}
    \forall x : \text{aux}_{x \mid y} = \eta p(y \mid x) p(x)
\end{equation}

\begin{equation}
    \text{aux}_{y} = \sum_{x} \text{aux}_{x \mid y}
\end{equation}

\begin{equation}
    \forall x : p(x \mid y) = \frac{\text{aux}_{x \mid y}}{\text{aux}_{y}}
\end{equation}

where $\text{aux}_{x \mid y}$ are auxiliary variables. These instructions
effectively calcualte  $p(x \mid y)$, but instead of explicitly computing
$p(y)$, they instead just normalize the result.



### Expectation of a Random Variable $X$

The expected value of a random variable is intuitively the long-run average
value of repetitions of the experiment it represents. For example, the expected
value of a six-sided die roll is 3.5 because, roughly speaking, the average of
an extremely large number of die rolls is practically always nearly equal to
3.5. Less roughly, the law of large numbers guarantees that the arithemetic
mean of the values almost surely converges to the expected value as the number
of repetitions goes to infinitiy. The expected value is also known as the mean.

\begin{equation}
    E[X] = \sum_{x} x p(x)
\end{equation}

\begin{equation}
    E[X] = \int x p(x) dx
\end{equation}

Not all random variables possess finite expectations. The expectation is
a linear function of a random variable. In particular:

\begin{equation}
    E[aX + b] = aE[X] + b
\end{equation}

for arbitrary numerical values $a$ and $b$. The covariance of $X$ is obtained
as follows:

\begin{equation}
    Cov[X] = E[X - E[X]]^{2} = E[X^{2}] - E[X]^{2}
\end{equation}

The covariance measures the squared expected deviation from the mean. As
stated above, the mean of a multivariate normal distribution $\mathcal{N}(x;
\mu, \Sigma)$ is $\mu$, and its covariance is $\Sigma$.



### Entropy

Another important characteristic of a random variable is its **entropy**.
For discrete random variables, the entropy is given by the following
expression:

\begin{equation}
    H(P) = E[-log_{2} p(x)] = -\sum_{x} p(x) log_{2} p(x)
\end{equation}

The concept of entropy originates in information theory. The entropy is the
expected information that the value of $x$ carries: $-log_{2} p(x)$ is the
number of bits required to encode $x$ using optimal encoding.

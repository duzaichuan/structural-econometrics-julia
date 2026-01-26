# Job Search Model

## Overview

This section presents a simple model of undirected job search. The model demonstrates how workers optimally choose which job offers to accept based on a reservation wage strategy.


## Economic Environment

Time is discrete and indexed by $t$ over an infinite horizon. Workers move between employment and unemployment, have linear utility, and cannot save.

### Parameters

| Parameter | Description |
| --------- | ------------|
| $\lambda$ | The probability an unemployed worker receives a job offer |
| $\delta$  | The probability an employed worker loses their job |
| $F_{W}$   | The distribution of wage offers |
| $1-\beta$   | The exponential rate of discounting |
| $b$       | Per-period utility when unemployed |

## Recursive Formulation

The classic approach to solve this model is to write the values of unemployment and employment recursively. For example:

$$ U = b + \beta[(1-\lambda)U + \lambda\int\max\{V(w),U\}dF_{W}(w)] $$
$$ V(w) = w + \beta[(1-\delta)V(w) + \delta U] $$

## Model solution

One can show that the optimal decision rule of the worker is characterized by a reservation wage $w^*$, defined as $V(w^*)=U$. We can also differentiate the expression for $V(w)$ to get:

$$ V'(w) = \frac{1}{1 - \beta(1-\delta)} $$

and applying integration by parts gives:

$$ U = b + \beta[U + \lambda\int_{w^*}\frac{1-F_{W}(w)}{1-\beta(1-\delta)}dw] $$

Now applying the definition of the reservation wage gives the reservation wage equation:

$$ w^* = b + \beta\lambda\int_{w^*}\frac{1-F_{W}(w)}{1 - \beta(1-\delta)}dw $$

and we can characterize the steady state rate of unemployment as:

$$ P[E = 0] = \frac{h}{h+\delta} $$

where $h = \lambda(1-F_{W}(w^*))$ is the rate at which workers exit unemployment.

Similarly, we can show that the steady state fraction of unemployment durations $t_{U}$ is

$$ P[t_{U}=t] = h(1-h)^{t} $$

## Numerical Model Solution

To solve the reservation wage equation numerically, we need to evaluate the integral on the right-hand side and find the value of $w^*$ that satisfies the equation. This requires two key numerical methods: quadrature (for integration) and root-finding.

### Gauss-Legendre Quadrature

When integrating numerically, we approximate the integral using a weighted sum at specific evaluation points (nodes):

$$ \int_a^b f(x)dx \approx \frac{b-a}{2}\sum_{k=1}^n w_k f\left(\frac{a+b}{2} + \frac{b-a}{2}x_k\right) $$

where $x_k$ are the nodes and $w_k$ are the weights from Gauss-Legendre quadrature. This method is particularly accurate for smooth functions and uses a fixed number of nodes, which is important for automatic differentiation (unlike adaptive methods like in the package `QuadGK` that adjust the number of nodes based on the integrand).

### Root Finding

The reservation wage $w^*$ is the value that makes the reservation wage equation equal to zero. We use the `Roots.jl` package, which implements efficient root-finding algorithms based on combinations of bisection, secant, and inverse quadratic interpolation methods.

The `find_zero` function takes:

- A function to find the root of
- An initial guess
- The type of the initial guess (to ensure type stability)

This approach has the advantage of being compatible with automatic differentiation tools like `ForwardDiff`, which is a very useful tool in numerical methods.

### Steady-State Statistics

Using the computed reservation wage, we can calculate the steady-state unemployment rate and average duration.

---

## Further Reading

- **McCall (1970)**: "Economics of Information and Job Search" - Original search model
- **Wolpin (1987)**: "Estimating a Structural Search Model" - Early structural estimation
- **Eckstein and van den Berg (2007)**: "Empirical Labor Search" - Survey of search models
- **Flinn and Heckman (1982)**: "New Methods for Analyzing Structural Models of Labor Force Dynamics" - Duration data analysis

\newpage

# A Life-Cycle Savings Model

## Overview

This section presents a stylized life-cycle model of consumption and savings. Households make dynamic decisions about consumption and asset accumulation over their lifetime, facing income uncertainty and borrowing constraints.

## Economic Environment

Time is discrete and indexed by $t$. Individuals live for a finite number of periods, $T$. They derive utility from consumption according to a CRRA utility function:

$$ u(c) = \frac{c^{1-\sigma}}{1-\sigma} $$

and from "bequests", which are modeled here as cash on hand net of consumption in the final period:

$$ \nu(a) = \psi \frac{a^{1-\sigma}}{1-\sigma} $$.

Consumption can be transferred between periods via a portfolio of one-period bonds ("savings', $a$) that can be purchased at the price $1 / (1+r)$, with a prdetermined limit, $\underline{a}$, on borrowing.

Inviduals receive income $y$ every period that is governed by a deterministic ($\mu_{t}$) and stochastic component:

$$ \log(y_{t}) = \mu_{t} + \varepsilon_{it} $$

where $\varepsilon_{it}$ is a first-order Markov process. A particular case of interest is the case where $\varepsilon$ is a stationary AR 1 process:

$$ \varepsilon_{it} = \rho \varepsilon_{it-1} + \eta_{it} $$

where $\eta_{it} \sim \mathcal{N}(0,\sigma^2_{\eta})$. The unconditional variance of $\varepsilon_{it}$ is therefore $\sigma^2_{\eta} / (1-\rho^2)$.

## Model Solution

Define

$$ V_{T}(a,\varepsilon) = \max_{c}\left\{u(c) + \nu(y + a - c)\right\} $$

And now define the remaining value functions recursively:

$$ V_{t}(a,\varepsilon) = \max_{c,a'}\left\{u(c) + \beta\mathbb{E}_{\varepsilon'|\varepsilon}V(a',\varepsilon')\right\} $$

subject to:

$$ c + \frac{1}{1+r}a' \leq y + a $$

and

$$ a' \geq \underline{a}$$

where $\underline{a}$ is the borrowing constraint.

We're going to write code to solve the model naively using this recursive formulation. You may already be aware that there are more efficient solution methods that exploit the first order conditions of the problem. Not the focus of our class! Please don't use the example below as a demonstration of best practice when it comes to solving savings models.

You can see that the discreteness creates some jumpiness in the policy functions. As I said, other solution methods that use interpolation can be more efficient and will create smoother pictures, but since that is not the focus of this class we will use this simple solution method.

\newpage

# Firm Entry-Exit Model

## Overview

This section presents a symmetric duopoly model of firm entry and exit decisions. Firms make discrete choices about market participation based on profitability and fixed costs. This model illustrates static discrete choice with strategic interactions and is used in Chapter 5 to demonstrate discrete choice estimation methods.

---


## Model Ingredients

Here are the basic ingredients of the model:

- There are two firms indexed by $f\in\{0,1\}$
- There are $M$ markets indexed by $m$
- Time is discrete and indexed by $t$
- Each firm makes an *entry decision* every period. We let $d\in\{0,1\}$ index this decision to enter or not. Let $d(f,m,t)$ indicate the choice of firm $f$ in market $m$ in period $t$.
- We let $a_{f,m,t}=d(f,m,t-1)$ indicate whether firm $f$ is active in market $m$ in period $t$, which means they entered in the previous period.
- Let $x_{m}$ be a market-level observable that shifts the profitability of operations in market $m$.
- In addition to the observed states, each firm draws a pair of idiosyncatic shocks to payoffs in each period, $\epsilon_{f}=[\epsilon_{f0},\epsilon_{f1}]$ that is *private information* to the firm and is iid over markets, firms, and time periods.
- Firms make their decisions in each period *simultaneously*

To simplify notation, suppress dependance of outcomes on the market $m$ and time period $t$. Because we are writing a symmetric model, we will also suppress dependence on $f$. The *deterministic* component of the payoff to entering is a function of the market primitives ($x$), the firm's activity status ($a$), and the other firm's entry decision $d^\prime$:

$$ u_{1}(x,a,d^{\prime}) = \phi_{0} + \phi_{1}x - \phi_{2}d^\prime - \phi_{3}(1-a) $$

The payoff to not entering is simply:

$${u}_{0}(x,a) = \phi_{4}a $$

## Solving the firm's problem

Let $d^*(x,a,a',\epsilon)$ be the firm's optimal decision given the state and the idiosyncratic shock. We will focus on *symmetric equilibria* so this policy function is sufficient to describe the behavior of both firms.

The value to either firm of arriving in a period with state $(x,a,a')$ can be written recursively as:

$$
\begin{aligned}
V(x,a,a') = \mathbb{E}_{\epsilon,\epsilon'}\max\big\{
    &u_{1}(x,a,d^*(x,a,a',\epsilon'))+\epsilon_{1} \\
    &\quad + \beta V(x,1,d^*(x,a,a',\epsilon')), \\
    &u_{0}(x,a) + \epsilon_{0} \\
    &\quad + \beta V(x,0,d^*(x,a,a',\epsilon'))\big\}
\end{aligned}
$$

Define the optimal choice probability in equilibrium as:

$$ p(x,a,a') = \int_{\epsilon}d^*(x,a,a',\epsilon)dF(\epsilon) $$

With this in hand we can integrate out the other firm's shocks $\epsilon'$ to get:

$$
\begin{aligned}
V(x,a,a') = \mathbb{E}_{\epsilon}\max\big\{
    &\phi_{0}+\phi_{1}x - \phi_{2}p(x,a',a) +\epsilon_{1} \\
    &\quad + \beta \big[p(x,a',a)V(x,1,1) + (1-p(x,a',a))V(x,1,0)\big], \\
    &a \phi_{4} + \epsilon_{0} \\
    &\quad + \beta \big[p(x,a',a)V(x,0,1) + (1-p(x,a',a))V(x,0,0)\big]\big\}
\end{aligned}
$$

Define the choice-specific values as:

$$
\begin{aligned}
v_{1}(x,a,a') = \phi_{0}+\phi_{1}x &- \phi_{2}p(x,a',a) \\
    &+ \beta \big[p(x,a',a)V(x,1,1) + (1-p(x,a',a))V(x,1,0)\big]
\end{aligned}
$$

and

$$
\begin{aligned}
v_{0}(x,a,a') = a \phi_{4} + \beta \big[p(x,a',a)V(x,0,1) + (1-p(x,a',a))V(x,0,0)\big]
\end{aligned}
$$

So assuming that $\epsilon$ is distributed as type I extreme value random variable with location parameter 0 and scale parameter 1 we get analytical expressions for the choice probabilities and the expected value of the maximum:

$$ V(x) = \gamma + \log\left(\exp(v_{0}(x,a,a'))+\exp(v_{1}(x,a,a'))\right)$$

where $\gamma$ is the Euler-Mascheroni constant and

$$ p(x,a,a') = \frac{\exp(v_{1}(x,a,a'))}{\exp(v_{0}(x,a,a'))+\exp(v_{1}(x,a,a'))} $$

## Equilibrium

The solution concept for this model is *Markov Perfect Equilibrium*. Fixing the market $x$, here the equilibrium be characterized as a fixed point in the value function $V$ and choice probabilities, $p$. In words, equilibrium is summarized by a $V$ and a $p$ such that:

1. Given $p$, $V$ is a fixed point in the recursive formulation of values; and
2. $p$ are the optimal choice probabilities of each firm given $V$ and given the other firm's choice probabilities are $p$.

In principle we could iterate on this mapping to find (for a fixed $p$), the firm's optimal solution. But that won't be an efficient way to try and solve for the equilibrium.

This seems to work! But notice that it takes a while for the iteration to converge. Also, unlike the single agent case, there is no guarantee that this iteration is always a contraction.

We can also solve this model relatively easily using Newton's Method and the magic of Automatic Differentiation.

In this case Newton's method is faster.

We can re-use this code when we get to thinking about estimation later on. To do this we will have to solve the model for different values of $x_{m}$, but that can be done by using this code and iterating (potentially in parallel) over different values of $x$.

If you play around with parameters, you will see how convergence times may change and that solution methods are not always stable, especially when choice probabilities in equilibrium are very close to one or zero.

\newpage

# A Simple Labor Supply Model

## Model Setup and Solution

Consider a dynamic labor supply model (with no uncertainty) where each agent $n$ chooses a sequence of consumption and hours, $\{c_{t},h_{t}\}_{t=1}^{\infty}$, to solve:
$$ \max \sum_{t=0}^\infty \beta^{t} \left(\frac{c_{t}^{1-\sigma}}{1-\sigma} - \frac{\alpha_{n}^{-1}}{1 + 1/\psi}h_{t}^{1+1/\psi}\right)$$
subject to the intertemporal budget constraint:
$$ \sum_{t}q_{t}c_{t} \leq A_{n,0} + \sum_{t}q_{t}W_{n,t}h_{t},\qquad q_{t} = (1+r)^{-t}.$$
Let $H_{n,t}$ and $C_{n,t}$ be the realizations of labor supply for agent $n$ at time $t$. Labor supply in this model obeys:
$$H_{n,t}^{1/\psi} = (\alpha_{n}W_{n,t})C^{-\sigma}_{n,t}.$$
To simplify below, assume that $\beta=(1+r)^{-1}$, so that the optimal solution features perfectly smoothed consumption, $C^*_{n}$. Making appropriate substitutions gives $C^*_{n}$ as the solution to:
$$ \left(\sum_{t}q_{t}\right)C^*_{n} = \sum_{t}\left(q_{t}W_{n,t}^{1+\psi}\right)\alpha_{n}^{\psi}(C_{n}^*)^{-\psi\sigma} + A_{n,0}.$$

## Code to solve the model

There is only one object to solve here which is consumption given a sequence of net wages. If one were to assume also constant wages the function solves optimal consumption.

## Code to simulate a cross-section

Here we'll assume that wages, tastes for work, and assets co-vary systematically. For simplicity we'll use a multivariate log-normal distribution.

\newpage

# How and Why to Use Models

Before we dive into methods, it will be helpful to review a number of important use cases for quantitative economic models. Why use a model? This might seem like a strange question to devote time to, given that there is no such thing as economics without them. Still, when you are out in the world presenting your research, you may sometimes encounter this question. Indeed, you will read many important and useful applied papers that have no need of an economic model, so it's not so surprising to think that there are research questions that do not demand a structural estimation exercise. What is my model for? How is it central to answering my research question?


## When you possibly don't need a model.

A lot of students, when they write their first paper, end up posing simple questions like "What is the effect of policy $X$ on outcome $Y$"? Often, answering these questions requires nothing more than simple statistical models of causality (such as the Potential Outcomes Model or the Generalized Roy Model, which formalize causality in terms of potential outcomes).

In particular, if your question concerns the causal effect of an observed historical change in some variable, you likely won't anything more elaborate than these simple frameworks. Consider below three examples of quasi-experimental variation from our prototype models that answer particular research questions without needing specification of the underlying models.

### Social Security

In the life-cycle savings model, consider the inclusion of a social security system that provides income at older ages. The budget constraint becomes:

$$ c_{t} + a_{t+1}/(1 + r) \leq (1-\tau)y_{t} + \mathbf{1}\{t\geq 65\}b $$

so that individuals become eligible for a benefit, $b$, after age 65, and pay into the system with proportional taxes, $\tau$.

Suppose that the age of eligibility for social security is decreased, unexpectedly, from 65 to 60. Suppose also you have data on consumption for two cohorts, $A$, and $B$. Cohort $A$ was not exposed to this change (they reached age 65 before becoming eligible), while cohort $B$ is exactly aged 60 when the change is announced. **Suppose your question is**: *"What was the effect of this eligibility expansion on consumption of cohort $B$ at each age after 60-64?"* This question can be answered without specifying the full economic model and imposing a simpler set of assumption on potential outcomes: parallel trends. Let $D\in\{0,1\}$ indicate exposure to the treatment (learning of the policy change at age 60). Parallel trends is an assumption on the counterfactual for cohort $B$:
$$ \mathbb{E}[C|B,D=0,t] = \gamma_{B} + \mathbb{E}[C|A,D=0,t] $$
it says that these two cohorts, in the absence of the treatment, would differ, but these differences would be constant with age. This allows us an estimate of the causal effect using observations for any age $s<60$:
$$ \mathbb{E}[C|B,D=1,t] - \mathbb{E}[C|B,D=0,t] = \mathbb{E}[C|B,t] - \mathbb{E}[C|A,t] -  \left(\mathbb{E}[C|B,s] - \mathbb{E}[C|A,s]\right) $$
Notice that the left-hand side defines the causal effect of eligibility on consumption via a *counterfactual* and the right-hand side consists only of estimable quantities.

We have only to defend the parallel trends assumption, which many view as less burdensome compared to defending the many layers of assumptions in a quantitative model. In general, letting $t^*$ index the age of an individual when the policy is announced (so that $B$ is cohort $t^*=60$), the parallel trends assumption allows us to write:
$$ \mathbb{E}[C_{t^*,t}] = \gamma_{t^*} + \mu_{t} + \alpha_{t^*,t}\mathbf{1}\{t\geq t^*\} $$
where $\alpha_{t^*,t}$ is the effect of learning about the policy expansion at age $t^*$ on consumption at age $t$. This imposes parallel trends across every single cohort, even ones that are not particular adjacent to each other, which is arguably a much stronger assumption.


### Firm Entry

Consider the firm entry model. Suppose you have panel data $(A_{m,t,1},A_{m,t,2},Z_{m,t})_{m=1,t=1}^{M,T}$ on a set of markets (indexed by $m$) in a number of periods (indexed by $t$). Recall that $A_{m,t,j}$ indicates whether firm $j$ is active in market $m$ at time $t$, and firm entry occurs when $A_{m,t+1,j}-A_{m,t,j}=1$. Recall that $X_{m}$ is a a market-level factor that is unobservable here. Let $Z\in\{0,1\}$ indicate the presence of a policy that applies locally to market $m$. For the purposes of this example, let's say it's a local minimum wage policy. To incorporate the policy $Z$, suppose we write:

$$ u_{1}(x,a,d') = \phi_{0} + \phi_{1}x - \phi_{2}d' - \phi_{3}(1-a) + \phi_{4}z $$

so that $\phi_{4}$ embodies the effect of the policy on payoffs for the firm. Finally, for simplicity, let's assume that $Z_{m,1}=0$ for all states initially, and in period $t^*$ a subset of states adopt the policy permanently and that this adoption is *unanticipated*.

**Suppose your question** is: *"What is the effect of the minimum wage on firm entry?"*  Let $N_{m,t} = D_{m,t,1}+D_{m,t,2}$ be the number of participating firms in market $m$ at time $t$. Let $N_{\tau,m}(z)$ be the potential outcomes of $N$ in market $m$, $\tau$ periods after the adoption of the minimum wage policy. Let's define the dynamic effect of treatment on the treated (the effect of the minimum wage on markets that adopt it) as:

$$ \alpha_{\tau} = \mathbb{E}[N_{\tau}(1) - N_{\tau}(0)|Z = 1] = \mathbb{E}[\Delta_{\tau}|Z=1] $$

Our model doesn't outline a theory of why certain markets adopt the minimum wage and why others don't, but it does highlight that the effects of the policy will differ across markets, so it is important to account for heterogeneity: if there is any selection into policy adoption, we know that
$\mathbb{E}[\Delta_{\tau}|Z=1] \neq \mathbb{E}[\Delta_{\tau}]$.

A parallel trends assumption that justifies the event-study approach would be:
$$ \mathbb{E}[N_{t}(0)|Z=1] - \mathbb{E}[N_{t}(0)|Z=0] = \text{constant}. $$
Note that if we assume that each market is in the ergodic distribution governed by the Markov Perfect Equilibria, then the distribution of $N$ is stationary in each market absent the policy intervention and the parallel trends assumption is justified. The event-study specification:
$$ D_{m,t} = \gamma_{m} + \mu_{t} + \sum_{t\geq t^*}\alpha_{t-t^*} + \epsilon_{m,t} $$
would then robustly identify the average effect of the policy among the markets that adopted it. This is partly true because the timing of adoption was uniform. We should note that if adoption was staggered, given that there may be heterogeneous treatment effects, a regression-based approach to the event study would deliver some weighted average of these impacts that is hard to interpret.

### Bundles of Tax Reforms

Finally, consider a suite of tax reforms in the dynamic labor supply model. Suppose that there are three states, $A$, $B$, $C$. Suppose that each runs a different experiment where they introduce a different set of taxes and transfers. Let $\mathcal{Y}_{j}$ indicate a net income function for each state $j$. Consider the following examples:

\begin{align}
\mathcal{Y}_{A}(W,H) = b_{A} + (1-\tau_{A})WH \\
\mathcal{Y}_{B}(W,H) = WH + \sum_{k=0}^{5}\tau_{k}(WH-\overline{E}_{k})\mathbf{1}\{WH>\overline{E}_{k}\} \\
\mathcal{Y}_{C}(WH) = WH(1-\tau_{C}) + \mathbf{1}\{H>20\}b_{C}
\end{align}

And suppose that these participants do not anticipate their assignment to treatment in this experiment, which is expected to last for 3 periods. Let $Z_{j}\in\{0,1\}$ indicate assignment to either treatment or control group in state $j$.

**If your research question** is: *"What is the effect of each unannounced, temporary, tax reform on labor supply?"* then one could simply compare the means of treatment and control in each state:

$$ \mathbb{E}[H|Z_{j}=1, j] - \mathbb{E}[H|Z_{j} = 0, j] $$

which uncovers the causal effect of $Z$ by virtue of random assignment.

## Reasons to Use a Model

Having covered these three (perfectly valid) applications of quasi-experimental methods to answer specific research questions, let us now consider some questions that these methods *don't* answer and consider the useful role that models can play in these (and related) contexts.

### When the question can't be articulated without one

Perhaps the most obvious reason to use a model is when you research question simply cannot be articulated without one. Some of the most useful insights from economic modeling are statements about economic efficiency, the potential for policies to resolve market inefficiencies, and the design of policies. Some examples:

1. In the labor supply model, given a particular weighted welfare objective, what does the optimal system of taxes and transfers look like?
2. What is the cheapest way to incentivize competition between two firms in dynamic duopoly?
3. What is are the welfare costs of incomplete markets in the life-cycle savings model?.

### To make welfare calculations

*Revealed preference* is one of the more powerful tools in an economist's toolkit: if we treat individuals in our data as people who know what they like, we can try to infer their preferences and decide how they value different policy environments. For example:

1. How do individuals value the social security program introduced in the savings model? How would they value a program with a different combination of taxes ($\tau$) and payments $b$?
2. In the labor supply model, what is the sum of individual's willingness to pay for a lumb sum payment $b$ that is financed by proportion taxes $\tau$? We'll come back to this one below.

### To make sense of otherwise puzzling data

Here it's hard to look past the most fundamental and basic causal question in our profession: what is the effect of price on quantities? You know of course that this is a silly question, but we only know this because the theory of supply and demand is so fundamental to our view of the world.

Suppose you observe prices and quantities in a market over time. Without the theory of supply and demand, all you would see is a cloud of points.

**With** the theory of supply and demand, you and understand that each point is the simultaneous equilibrium outcome of two underlying structural relationships in equilibrium. Phillip and / or Sewall Wright proposed the solution: Instrumtal Variables, which is perhaps the earliest known example of an estimated structural model.

### When variation does not identify the counterfactual of interest

In our examples above, you may have noted that we were careful to very specifically define the "treatment" in order to specify the causal object of interest. Models can help us articulate just exactly what kind of causal parameters can and cannot be identified by observed variation, as well as outlining a specific set of assumptions under which related counterfactuals can be forecast even though they are not exactly replicated by existing variation. Each section below discussed a number of examples in the context of each application. Researchers often frame this as a question of *internal vs external validity*, but there are too many interesting examples in the "external validity" column to not discuss them in more depth. In general, a key point made by Heckman and Vytlacil (2005) is that estimands from simple statistical mdoels designed to infer causal effects (such as those we get from difference-in-differences, IV, and regression discontinuity) are rarely parameters of exact policy interest.

We'll use examples to explore these ideas.

#### Social Security

To make the example concrete, let's make some additional simplifying assumptions for the savings model. These make the quantitative model a bit less interesting, but help us think through the issues. Specifically, let's assume:

1. Each individual faces a *known* sequence $\{y_{n,t}\}_{t=1}^{T}$ of income realizations.
2. Agents face a *natural borrowing constraint*, yielding an intertemporal borrowing constraint at each $t$:
$$ \sum_{s=t}^{T}q_{s-t}c_{s} \leq a_{t} + \sum_{s=t}^{T}q_{s-t}(1-\tau)y_{n,s} + \sum_{s\geq 65}^{T}q_{s-t}b $$
where $q_{\tau} = 1/(1+r)^{\tau}$ is the price of a unit of consumption $\tau$ periods ahead.
3. Set $\beta(1+r)=1$, and $\psi=0$ (no bequest motive), indicating that agents will elect to perfectly smoooth their consumption over periods so that consumption is equal to the net present value of net income (something they can do due to the natural borrowing constraint).

With these assumptions, we can write the mean effect on consumption for cohort $t^*$ at any age $t\geq t^*$ as simply the effect of the announcement on the NPV of net income at age $t^*$:

$$ \Delta C_{t^*,t} = \sum_{s=60}^{64}q_{s-t^*}b + \sum_{s=t^*}^{T}(\tau - \tau')\overline{y}_{t^*,t} $$
where $\overline{y}_{t^*,t}$ is average income for cohort $t^*$ at age $t$.

With these assumptions, note that:

1. The parallel trends assumption holds: under the counterfactual of no policy change, differences across cohorts are constant with age ($\mu_{t}$ can be normalized to zero).
2. The difference-in-difference approach therefore robustly identifies, with $\alpha_{t^*,t}$, the causal effect of the policy on cohort $t^*$ at age $t$.
3. The model implies that $\alpha_{t^*,t}$ is constant with $t$.
4. Each effect depends on **how each cohort $t^*$ expects the policy expansion to be financed**, through the change in marginal tax rates $\tau' - \tau$. We haven't specified financing constraints and hence we cannot speculate on this without more structure.


Are the parameters $\alpha_{t^*,t}$ policy relevant quantities? They are certainly informative, but now that eligibility has been expanded, these effects don't tell us about the effect of future changes in policy. They don't even (without additional assumptions) tell us what the effect would be if the expansion were repealed. We need more theory and assumptions to extrapolate. This is a good task for economic models!

The points below suggest some compelling counterfactuals that the DD approach does not recover.

1. The total effect of the social security policy (not just the expansion).
2. The effect of additional changes in the age of eligibility.
3. The effect of changes in $b$ and $\tau$ on consumption at different ages, and at differen horizons of anticipation.
4. If there is any reason to think that effects are heterogeneous by cohort (if they face different wage profiles, for example), we also cannot construct estimates on the effect on cohort $t^*$ if they had known
5. The effect on consumption for cohorts who have known about the expansion for their whole life-cycle.
6. The effect of the expansion with under alternative financing arrangements.

#### Tax Reform

For concreteness, let's consider the effect of reform $A$ on labor supply, when $\beta(1+r)=1$ and optimal consumption is stationary over time. It is given by:
$$\Delta H_{n,t} = \psi\log(1-\tau) - \sigma\Delta \log\left(C^*_{n}\right). $$
The consumption response $\Delta C^*_{n}$ embodies the income effect and is harder. This concrete example helps us to understand three more general points about each reform:

1. The average treatment effect depends on income effects, which in turn depend on the perceived length of time that the tax reform is enforced.
2. The model exhibits lots of heterogeneity in treatment effects. As such, if there are differences in underlying distributions of wages or work costs, we should expect different impacts.

Thus, although each welfare experiment robustly identifies the effect of tax reform $A$ and population $A$, tax reform $B$ on population $B$, and so forth. It does not identify:

1. The effect of any tax reform with different persistence (real or perceived). Moreover, one would want to interpret the experimental findings very carefully to ensure that individuals in the experiment were given adequate information about the length of time of the experiment.
2. The effect of tax reform $A$ on population $B$, $C$, etc, and likewise for tax reforms $B$, $C$, etc on alternative populations.
3. The distribution of treatment effects at each location.
4. The effect of tax reforms that are *already partly anticipated* by individuals.
5. The effect of a scaled up tax reform, where *equilibrium effects* on wages might appear.
6. The distribution of effects of each tax reform. Here the model can be used to be interpret available panel data and invert out distributions in observed labor market productivities and work costs. One can then return to the experimental data and estimate average treatment effects along these latent dimensions.

#### Entry-Exit Model

By now we have made the point several ways, but when it comes to the model of dynamic duopoly, we can note that the model-free approach does not identify:

1. The effect of the minimum wage policy when it's introduction is anticipated several periods earlier.
2. The effect of the minimum wage on the markets that do not adopt it.
3. The effect of repealing the minimum wage.
4. The effect of nominal changes to the minimum wage.

Each of which might be considered much more useful or compelling policy calculations.

### To interpolate existing variation in the data

Sticking with the tax reform example, let's consider what it would take to jointly understand the effects. We know that each reform is related because each comes in the form of an infinite dimensional object: a function $\mathcal{T}(W,H)$ of wages and hours. A model-free theory that attempts to estimate labor supply as a non-parametric function of this function would not get very far due to the implausible quantities of policy variation that would be needed to estimate it.

Economic models often provide a useful way to interpolate related -- but not functionally identically related -- variation. Notice that in the labor supply model any function $\mathcal{T}$ is articulated through its effect on the budget constraint, and different policy reforms can be compared in the model without the addition of any new parameters. I refer to this property as *articulated variation*: when the effect of a variable can be modeled without the need for additional parameters. Typically, this involves *a priori* known changes to prices and endowments.

In this example, with structural parameters in hand, any function $\mathcal{T}$ of wages and hours and can be modeled, and hence any observed variation in this function can be interpreted through the model without additional parameters.

Here is a useful *counterexample*: policy variation that is not well-articulated inside a model. Consider our choice of modeling the minimum wage in our entry and exit model: embodied as a parameter $\phi_{4}$. Our choice to not model within-period production decisions of the firm (and instead estimate a reduced form) means that this policy is *not* well-articulated. We need an additional parameter $\phi_{4}$ to model its effect, and further changes to the nominal wage cannot be studied.


## Marschak's Maxim

Our discussion so far has focused on exploring the boundaries of what simpler causal inference strategies can identify in order to make statements about how economic modeling can add value in research. This is not very instructive for *how* we do economic modeling. Of course this topic is more of an art than a science, but Marschak's Maxim is a useful principle that can guide us. Here it is:

> Researchers should specify the minimal set of model ingredients in order to answer the research question of interest.

This may seem obvious, and that the hard part is figuring out what this minimal set is. It's true that this can be hard to decide, but it's surprisingly easy to forget this simple rule once you are deep inside your model and making decisions. The question "Is this essential to my question of interest?" is not always easy to answer but it is one you should be repeatedly asking yourself. Your research question is the mast you tie yourself to, and you should decide as early as possible what it is.

Let's do some clean examples to make the point clear.

### Marschak's Original Example

### Two-Stage Budgeting

### Sufficient Statistics
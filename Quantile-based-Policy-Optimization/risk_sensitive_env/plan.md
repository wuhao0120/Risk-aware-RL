```markdown
# BDQC Implementation Plan: On-policy and PPO-like Abel-BDQC

## 0. Goal

This document describes two implementable versions of **Budgeted Distributional/Constraint Actor-Critic (BDQC)** for infinite-horizon quantile/chance-constrained reinforcement learning.

The method combines three ideas:

1. **Budget augmentation**: convert the trajectory-level event $Z \le q$ into a local event $G_t \le b_t$.
2. **Budget-CDF critic**: learn $C(s,a,b) \approx P(G_t \le b \mid s_t=s,a_t=a)$.
3. **Abel risk discount**: replace the non-discounted infinite-horizon constraint gradient by an Abel-regularized risk occupancy with discount $\beta < 1$.

We provide two algorithmic versions:

- **Version A: On-policy Abel-BDQC-AC**
  - theoretically cleaner;
  - actor and critic mainly use current rollout data;
  - lower data efficiency.

- **Version B: PPO-like Abel-BDQC**
  - actor uses a short-term rollout buffer with PPO-style clipped ratio;
  - critic can additionally use a longer replay buffer;
  - higher data efficiency and more practical.

---

## 1. Problem Formulation

We consider an infinite-horizon discounted MDP with policy $\pi_\theta(a|s)$ and discount $\gamma \in (0,1)$.

The discounted return is

$$
Z = \sum_{t=0}^{\infty} \gamma^t r_t.
$$

The objective is to maximize expected return subject to a chance/quantile constraint:

$$
\max_\theta J_m(\theta) = \mathbb{E}_\theta[Z]
$$

subject to

$$
G(\theta) = P_\theta(Z \le q) \le \alpha.
$$

The Lagrangian is

$$
L(\theta,\lambda)
=
J_m(\theta)
-
\lambda \left(G(\theta)-\alpha\right),
\qquad
\lambda \ge 0.
$$

The dual update should always be based on the original constraint $P_\theta(Z \le q) \le \alpha$, not on the Abel-regularized gradient.

---

## 2. Budget Augmentation

Define the prefix discounted return:

$$
P_t = \sum_{k=0}^{t-1} \gamma^k r_k.
$$

Define the return-to-go:

$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}.
$$

Then the total return decomposes as

$$
Z = P_t + \gamma^t G_t.
$$

Therefore,

$$
Z \le q
\iff
G_t \le \frac{q-P_t}{\gamma^t}.
$$

Define the running budget:

$$
b_t = \frac{q-P_t}{\gamma^t}.
$$

Then

$$
Z \le q
\iff
G_t \le b_t.
$$

The budget admits the one-step recursion:

$$
b_{t+1}
=
\frac{b_t-r_t}{\gamma}.
$$

We also maintain two scalar weights:

$$
d_t = \gamma^t,
\qquad
e_t = \beta^t.
$$

Their recursions are

$$
d_{t+1} = \gamma d_t,
\qquad
e_{t+1} = \beta e_t.
$$

Initialization at the start of each rollout:

$$
b_0=q,
\qquad
d_0=1,
\qquad
e_0=1.
$$

---

## 3. Critics

We use two critics.

### 3.1 Mean Critic

The mean critic estimates the usual return-to-go:

$$
Q_m^\pi(s,a)
=
\mathbb{E}_\pi
\left[
G_t
\mid
s_t=s,a_t=a
\right].
$$

The Bellman equation is

$$
Q_m^\pi(s,a)
=
\mathbb{E}_{r,s',a'}
\left[
r + \gamma Q_m^\pi(s',a')
\right],
\qquad
a' \sim \pi_\theta(\cdot|s').
$$

TD target:

$$
y_m = r + \gamma \bar V_m(s').
$$

Here

$$
\bar V_m(s')
=
\mathbb{E}_{a'\sim\pi_\theta(\cdot|s')}
[
\bar Q_m(s',a')
].
$$

For implementation, estimate $\bar V_m(s')$ with action sampling:

$$
\bar V_m(s')
\approx
\frac{1}{K}
\sum_{j=1}^{K}
\bar Q_m(s',\tilde a'_j),
\qquad
\tilde a'_j \sim \pi_\theta(\cdot|s').
$$

Mean critic loss:

$$
\mathcal{L}_m
=
\mathbb{E}
\left[
\left(Q_m(s,a)-y_m\right)^2
\right].
$$

---

### 3.2 Budget-CDF Critic

The Budget-CDF critic estimates the local violation probability:

$$
C^\pi(s,a,b)
=
P_\pi(G_t \le b \mid s_t=s,a_t=a).
$$

Because

$$
G_t = r_t + \gamma G_{t+1},
$$

we have

$$
G_t \le b
\iff
G_{t+1} \le \frac{b-r_t}{\gamma}.
$$

Define

$$
b' = \frac{b-r}{\gamma}.
$$

The Budget-CDF Bellman equation is

$$
C^\pi(s,a,b)
=
\mathbb{E}_{r,s',a'}
\left[
C^\pi(s',a',b')
\right],
\qquad
a' \sim \pi_\theta(\cdot|s').
$$

TD target:

$$
y_c = \bar V_c(s',b').
$$

Here

$$
\bar V_c(s',b')
=
\mathbb{E}_{a'\sim\pi_\theta(\cdot|s')}
[
\bar C(s',a',b')
].
$$

Implementation approximation:

$$
\bar V_c(s',b')
\approx
\frac{1}{K}
\sum_{j=1}^{K}
\bar C(s',\tilde a'_j,b'),
\qquad
\tilde a'_j \sim \pi_\theta(\cdot|s').
$$

Risk critic loss:

$$
\mathcal{L}_c
=
\mathbb{E}
\left[
\left(C(s,a,b)-y_c\right)^2
\right].
$$

If the environment has terminal states, use the terminal target

$$
y_c = \mathbf{1}\{r \le b\}
$$

when `done=True`.

For continuing infinite-horizon tasks, terminal handling can be omitted or implemented through an absorbing state.

---

## 4. Advantages

Mean value baseline:

$$
V_m(s)
=
\mathbb{E}_{a\sim\pi_\theta(\cdot|s)}
[
Q_m(s,a)
].
$$

Risk value baseline:

$$
V_c(s,b)
=
\mathbb{E}_{a\sim\pi_\theta(\cdot|s)}
[
C(s,a,b)
].
$$

Mean advantage:

$$
A_m(s,a)
=
Q_m(s,a)-V_m(s).
$$

Risk advantage:

$$
A_c(s,a,b)
=
C(s,a,b)-V_c(s,b).
$$

Composite BDQC advantage:

$$
A_{\mathrm{BDQC}}(s,a,b,d,e)
=
d A_m(s,a)
-
\lambda e A_c(s,a,b).
$$

Since $d=\gamma^t$ and $e=\beta^t$, this corresponds to the Abel-BDQC gradient estimator:

$$
g_t
=
\nabla_\theta \log \pi_\theta(a_t|s_t)
\left[
\gamma^t A_m(s_t,a_t)
-
\lambda \beta^t A_c(s_t,a_t,b_t)
\right].
$$

---

## 5. Dual Update

The dual variable must be updated using the original constraint, not the Abel-regularized objective.

At the initial state, $b_0=q$ and $G_0=Z$. Therefore,

$$
P_\theta(Z \le q)
=
\mathbb{E}_{s_0\sim\rho,a_0\sim\pi_\theta(\cdot|s_0)}
[
C(s_0,a_0,q)
].
$$

Estimate the violation probability by

$$
\hat G(\theta)
=
\frac{1}{M_0}
\sum_{i=1}^{M_0}
C(s_0^{(i)},a_0^{(i)},q),
\qquad
a_0^{(i)}\sim\pi_\theta(\cdot|s_0^{(i)}).
$$

Dual update:

$$
\lambda
\leftarrow
\left[
\lambda
+
\eta_\lambda
\left(
\hat G(\theta)-\alpha
\right)
\right]_+.
$$

Do not multiply this update by $\beta^t$.

---

## 6. Version A: On-policy Abel-BDQC-AC

### 6.1 Intended Use

Use this version when you want the cleanest theoretical alignment with the Abel-BDQC policy gradient.

Properties:

- actor uses only current policy rollout data;
- no importance ratio is needed;
- critic can be trained from current rollout data;
- data efficiency is lower than PPO-like reuse.

---

### 6.2 Data Collected Per Transition

For each transition, store:

```text
s_t
a_t
r_t
s_{t+1}
done_t
b_t
d_t
e_t
logp_old_t
```

For the purely on-policy version, `logp_old_t` is optional, but storing it is useful for debugging and for switching to the PPO-like version.

---

### 6.3 Rollout Collection

At the beginning of each rollout:

$$
b_0=q,
\qquad
d_0=1,
\qquad
e_0=1.
$$

For $t=0,\dots,H-1$:

1. Sample action:

$$
a_t \sim \pi_{\theta_k}(\cdot|s_t).
$$

2. Execute action and observe $r_t,s_{t+1},done_t$.

3. Store transition:

```text
(s_t, a_t, r_t, s_{t+1}, done_t, b_t, d_t, e_t, logp_old_t)
```

4. Update budget and weights:

$$
b_{t+1}=\frac{b_t-r_t}{\gamma},
$$

$$
d_{t+1}=\gamma d_t,
$$

$$
e_{t+1}=\beta e_t.
$$

If `done=True`, reset the environment and reset $b,d,e$.

---

### 6.4 Critic Update

For each minibatch transition $(s,a,r,s',done,b,d,e)$:

1. Compute next budget:

$$
b'=\frac{b-r}{\gamma}.
$$

2. Compute mean target:

$$
y_m =
r
+
\gamma(1-done)\bar V_m(s').
$$

3. Compute risk target:

$$
y_c =
\begin{cases}
\mathbf{1}\{r \le b\}, & done=True,\\
\bar V_c(s',b'), & done=False.
\end{cases}
$$

4. Critic loss:

$$
\mathcal{L}_{\mathrm{critic}}
=
\mathbb{E}
\left[
(Q_m(s,a)-y_m)^2
+
\xi(C(s,a,b)-y_c)^2
\right].
$$

5. Update critic parameters by descending $\nabla \mathcal{L}_{\mathrm{critic}}$.

---

### 6.5 Actor Update

For each transition:

1. Estimate baselines:

$$
V_m(s)
=
\mathbb{E}_{\tilde a\sim\pi_\theta(\cdot|s)}
[
Q_m(s,\tilde a)
],
$$

$$
V_c(s,b)
=
\mathbb{E}_{\tilde a\sim\pi_\theta(\cdot|s)}
[
C(s,\tilde a,b)
].
$$

2. Compute advantages:

$$
A_m(s,a)=Q_m(s,a)-V_m(s),
$$

$$
A_c(s,a,b)=C(s,a,b)-V_c(s,b).
$$

3. Compute composite advantage:

$$
\hat A_{\mathrm{BDQC}}
=
d A_m(s,a)
-
\lambda e A_c(s,a,b).
$$

4. Actor loss:

$$
\mathcal{L}_{\mathrm{actor}}
=
-
\mathbb{E}_{\mathrm{rollout}}
[
\log \pi_\theta(a|s)
\hat A_{\mathrm{BDQC}}
]
-
\tau
\mathbb{E}_{\mathrm{rollout}}
[
\mathcal{H}(\pi_\theta(\cdot|s))
].
$$

5. Update actor parameters by descending $\nabla \mathcal{L}_{\mathrm{actor}}$.

---

### 6.6 Dual Update

Estimate

$$
\hat G(\theta)
=
\frac{1}{M_0}
\sum_{i=1}^{M_0}
C(s_0^{(i)},a_0^{(i)},q).
$$

Update

$$
\lambda
\leftarrow
\left[
\lambda
+
\eta_\lambda
\left(
\hat G(\theta)-\alpha
\right)
\right]_+.
$$

---

### 6.7 On-policy Pseudocode

```text
Initialize actor pi_theta
Initialize mean critic Q_m
Initialize Budget-CDF critic C
Initialize target critics Q_m_bar, C_bar
Initialize lambda = 0

for iteration k = 1,2,...:
    rollout_buffer = []

    for each environment:
        reset env if needed
        b = q
        d = 1
        e = 1

        for t = 0,...,H-1:
            a, logp = sample pi_theta(.|s)
            r, s_next, done = env.step(a)

            store (s, a, r, s_next, done, b, d, e, logp) in rollout_buffer

            b = (b - r) / gamma
            d = gamma * d
            e = beta * e

            s = s_next

            if done:
                reset env
                b = q
                d = 1
                e = 1

    for critic_step = 1,...,K_c:
        sample minibatch from rollout_buffer
        compute y_m
        compute y_c
        minimize critic loss

    compute A_m, A_c, and A_BDQC for rollout_buffer

    for actor_step = 1,...,K_pi:
        minimize actor loss using rollout_buffer

    estimate original violation G_hat using C(s0,a0,q)
    lambda = max(0, lambda + eta_lambda * (G_hat - alpha))

    update target critics
```

---

## 7. Version B: PPO-like Abel-BDQC

### 7.1 Intended Use

Use this version when you want better data efficiency while still keeping the actor update close to on-policy.

Properties:

- actor uses a short-term rollout buffer;
- actor data is reused for multiple epochs;
- PPO clipped ratio controls policy drift;
- critic can use both rollout buffer and long-term replay buffer.

---

### 7.2 Buffers

Use two buffers.

#### 7.2.1 Rollout Buffer for Actor

Short-term buffer collected by the current old policy $\pi_{\theta_{\mathrm{old}}}$.

Store:

```text
s_t
a_t
r_t
s_{t+1}
done_t
b_t
d_t
e_t
logp_old_t
```

This buffer is used for PPO-like actor updates and is cleared after each iteration.

#### 7.2.2 Replay Buffer for Critic

Long-term buffer.

Store:

```text
s_t
a_t
r_t
s_{t+1}
done_t
b_t
```

This buffer is used to train $Q_m$ and $C$ repeatedly.

---

### 7.3 Rollout Collection

Set

$$
\theta_{\mathrm{old}}=\theta.
$$

Collect rollout using $\pi_{\theta_{\mathrm{old}}}$.

For each rollout segment, initialize:

$$
b_0=q,
\qquad
d_0=1,
\qquad
e_0=1.
$$

For $t=0,\dots,H-1$:

1. Sample action:

$$
a_t \sim \pi_{\theta_{\mathrm{old}}}(\cdot|s_t).
$$

2. Store old log probability:

$$
\log p_{\mathrm{old},t}
=
\log \pi_{\theta_{\mathrm{old}}}(a_t|s_t).
$$

3. Execute action and observe $r_t,s_{t+1},done_t$.

4. Store in rollout buffer:

```text
(s_t, a_t, r_t, s_{t+1}, done_t, b_t, d_t, e_t, logp_old_t)
```

5. Store in critic replay buffer:

```text
(s_t, a_t, r_t, s_{t+1}, done_t, b_t)
```

6. Update:

$$
b_{t+1}=\frac{b_t-r_t}{\gamma},
$$

$$
d_{t+1}=\gamma d_t,
$$

$$
e_{t+1}=\beta e_t.
$$

If `done=True`, reset the environment and reset $b,d,e$.

---

### 7.4 Critic Update with Replay

Sample minibatches from either:

- current rollout buffer;
- long-term critic replay buffer;
- a mixture of both.

For transition $(s,a,r,s',done,b)$:

1. Compute

$$
b'=\frac{b-r}{\gamma}.
$$

2. Sample next action from current policy:

$$
a' \sim \pi_\theta(\cdot|s').
$$

3. Mean target:

$$
y_m =
r
+
\gamma(1-done)\bar Q_m(s',a').
$$

4. Risk target:

$$
y_c =
\begin{cases}
\mathbf{1}\{r \le b\}, & done=True,\\
\bar C(s',a',b'), & done=False.
\end{cases}
$$

5. Critic loss:

$$
\mathcal{L}_{\mathrm{critic}}
=
\mathbb{E}
\left[
(Q_m(s,a)-y_m)^2
+
\xi(C(s,a,b)-y_c)^2
\right].
$$

Update critics for $K_c$ steps.

---

### 7.5 Compute Composite Advantage

For each transition in rollout buffer:

1. Compute

$$
V_m(s)
=
\mathbb{E}_{\tilde a\sim\pi_\theta(\cdot|s)}
[
Q_m(s,\tilde a)
].
$$

2. Compute

$$
V_c(s,b)
=
\mathbb{E}_{\tilde a\sim\pi_\theta(\cdot|s)}
[
C(s,\tilde a,b)
].
$$

3. Compute

$$
A_m(s,a)=Q_m(s,a)-V_m(s).
$$

4. Compute

$$
A_c(s,a,b)=C(s,a,b)-V_c(s,b).
$$

5. Composite advantage:

$$
\hat A_{\mathrm{BDQC}}
=
d A_m(s,a)
-
\lambda e A_c(s,a,b).
$$

Optional normalization:

- normalize $A_m$ and $A_c$ separately before combining; or
- normalize $\hat A_{\mathrm{BDQC}}$ after combining.

Recommended initial implementation:

```text
normalize A_m over rollout buffer
normalize A_c over rollout buffer
A_BDQC = d * A_m - lambda * e * A_c
```

---

### 7.6 PPO-like Actor Update

For each transition in rollout buffer, compute the probability ratio:

$$
\rho_t(\theta)
=
\frac{\pi_\theta(a_t|s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t|s_t)}
=
\exp
\left(
\log\pi_\theta(a_t|s_t)
-
\log p_{\mathrm{old},t}
\right).
$$

PPO clipped surrogate:

$$
L_{\mathrm{clip}}(\theta)
=
\mathbb{E}
\left[
\min
\left(
\rho_t(\theta)\hat A_{\mathrm{BDQC},t},
\mathrm{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)
\hat A_{\mathrm{BDQC},t}
\right)
\right].
$$

Actor loss:

$$
\mathcal{L}_{\mathrm{actor}}
=
-
L_{\mathrm{clip}}(\theta)
-
\tau
\mathbb{E}
[
\mathcal{H}(\pi_\theta(\cdot|s))
].
$$

Update actor for $K_\pi$ epochs.

Recommended:

```text
K_pi = 3 to 10
clip_epsilon = 0.1 to 0.3
target_KL = optional early stopping
```

---

### 7.7 Dual Update

Estimate the original violation probability:

$$
\hat G(\theta)
=
\frac{1}{M_0}
\sum_{i=1}^{M_0}
C(s_0^{(i)},a_0^{(i)},q),
\qquad
a_0^{(i)}\sim\pi_\theta(\cdot|s_0^{(i)}).
$$

Update:

$$
\lambda
\leftarrow
\left[
\lambda
+
\eta_\lambda
\left(
\hat G(\theta)-\alpha
\right)
\right]_+.
$$

Again, this update does not use $\beta$.

---

### 7.8 PPO-like Pseudocode

```text
Initialize actor pi_theta
Initialize mean critic Q_m
Initialize Budget-CDF critic C
Initialize target critics Q_m_bar, C_bar
Initialize lambda = 0
Initialize critic_replay_buffer

for iteration k = 1,2,...:
    theta_old = theta
    rollout_buffer = []

    for each environment:
        reset env if needed
        b = q
        d = 1
        e = 1

        for t = 0,...,H-1:
            a, logp_old = sample pi_theta_old(.|s)
            r, s_next, done = env.step(a)

            store (s, a, r, s_next, done, b, d, e, logp_old) in rollout_buffer
            store (s, a, r, s_next, done, b) in critic_replay_buffer

            b = (b - r) / gamma
            d = gamma * d
            e = beta * e

            s = s_next

            if done:
                reset env
                b = q
                d = 1
                e = 1

    for critic_step = 1,...,K_c:
        sample minibatch from rollout_buffer and/or critic_replay_buffer
        compute b_next = (b - r) / gamma
        sample a_next ~ pi_theta(.|s_next)
        compute y_m = r + gamma * (1-done) * Q_m_bar(s_next, a_next)
        compute y_c = done ? 1{r <= b} : C_bar(s_next, a_next, b_next)
        minimize critic loss

    compute A_m, A_c, and A_BDQC for rollout_buffer

    for actor_epoch = 1,...,K_pi:
        for minibatch in rollout_buffer:
            logp = log pi_theta(a|s)
            ratio = exp(logp - logp_old)
            surrogate_1 = ratio * A_BDQC
            surrogate_2 = clip(ratio, 1-eps, 1+eps) * A_BDQC
            actor_loss = -mean(min(surrogate_1, surrogate_2)) - tau * entropy
            update actor

        if approximate_KL > target_KL:
            break

    estimate original violation G_hat using C(s0,a0,q)
    lambda = max(0, lambda + eta_lambda * (G_hat - alpha))

    update target critics
    clear rollout_buffer
```

---

## 8. Practical Implementation Details

### 8.1 Budget Clipping

Because

$$
b_t = \frac{q-P_t}{\gamma^t},
$$

$b_t$ can become numerically large.

If rewards are bounded by

$$
r_t \in [r_{\min}, r_{\max}],
$$

then

$$
G_t \in
\left[
\frac{r_{\min}}{1-\gamma},
\frac{r_{\max}}{1-\gamma}
\right].
$$

Therefore, clip budget to

$$
b_{\min}=\frac{r_{\min}}{1-\gamma},
\qquad
b_{\max}=\frac{r_{\max}}{1-\gamma}.
$$

Use

$$
\tilde b = \mathrm{clip}(b,b_{\min},b_{\max}).
$$

Feed $\tilde b$ into $C(s,a,b)$.

---

### 8.2 Network Design

Recommended architecture:

```text
shared_encoder(s)
action_encoder(a)
budget_encoder(b)

Q_m_head(shared_features, action_features)
C_head(shared_features, action_features, budget_features)
```

For image observations:

```text
vision_encoder(image)
proprio_encoder(proprio)
action_encoder(action)
budget_encoder(b)
fusion_mlp
Q_m_head
C_head
```

The Budget-CDF critic output should be in $[0,1]$.

Use

```text
C = sigmoid(raw_C)
```

or clamp output.

---

### 8.3 Loss Choices for C

Default:

$$
\mathcal{L}_c = (C-y_c)^2.
$$

Alternative with BCE:

$$
\mathcal{L}_c
=
-
y_c \log C
-
(1-y_c)\log(1-C).
$$

MSE is simpler for soft TD targets.

---

### 8.4 Choosing $\beta$

$\beta$ controls the Abel risk occupancy.

Recommended schedule:

```text
start beta = 0.90 or 0.95
increase beta toward 0.98 or 0.995
do not set beta = 1 in infinite horizon training
```

Example:

$$
\beta_k
=
1 - (1-\beta_0)\eta_\beta^k.
$$

Or piecewise:

```text
0 - 20% training: beta = 0.90
20% - 50% training: beta = 0.95
50% - 80% training: beta = 0.98
80% - 100% training: beta = 0.995
```

---

### 8.5 Choosing $\lambda$ Update Frequency

Dual update should be slower than actor update.

Recommended:

```text
update lambda every 1 to 10 policy iterations
eta_lambda smaller than actor learning rate
clip lambda to [0, lambda_max]
```

Use:

$$
\lambda
\leftarrow
\mathrm{clip}
\left(
\lambda+\eta_\lambda(\hat G-\alpha),
0,
\lambda_{\max}
\right).
$$

---

### 8.6 Advantage Normalization

Recommended initial implementation:

1. compute raw $A_m$;
2. compute raw $A_c$;
3. normalize each separately;
4. combine:

$$
\hat A_{\mathrm{BDQC}}
=
d \cdot \mathrm{norm}(A_m)
-
\lambda e \cdot \mathrm{norm}(A_c).
$$

This avoids scale mismatch between reward and risk critic.

---

### 8.7 Critic Target Networks

Use target networks:

```text
Q_m_bar
C_bar
```

Soft update:

$$
\bar \omega
\leftarrow
\tau_{\mathrm{target}}\omega
+
(1-\tau_{\mathrm{target}})\bar \omega.
$$

Recommended:

```text
tau_target = 0.005
```

---

## 9. Recommended Experiment Plan

### 9.1 Ablations

Run the following ablations:

1. **No risk constraint**
   - standard actor-critic or PPO baseline.

2. **Trajectory-level Monte Carlo QCPO**
   - update risk only after full trajectory;
   - low data efficiency baseline.

3. **BDQC without Abel**
   - use $\beta=1$ for short finite rollouts only;
   - mainly diagnostic, not recommended for infinite horizon.

4. **Abel-BDQC without Budget-CDF critic**
   - use Monte Carlo violation label;
   - tests critic contribution.

5. **Full On-policy Abel-BDQC-AC**
   - version A.

6. **Full PPO-like Abel-BDQC**
   - version B.

7. **Different $\beta$ values**
   - $\beta \in \{0.90,0.95,0.98,0.995\}$.

8. **Different constraint levels**
   - $\alpha \in \{0.05,0.10,0.20\}$.

---

### 9.2 Metrics

Report:

```text
Average return
Violation probability P(Z <= q)
Constraint satisfaction gap: P(Z <= q) - alpha
Sample efficiency
Dual variable lambda curve
Mean critic loss
Budget-CDF critic loss
Distribution of b_t
Distribution of C(s,a,b)
```

Important risk metrics:

$$
\hat P(Z\le q)
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}\{Z^{(i)}\le q\}.
$$

Critic-estimated violation:

$$
\hat G_C
=
\frac{1}{M}
\sum_{i=1}^{M}
C(s_0^{(i)},a_0^{(i)},q).
$$

Compare both.

---

## 10. Implementation Checklist

### Core State Variables

- [ ] maintain $b_t$;
- [ ] maintain $d_t=\gamma^t$;
- [ ] maintain $e_t=\beta^t$;
- [ ] reset $b,d,e$ at new rollout;
- [ ] store $b,d,e$ in rollout buffer;
- [ ] store $b$ in critic replay buffer.

### Critics

- [ ] implement $Q_m(s,a)$;
- [ ] implement $C(s,a,b)$;
- [ ] ensure $C$ output is in $[0,1]$;
- [ ] implement target networks;
- [ ] implement mean TD target;
- [ ] implement Budget-CDF TD target;
- [ ] handle terminal states if applicable;
- [ ] clip budget before feeding into $C$.

### Actor

- [ ] compute $A_m$;
- [ ] compute $A_c$;
- [ ] compute $\hat A_{\mathrm{BDQC}}=dA_m-\lambda eA_c$;
- [ ] implement on-policy actor loss;
- [ ] implement PPO clipped actor loss;
- [ ] store and use `logp_old` for PPO-like version;
- [ ] optionally add entropy regularization.

### Dual

- [ ] estimate $\hat G(\theta)$ using $C(s_0,a_0,q)$;
- [ ] update $\lambda$ with projected ascent;
- [ ] do not use $\beta$ in dual update;
- [ ] optionally clip $\lambda$.

### Evaluation

- [ ] compute empirical $P(Z\le q)$ from full evaluation rollouts;
- [ ] compare empirical violation with critic estimate;
- [ ] plot return-risk tradeoff;
- [ ] plot $\lambda$ over training;
- [ ] plot $b_t$ distribution.

---

## 11. Which Version to Implement First?

Recommended order:

1. Implement **On-policy Abel-BDQC-AC** first.
   - This validates the math.
   - Easier to debug.
   - No PPO ratio complications.

2. Add **PPO-like Abel-BDQC** second.
   - Reuse rollout data for multiple actor epochs.
   - Add clipped ratio.
   - Add critic replay buffer.

3. Add engineering improvements.
   - budget clipping;
   - beta annealing;
   - dual clipping;
   - critic target networks;
   - advantage normalization.

---

## 12. Minimal MVP

The minimal working version should include:

```text
gamma
beta
q
alpha
lambda
budget b_t
reward weight d_t
risk weight e_t
Q_m critic
C critic
actor loss with d*A_m - lambda*e*A_c
dual update with C(s0,a0,q)
```

Start with the on-policy version before PPO-like reuse.

---

## 13. Core Equations Summary

Budget:

$$
b_{t+1}=\frac{b_t-r_t}{\gamma}.
$$

Mean critic:

$$
Q_m(s,a)
=
\mathbb{E}
[
r+\gamma Q_m(s',a')
].
$$

Budget-CDF critic:

$$
C(s,a,b)
=
\mathbb{E}
\left[
C\left(s',a',\frac{b-r}{\gamma}\right)
\right].
$$

Advantages:

$$
A_m(s,a)=Q_m(s,a)-V_m(s).
$$

$$
A_c(s,a,b)=C(s,a,b)-V_c(s,b).
$$

Actor gradient:

$$
g_t
=
\nabla_\theta \log \pi_\theta(a_t|s_t)
\left[
\gamma^t A_m(s_t,a_t)
-
\lambda \beta^t A_c(s_t,a_t,b_t)
\right].
$$

PPO-like composite advantage:

$$
\hat A_{\mathrm{BDQC}}
=
d_t A_m(s_t,a_t)
-
\lambda e_t A_c(s_t,a_t,b_t).
$$

PPO-like clipped surrogate:

$$
L_{\mathrm{clip}}
=
\mathbb{E}
\left[
\min
\left(
\rho_t \hat A_{\mathrm{BDQC}},
\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_{\mathrm{BDQC}}
\right)
\right].
$$

Dual update:

$$
\lambda
\leftarrow
\left[
\lambda+\eta_\lambda
\left(
\mathbb{E}_{s_0,a_0}[C(s_0,a_0,q)]-\alpha
\right)
\right]_+.
$$

---

## 14. Main Research Story

The original chance constraint $P(Z\le q)\le \alpha$ is trajectory-level and does not naturally admit a standard discounted policy gradient theorem. The budget variable $b_t$ transforms this global event into a local event $G_t\le b_t$. The Budget-CDF critic $C(s,a,b)$ learns the local probability of violating the remaining budget. In infinite horizon, the exact risk gradient is a non-discounted infinite sum, which is difficult to optimize directly. Abel risk discount $\beta<1$ turns this risk gradient into a well-defined discounted occupancy expectation, enabling stable per-transition learning and replay-compatible implementation.

The final method can therefore be summarized as:

```text
Budget augmentation
+ Budget-CDF critic
+ Abel risk occupancy
+ on-policy or PPO-like actor update
```

```


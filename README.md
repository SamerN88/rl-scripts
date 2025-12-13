# rl-scripts

A collection of Python scripts implementing reinforcement learning algorithms and solving homework problems from Sutton & Barto's "Reinforcement Learning: An Introduction" (2nd Edition).

## Chapter 3: Finite Markov Decision Processes

### [`compute_returns_with_gamma.py`](ch3/compute_returns_with_gamma.py)
Computes discounted returns $G_0, G_1, \ldots, G_T$ for a trajectory using backward recursion: $G_t = R_{t+1} + \gamma G_{t+1}$.

---

## Chapter 4: Dynamic Programming

### [`gridworld_q_value_example4_1.py`](ch4/gridworld_q_value_example4_1.py)
Computes action-value function $q_\pi(s,a)$ for a 4×4 gridworld under an equiprobable random policy. Implements the Bellman expectation equation for Q-values with deterministic transitions.

---

## Chapter 5: Monte Carlo Methods

### [`first_visit_every_visit_estimators.py`](ch5/first_visit_every_visit_estimators.py)
Compares first-visit and every-visit Monte Carlo estimators for a single-state MDP. First-visit uses only $G_0$; every-visit averages all returns $G_t$ over the episode.

---

## Chapter 6: Temporal-Difference Learning

### [`TD0_update_random_walk.py`](ch6/TD0_update_random_walk.py)
Implements the TD(0) update rule: $V(s) \leftarrow V(s) + \alpha [R + \gamma V(s') - V(s)]$ for a random walk Markov Reward Process.

---

## Chapter 7: n-step Bootstrapping

### [`n_step_TD_prediction_example.py`](ch7/n_step_TD_prediction_example.py)
Computes n-step TD returns $G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k R_{t+1+k} + \gamma^n V(S_{t+n})$ and performs n-step TD updates for state-value prediction.

---

## Chapter 9: On-policy Prediction with Approximation

### [`ch9_q5_baird_semigradient_TD0_batch_update.py`](ch9/ch9_q5_baird_semigradient_TD0_batch_update.py)
Implements semi-gradient TD(0) with batch updates using linear function approximation. Computes TD errors $\delta_t = R_{t+1} + \gamma \hat{v}(S_{t+1}, w) - \hat{v}(S_t, w)$ and accumulates gradient updates across all transitions before applying a single batch update.

---

## Chapter 10: On-policy Control with Approximation

### [`avg_reward_when_violating_ergodicity.py`](ch10/avg_reward_when_violating_ergodicity.py)
Computes the average reward $\bar{r}(\pi)$ for a deterministic periodic reward sequence using the average-reward criterion (Eq. 10.6). Handles non-ergodic MDPs where standard limits don't exist.

---

## Chapter 12: Eligibility Traces

### Homework Solutions

#### [`q1__forward_lambda_return_full_episode.py`](ch12/hw/q1__forward_lambda_return_full_episode.py)
Computes the forward-view $\lambda$-return for full episodes: 
$$G_t^\lambda = (1-\lambda) \sum_{n=1}^{T-t-1} \lambda^{n-1} G_t^{(n)} + \lambda^{T-t-1} G_t$$

#### [`q2__forward_lambda_return_truncated.py`](ch12/hw/q2__forward_lambda_return_truncated.py)
Computes the n-step truncated $\lambda$-return $G_{t:h}^\lambda$ with a fixed horizon $h$, used in online TD($\lambda$) algorithms.

#### [`q3__lambda_return_from_td_errors.py`](ch12/hw/q3__lambda_return_from_td_errors.py)
Computes $\lambda$-returns from TD errors using: 
$$G_t^\lambda = \hat{v}(S_t, w) + \sum_{k=t}^{T-1} (\gamma\lambda)^{k-t} \delta_k$$
where $\delta_k = R_{k+1} + \gamma \hat{v}(S_{k+1}, w) - \hat{v}(S_k, w)$.

### Tools

#### [`eligibility_trace.py`](ch12/tools/eligibility_trace.py)
Computes eligibility traces using the accumulating trace: $z_t = \gamma\lambda z_{t-1} + \nabla\hat{v}(S_t, w)$. Includes utilities for one-hot feature vectors in tabular problems.

#### [`lam_return.py`](ch12/tools/lam_return.py)
General implementation of forward-view $\lambda$-return computation from episode trajectories and value estimates.

#### [`lam_return_using_TD_errors.py`](ch12/tools/lam_return_using_TD_errors.py)
Computes $\lambda$-returns from TD errors with optional truncation horizon. Provides both episode-based and pre-computed TD error interfaces.

#### [`nstep_return.py`](ch12/tools/nstep_return.py)
Computes standard n-step returns $G_{t:t+n}$ with bootstrapping for episodic MRPs.

#### [`nstep_truncated_TD_lam.py`](ch12/tools/nstep_truncated_TD_lam.py)
Computes truncated n-step TD($\lambda$) returns with detailed breakdowns of weighted contributions from each n-step return.

#### [`TD_lam_tabular_update.py`](ch12/tools/TD_lam_tabular_update.py)
Implements backward-view TD($\lambda$) with eligibility traces for tabular state-value functions. Updates all states in a single episode pass using accumulating traces.

#### [`eligibility_trace_update_accumulating_trace.py`](ch12/tools/eligibility_trace_update_accumulating_trace.py)
Step-by-step implementation of tabular TD($\lambda$) with accumulating eligibility traces. Demonstrates the full backward-view algorithm with detailed printing of TD errors, eligibility traces, and value updates at each step.

---

## Chapter 13: Policy Gradient Methods

### [`corridor_example.py`](ch13/corridor_example.py)
Monte Carlo simulation of Example 13.1 (Short corridor with switched actions). Estimates optimal policy parameters by averaging returns over many episodes for different action probabilities.

### Homework Solutions

#### [`q1__return_G0_single_terminal_reward.py`](ch13/hw/q1__return_G0_single_terminal_reward.py)
Computes the discounted return $G_0$ for a REINFORCE trajectory where only the final transition yields a non-zero reward.

#### [`q2__reinforce_W_update.py`](ch13/hw/q2__reinforce_W_update.py)
Implements REINFORCE policy gradient update for a Gaussian policy with linear mean function. Computes $\nabla_W \log \pi(a|s) = (a - Ws)s^T$ and updates parameters: $W \leftarrow W + \alpha G_t \nabla_W \log \pi(a_t|s_t)$.

---

## Installation

```bash
pip install -r requirements.txt
```


## **Chapter 1 — Introduction**

* Reinforcement learning problem setup
* Agent–environment interaction loop
* Rewards, returns, discounting
* Value functions vs policies
* Examples (tic-tac-toe, intuitive motivation)

---

## **Chapter 2 — Multi-armed Bandits**

* k-armed bandit problem
* Exploration vs exploitation
* Action-value estimation
* ε-greedy methods
* Optimistic initial values
* Upper Confidence Bound (UCB)
* Gradient bandit algorithms
* Regret (logarithmic regret)
* Thompson sampling
* Contextual bandits (associative search)

---

## **Chapter 3 — Finite Markov Decision Processes**

* Markov property
* States, actions, rewards
* Transition probabilities ( p(s',r|s,a) )
* Expected rewards ( r(s,a) )
* Episodic vs continuing tasks
* Policies (deterministic & stochastic)
* State-value ( v_\pi ) and action-value ( q_\pi )

---

## **Chapter 4 — Dynamic Programming**

* Policy evaluation
* Policy improvement
* Policy iteration
* Value iteration
* Bellman expectation equations
* Bellman optimality equations
* Generalized Policy Iteration (GPI)
* Synchronous vs asynchronous DP
* Model-based assumption

---

## **Chapter 5 — Monte Carlo Methods**

* Monte Carlo prediction
* First-visit vs every-visit MC
* MC estimation of action values
* Monte Carlo control
* Exploring starts
* ε-soft policies
* On-policy vs off-policy MC
* Importance sampling
* No bootstrapping

---

## **Chapter 6 — Temporal-Difference Learning**

* TD(0) prediction
* TD error
* Bootstrapping
* Sarsa
* Q-learning
* Expected Sarsa
* On-policy vs off-policy TD control
* Afterstates
* Batch TD vs batch MC
* Convergence properties

---

## **Chapter 7 — n-step Bootstrapping**

* n-step TD prediction
* n-step Sarsa
* n-step off-policy learning
* Importance sampling ratios
* Per-decision importance sampling
* Control variates
* Tree Backup algorithm
* Relationship between TD and MC

---

## **Chapter 8 — Planning and Learning with Tabular Methods**

* Models and planning
* Dyna architecture
* Integrated planning + learning
* Prioritized sweeping
* Expected vs sample updates
* Trajectory sampling
* Real-Time Dynamic Programming (RTDP)
* Heuristic search
* Rollout algorithms
* Monte Carlo Tree Search (MCTS)
* Decision-time planning
* Unification of planning and learning

---

## **Chapter 9 — On-policy Prediction with Approximation**

* Value-function approximation
* Feature representations
* Linear function approximation
* Prediction objective (VE / MSVE)
* Stochastic gradient descent
* Semi-gradient TD methods
* Gradient Monte Carlo
* n-step semi-gradient TD
* Feature types:

  * Polynomials
  * Fourier basis
  * Tile coding
  * Radial Basis Functions (RBFs)
* Least-Squares TD (LSTD)
* Interest and emphasis
* Stability considerations

---

## **Chapter 10 — On-policy Control with Approximation**

* Semi-gradient control
* Episodic semi-gradient Sarsa
* n-step semi-gradient Sarsa
* Mountain Car example
* Average-reward formulation
* Differential semi-gradient methods
* Continuing tasks vs episodic tasks

---

## **Chapter 12 — Eligibility Traces**

* Eligibility traces concept
* Forward view (λ-return)
* Backward view
* TD(λ)
* Sarsa(λ)
* Accumulating traces
* Replacing traces
* Dutch traces
* Bias–variance tradeoff via λ
* Implementation issues

---

## **Chapter 13 — Policy Gradient Methods**

* Policy approximation
* Advantages over value-based methods
* Policy Gradient Theorem
* REINFORCE algorithm
* Baselines for variance reduction
* Actor–Critic methods
* Policy gradients for continuing tasks
* Continuous action spaces
* Gaussian policies
* Score function (log-gradient)

---

## **Chapter 16 — Applications and Case Studies**

* TD-Gammon
* Samuel’s Checkers
* Watson’s Daily Double wagering
* Memory control optimization
* Human-level video game play
* AlphaGo
* AlphaGo Zero
* Personalized web services
* Robotics & control examples

---

## **Chapter 17 — Frontiers**

* General Value Functions (GVFs)
* Auxiliary prediction tasks
* Temporal abstraction
* Options framework
* Observations vs state
* Open research directions in RL

---

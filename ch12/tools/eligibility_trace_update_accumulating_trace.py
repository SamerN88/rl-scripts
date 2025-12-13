def run_td_lambda():
    # --- 1. Configuration Parameters ---
    gamma = 0.9      # Discount factor
    lam = 0.5        # Trace decay (lambda)
    alpha = 0.1      # Learning rate
    
    # Trace decay factor combined
    decay_factor = gamma * lam

    # --- 2. Initialize States ---
    # We use a dictionary for V (Values) and E (Eligibility Traces)
    # 'G' represents the Goal state
    states = [1, 2, 3, 4, 5]
    
    # Initial Value Estimates (as given in the problem)
    V = {
        1: 0.0,
        2: 0.0, 
        3: -1.0, 
        4: 0.0, 
        5: 0.0, 
        'G': 0.0
    }

    # Initial Eligibility Traces (all start at 0)
    E = {s: 0.0 for s in states}
    # We technically don't need a trace for G as it's terminal, 
    # but we initialize it for safety if needed.
    E['G'] = 0.0

    # --- 3. Define the Trajectory ---
    # Format: (Current State, Next State, Reward)
    trajectory = [
        (5, 4, 0),      # Transition 5 -> 4, Reward 0
        (4, 3, -0.1),   # Transition 4 -> 3, Reward -0.1
        (3, 2, -0.1),   # Transition 3 -> 2, Reward -0.1
        (2, 1, 0),      # Transition 2 -> 1, Reward 0
        (1, 'G', 1)     # Transition 1 -> Goal, Reward +1
    ]

    print(f"{'Step':<6} | {'Trans':<8} | {'Reward':<6} | {'Delta':<8} | {'State Values (V) after update'}")
    print("-" * 80)

    # --- 4. Process the Episode ---
    for step_i, (s_curr, s_next, reward) in enumerate(trajectory):
        
        # A. Update Eligibility Trace for the current state
        # Accumulating trace: e(s) = e(s) + 1
        E[s_curr] += 1
        
        # B. Calculate TD Error (delta)
        # delta = R + gamma * V(next) - V(current)
        v_next = V[s_next]
        v_curr = V[s_curr]
        delta = reward + (gamma * v_next) - v_curr
        
        # C. Update Values and Decay Traces for ALL states
        for s in states:
            # Update Value: V(s) <- V(s) + alpha * delta * e(s)
            V[s] += alpha * delta * E[s]
            
            # Decay Trace: e(s) <- gamma * lambda * e(s)
            E[s] *= decay_factor
            
        # Optional: Print progress
        v_str = ", ".join([f"{k}:{v:.3f}" for k,v in V.items() if k != 'G'])
        print(f"{step_i+1:<6} | {s_curr}->{s_next:<5} | {reward:<6} | {delta:<8.4f} | {v_str}")

    # --- 5. Final Output ---
    print("\n" + "="*30)
    print("FINAL VALUES (Rounded to 3 decimals)")
    print("="*30)
    for s in states:
        print(f"State {s}: {V[s]:.3f}")

if __name__ == "__main__":
    run_td_lambda()
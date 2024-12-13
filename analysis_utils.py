import numpy as np
import scipy
from collections import defaultdict, Counter

def basic_statistics(time_series):
    stats_dict = {
        'mean': np.mean(time_series),
        'variance': np.var(time_series),
        'std_dev': np.std(time_series)
    }
    return stats_dict

def create_transition_matrix(time_series):
    """
    Create first-order transition matrix from observed transitions
    
    Parameters:
    time_series: numpy array of the sum at each time step
    
    Returns:
    dict: Transition probabilities and unique states
    """
    # Count transitions
    transitions = defaultdict(lambda: defaultdict(int))
    unique_states = sorted(set(time_series))
    state_to_idx = {state: idx for idx, state in enumerate(unique_states)}
    
    # Count occurrences of each transition
    for t in range(len(time_series)-1):
        current = time_series[t]
        next_state = time_series[t+1]
        transitions[current][next_state] += 1
    
    # Create and fill transition matrix
    n_states = len(unique_states)
    trans_matrix = np.zeros((n_states, n_states))
    
    for i, state_i in enumerate(unique_states):
        total = sum(transitions[state_i].values())
        if total > 0:  # Avoid division by zero
            for j, state_j in enumerate(unique_states):
                trans_matrix[i,j] = transitions[state_i][state_j] / total
    
    return {
        'transition_matrix': trans_matrix,
        'states': unique_states,
        'state_to_idx': state_to_idx
    }

def calculate_expectations(time_series, transition_info):
    """
    Calculate expected value at each time step using the transition matrix
    
    Parameters:
    time_series: numpy array of the sum at each time step
    transition_info: dict containing transition_matrix, states, and state_to_idx
                    (output from create_transition_matrix function)
    
    Returns:
    numpy array: Expected values for each time step after the first
    """
    trans_matrix = transition_info['transition_matrix']
    states = transition_info['states']
    state_to_idx = transition_info['state_to_idx']
    
    # Initialize array for expected values
    expectations = np.zeros(len(time_series)-1)  # -1 because we can't predict first step
    
    # For each time step (except the last one)
    for t in range(len(time_series)-1):
        current_state = time_series[t]
        current_idx = state_to_idx[current_state]
        
        # Calculate E[X_{t+1} | X_t = x] = Σ (y * P(X_{t+1} = y | X_t = x))
        expectation = 0
        for next_state, next_idx in state_to_idx.items():
            expectation += next_state * trans_matrix[current_idx, next_idx]
            
        expectations[t] = expectation
    
    return expectations

def analyze_state_repetition(time_series):
    unique_states = set(time_series)
    state_counts = Counter(time_series)
    transitions = defaultdict(list)
    
    for t in range(len(time_series)-1):
        current = time_series[t]
        next_state = time_series[t+1]
        transitions[current].append(next_state)
    
    print(f"Total time steps: {len(time_series)}")
    print(f"Unique states observed: {len(unique_states)}")
    print(f"States that appear more than once: {sum(1 for count in state_counts.values() if count > 1)}")
    print(f"States with multiple different next states: {sum(1 for next_states in transitions.values() if len(set(next_states)) > 1)}")
    
    return state_counts, transitions

def analyze_markov_property(transitions, state_counts):
    # For states with multiple occurrences, analyze their next-state distributions
    markov_analysis = {}
    
    for state in transitions:
        if state_counts[state] >= 4:  # Only analyze states with enough data
            next_states = transitions[state]
            next_state_dist = Counter(next_states)
            
            # Calculate entropy of next-state distribution
            total = len(next_states)
            entropy = -sum((count/total) * np.log2(count/total) 
                         for count in next_state_dist.values())
            
            markov_analysis[state] = {
                'entropy': entropy,
                'unique_next_states': len(next_state_dist),
                'transitions': dict(next_state_dist)
            }
    
    return markov_analysis
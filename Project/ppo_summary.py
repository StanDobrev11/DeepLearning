import numpy as np

# Define softmax policy function
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

# Initial policy parameters (for 3 actions)
theta = np.array([0.2, 0.5, -0.3, 0.8])

# Compute action probabilities using softmax
probabilities = softmax(theta)
print("Initial probabilities:", probabilities)

# Assume the agent took action 1 (index 1) and got a positive reward
action_taken = 1
reward = 1.0  # Positive reinforcement

# Compute gradient of log probability
grad_log_pi = np.zeros_like(theta)
grad_log_pi[action_taken] = 1 - probabilities[action_taken]

# Policy update step (gradient ascent)
alpha = 0.1  # Learning rate
theta += alpha * reward * grad_log_pi

# Compute new probabilities after update
new_probabilities = softmax(theta)
print("Updated probabilities:", new_probabilities)

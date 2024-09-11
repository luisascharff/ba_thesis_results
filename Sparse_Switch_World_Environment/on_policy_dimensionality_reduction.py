# Import required libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC, TD3
from SwitchWorldContinuousSA import SwitchWorldContinuousSA
from sklearn.manifold import TSNE
from collections import defaultdict

# Configuration flag to include specific states
include_specific_states = False  # Set to True to include specific states
sample_from_policy = True

# List of algorithms to evaluate
algorithms = ["PPO", "Frequent Rewards PPO", "SAC", "TD3"]

# Create the environment
env = SwitchWorldContinuousSA(render_mode=False)

states = []
# Use existing sampled states or sample randomly from the environment
for _ in range(2000):  # Number of states to sample
    state = env.sample()
    states.append(state)
states = np.array(states)

# Function to extract representations
def extract_representation(model, state, algorithm):
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        if algorithm == "PPO" or algorithm == "Frequent Rewards PPO":
            features = model.policy.features_extractor(state_tensor)
        elif algorithm in ["SAC", "TD3"]:
            features = model.policy.actor.features_extractor(state_tensor)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        return features.squeeze(0).numpy().astype(np.float64)

# Loop through each algorithm and generate the t-SNE 2D plots
for algorithm in algorithms:
    # Load the model for the current algorithm
    if algorithm == "PPO":
        model = PPO.load("SB3_SW/agents/PPO_agents/models/PPO-trunc1000_CSCA/agent_5/PPO_step_1000000")
    #elif algorithm == "Frequent Rewards PPO":
        #model = PPO.load("SB3_SW/changed_rewards/PPO/models/PPO-trunc1000_CSCA_rew/agent_11/PPO_step_1000000")
    #elif algorithm == "SAC":
        #model = SAC.load("SB3_SW/agents/SAC_agents/models/SAC-trunc1000_CSCA/agent_9/SAC_step_1000000")
    #elif algorithm == "TD3":
        #model = TD3.load("SB3_SW/agents/TD3_agents/models/TD3-trunc1000_CSCA/agent_14/TD3_step_1000000")
    #else:
        #raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    # Initialize a dictionary to count on-policy states per switch state
    switch_state_count = defaultdict(int)

    # Sample states from the policy
    if sample_from_policy:
        policy_states = []
        for _ in range(100):  # Number of states to sample
            obs = env.reset()[0]
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _, _ = env.step(action)
                policy_states.append(obs)
                
                # Extract switch state (last two elements of the state)
                switch_state = tuple(obs[-2:])
                switch_state_count[switch_state] += 1  # Count on-policy state for the switch state
                
        policy_states = np.array(policy_states)
        print(f"Total on-policy states: {len(policy_states)}")
        print("On-policy state counts per switch state:")
        for switch_state, count in switch_state_count.items():
            print(f"Switch state {switch_state}: {count} states")
    
    # Extract representations for all sampled states (random and on-policy)
    random_representations = np.array([extract_representation(model, state, algorithm) for state in states])
    policy_representations = np.array([extract_representation(model, state, algorithm) for state in policy_states])

    # Combine random and on-policy representations for t-SNE
    all_representations = np.vstack([random_representations, policy_representations])

    # Apply t-SNE with 2 components (2D)
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    reduced_representations_tsne = tsne.fit_transform(all_representations)

    # Plotting the t-SNE 2D representation with colors based on x and y values of the states
    plt.figure(figsize=(8, 6))
    
    # Extract x and y positions of the random states
    x_positions_random = states[:, 0]  # Assuming x position is the first element in the state
    y_positions_random = states[:, 1]  # Assuming y position is the second element in the state
    color_values_random = np.sqrt(x_positions_random**2 + y_positions_random**2)

    # Extract x and y positions of the policy states
    x_positions_policy = policy_states[:, 0]
    y_positions_policy = policy_states[:, 1]
    color_values_policy = np.sqrt(x_positions_policy**2 + y_positions_policy**2)

    # Plot random states with one color map
    scatter_random = plt.scatter(reduced_representations_tsne[:len(random_representations), 0], 
                                 reduced_representations_tsne[:len(random_representations), 1], 
                                 c=color_values_random, cmap='Blues', alpha=0.9, label="Random States")

    # Plot on-policy states with another color map
    scatter_policy = plt.scatter(reduced_representations_tsne[len(random_representations):, 0], 
                                 reduced_representations_tsne[len(random_representations):, 1], 
                                 c=color_values_policy, cmap='Reds', alpha=0.6, label="On-Policy States")

    # Add color bar and legend
    plt.colorbar(scatter_random, label='Random States (x, y)')
    plt.colorbar(scatter_policy, label='On-Policy States (x, y)')
    plt.legend()

    # Title and labels
    plt.title(f"State Representations Random vs On-Policy")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    # Show the plot for the current algorithm
    plt.show()

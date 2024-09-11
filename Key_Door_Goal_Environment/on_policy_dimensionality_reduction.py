import torch
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from Key_changed_distances import KeyDoorGoalContinuousEnv
from sklearn.manifold import TSNE
from collections import defaultdict

include_specific_states = False
sample_from_policy = True

algorithms = ["PPO"]

env = KeyDoorGoalContinuousEnv(render_mode=False)

random_states = []
for _ in range(2000):
    state = env.sample()
    random_states.append(state)
random_states = np.array(random_states)

def extract_representation(model, state, algorithm):
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        if algorithm == "PPO":
            features = model.policy.features_extractor(state_tensor)
        return features.squeeze(0).numpy().astype(np.float64)

for algorithm in algorithms:
    model = PPO.load("SB3_Key/Changed_Distance/PPO/models/PPO-trunc1000_CSCA_dist/agent_8/PPO_step_1000000")

    switch_state_count = defaultdict(int)

    if sample_from_policy:
        policy_states = []
        for _ in range(30):
            obs = env.reset()[0]
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _, _ = env.step(action)
                policy_states.append(obs)
                switch_state = tuple(obs[-3:])
                switch_state_count[switch_state] += 1

        policy_states = np.array(policy_states)

        print(f"Total on-policy states: {len(policy_states)}")
        print("On-policy state counts per switch state:")
        for switch_state, count in switch_state_count.items():
            print(f"Switch state {switch_state}: {count} states")

    random_representations = np.array([extract_representation(model, state, algorithm) for state in random_states])
    policy_representations = np.array([extract_representation(model, state, algorithm) for state in policy_states])

    all_representations = np.vstack([random_representations, policy_representations])

    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    reduced_representations_tsne = tsne.fit_transform(all_representations)

    plt.figure(figsize=(8, 6))

    x_positions_random = random_states[:, 0]
    y_positions_random = random_states[:, 1]
    color_values_random = np.sqrt(x_positions_random**2 + y_positions_random**2)

    x_positions_policy = policy_states[:, 0]
    y_positions_policy = policy_states[:, 1]
    color_values_policy = np.sqrt(x_positions_policy**2 + y_positions_policy**2)

    scatter_random = plt.scatter(reduced_representations_tsne[:len(random_representations), 0],
                                 reduced_representations_tsne[:len(random_representations), 1],
                                 c=color_values_random, cmap='Blues', alpha=0.9, label="Random States")

    scatter_policy = plt.scatter(reduced_representations_tsne[len(random_representations):, 0],
                                 reduced_representations_tsne[len(random_representations):, 1],
                                 c=color_values_policy, cmap='Reds', alpha=0.6, label="On-Policy States")

    plt.colorbar(scatter_random, label='Random States (x, y)')
    plt.colorbar(scatter_policy, label='On-Policy States (x, y)')
    plt.legend()

    plt.title(f"State Representations Random vs On-Policy")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    plt.show()

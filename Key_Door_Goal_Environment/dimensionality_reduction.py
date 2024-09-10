import torch
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from Key_Door_Goal_Environment.Key_Door_Goal_MDP import KeyDoorGoalContinuousEnv
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

algorithms = ["PPO", "SAC", "TD3"]
sample_from_policy = True
include_specific_states = False


env = KeyDoorGoalContinuousEnv(render_mode=False)

states = []
for _ in range(2000): 
    state = env.sample()
    states.append(state)
states = np.array(states)


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


for algorithm in algorithms:
    if algorithm == "PPO":
        model = PPO.load("Key_Door_Goal_Environment/agents/PPO/models/PPO-trunc1000_CSCA_dist/agent_8/PPO_step_1000000")  
    elif algorithm == "SAC":
        model = SAC.load("Key_Door_Goal_Environment/agents/SAC/models/SAC-trunc1000_CSCA_dist/agent_10/SAC_step_1000000")
    elif algorithm == "TD3":
        model = TD3.load("Key_Door_Goal_Environment/agents/TD3/models/TD3-trunc1000_CSCA_dist/agent_4/TD3_step_1000000")

   
    if sample_from_policy:
        states = []
        for _ in range(100):  
            obs = env.reset()[0]
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _, _ = env.step(action)
                states.append(obs)
        states = np.array(states)
    
  
    representations = np.array([extract_representation(model, state, algorithm) for state in states])

 
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    reduced_representations_tsne = tsne.fit_transform(representations)

   
    plt.figure(figsize=(8, 6))
    
    
    x_positions = states[:, 0]  
    y_positions = states[:, 1]  

   
    color_values = np.sqrt(x_positions**2 + y_positions**2)  

   
    scatter = plt.scatter(reduced_representations_tsne[:, 0], reduced_representations_tsne[:, 1], 
                          c=color_values, cmap='viridis', alpha=0.8)
    

    plt.colorbar(scatter, label='Color based on state positions (x, y)')

   
    if include_specific_states:
        specific_states = [
            np.array([0.3, 0.3, 0, 0]),
            np.array([0.3, 0.3, 1, 0]),
            np.array([1.4, 1.4, 0, 0]),
            np.array([1.4, 1.4, 1, 0])
        ]
        state_labels = ["[0.3, 0.3, 0, 0]", "[0.3, 0.3, 1, 0]", "[1.4, 1.4, 0, 0]", "[1.4, 1.4, 1, 0]"]
        
        for i, specific_state in enumerate(specific_states):
            tsne_point = reduced_representations_tsne[-len(specific_states) + i]  
            plt.scatter(tsne_point[0], tsne_point[1], color='red', marker='X', s=100, label=f'State {state_labels[i]}')

    plt.title(f"t-SNE 2D of State Representations for {algorithm}")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    
    if include_specific_states:
        plt.legend()

   
    plt.show()

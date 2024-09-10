import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import torch
import matplotlib.pyplot as plt
from Key_Door_Goal_Environment.Key_Door_Goal_MDP import KeyDoorGoalContinuousEnv
from stable_baselines3 import PPO, SAC, TD3
import os
from tqdm import tqdm 


algorithm = "PPO"  
if algorithm == "PPO":
    model = PPO.load("SB3_SW/agents/PPO_agents/models/PPO-trunc1000_CSCA/agent_5/PPO_step_1000000")
elif algorithm == "SAC":
    model = SAC.load("SB3_SW/agents/SAC_agents/models/SAC-trunc1000_CSCA/agent_9/SAC_step_1000000")
elif algorithm == "TD3":
    model = TD3.load("SB3_SW/agents/TD3_agents/models/TD3-trunc1000_CSCA/agent_14/TD3_step_1000000")
else:
    raise ValueError("Unsupported algorithm. Choose 'PPO', 'SAC', or 'TD3'.")


env = KeyDoorGoalContinuousEnv(render_mode=False)


eps_values = np.arange(0.1, 1.0, 0.1)
min_samples_values = np.arange(5, 1000, 10)


results = []


states = []
for _ in range(1000): 
    state = env.sample()
    states.append(state)
states = np.array(states)


def extract_representation(model, state):
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        if algorithm == "PPO":
            features = model.policy.features_extractor(state_tensor)
        elif algorithm in ["SAC", "TD3"]:
            features = model.policy.actor.features_extractor(state_tensor)
        return features.squeeze(0).numpy().astype(np.float64)

representations = np.array([extract_representation(model, state) for state in states])
total_iterations = len(eps_values) * len(min_samples_values)


with tqdm(total=total_iterations, desc="DBSCAN Hyperparameter Search") as pbar:
    for eps in eps_values:
        for min_samples in min_samples_values:
            
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan_labels = dbscan.fit_predict(representations)  

            
            if len(set(dbscan_labels)) > 1:  
                silhouette_avg = silhouette_score(representations, dbscan_labels)
                davies_bouldin_avg = davies_bouldin_score(representations, dbscan_labels)
            else:
                silhouette_avg = -1  
                davies_bouldin_avg = np.inf  
            
            num_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)  # Exclude noise (-1)

            
            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'silhouette_score': silhouette_avg,
                'davies_bouldin_score': davies_bouldin_avg,
                'num_clusters': num_clusters
            })

            
            pbar.update(1)


results_df = pd.DataFrame(results)


sorted_results = results_df.sort_values(by='silhouette_score', ascending=False)


print("Best result based on silhouette score:")
print(sorted_results.iloc[0])


print("All results sorted by silhouette score:")
print(sorted_results)



total_results = len(results_df)


four_cluster_results = results_df[results_df['num_clusters'] == 4]


percentage_four_clusters = (len(four_cluster_results) / total_results) * 100


mean_silhouette = four_cluster_results['silhouette_score'].mean()
min_silhouette = four_cluster_results['silhouette_score'].min()
max_silhouette = four_cluster_results['silhouette_score'].max()


mean_min_samples = four_cluster_results['min_samples'].mean()
min_min_samples = four_cluster_results['min_samples'].min()
max_min_samples = four_cluster_results['min_samples'].max()


mean_eps = four_cluster_results['eps'].mean()
min_eps = four_cluster_results['eps'].min()
max_eps = four_cluster_results['eps'].max()


print(f"Percentage of results with 4 clusters: {percentage_four_clusters:.2f}%")
print(f"Mean silhouette score for 4 clusters: {mean_silhouette:.2f}")
print(f"Min silhouette score for 4 clusters: {min_silhouette:.2f}")
print(f"Max silhouette score for 4 clusters: {max_silhouette:.2f}")

print(f"Mean min_samples for 4 clusters: {mean_min_samples:.2f}")
print(f"Min min_samples for 4 clusters: {min_min_samples}")
print(f"Max min_samples for 4 clusters: {max_min_samples}")

print(f"Mean eps for 4 clusters: {mean_eps:.2f}")
print(f"Min eps for 4 clusters: {min_eps:.2f}")
print(f"Max eps for 4 clusters: {max_eps:.2f}")



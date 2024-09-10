import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC, TD3
from Sparse_Switch_World_Environment.Sparse_Switch_World_MDP import SwitchWorldContinuousSA
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist

dim_red = True
sample_from_policy = True  


algorithm = "PPO"  
if algorithm == "PPO":
    model = PPO.load("SB3_SW/changed_rewards/PPO/models/PPO-trunc1000_CSCA_rew/agent_11/PPO_step_1000000")
elif algorithm == "SAC":
    model = SAC.load("SB3_SW/agents/SAC_agents/models/SAC-trunc1000_CSCA/agent_9/SAC_step_1000000")
elif algorithm == "TD3":
    model = TD3.load("SB3_SW/agents/TD3_agents/models/TD3-trunc1000_CSCA/agent_14/TD3_step_1000000")
else:
    raise ValueError("Unsupported algorithm. Choose 'PPO', 'SAC', or 'TD3'.")


env = SwitchWorldContinuousSA(render_mode=False)


states = []
if sample_from_policy:
    for _ in range(100):  
        obs = env.reset()[0]
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)
            states.append(obs)
else:
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


dbscan = DBSCAN(eps=0.69, min_samples=82)
dbscan_labels = dbscan.fit_predict(representations)  


dbscan_clusters = {i: [] for i in range(len(set(dbscan_labels)))}
for i, label in enumerate(dbscan_labels):
    if label != -1: 
        dbscan_clusters[label].append(states[i])


print("DBSCAN Clustering:")
for cluster_id, cluster_states in dbscan_clusters.items():
    print(f"Cluster {cluster_id}:")
    for state in cluster_states[:10]:  
        print(state)
    print("\n")  

if dim_red:
    
    pca = PCA(n_components=2)
    reduced_representations = pca.fit_transform(representations)

    
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(representations)

    
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 1, 1)
    plt.scatter(reduced_representations[:, 0], reduced_representations[:, 1], c=dbscan_labels, cmap='viridis')
    plt.title("PCA of State Representations - DBSCAN")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()

    
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 1, 1)
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=dbscan_labels, cmap='viridis')
    plt.title("t-SNE of State Representations - DBSCAN")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()

def calculate_patterns(states, labels):
    patterns = {
        "[x, y, 1, 1]": [1, 1],
        "[x, y, 1, 0]": [1, 0],
        "[x, y, 0, 0]": [0, 0]
    }

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    for cluster_id in range(n_clusters):
        cluster_states = states[labels == cluster_id]
        total_count = len(cluster_states)
        print(f"\nCluster {cluster_id} (Total: {total_count} states):")

        for pattern_name, pattern_values in patterns.items():
            pattern_count = np.sum((cluster_states[:, 2:] == pattern_values).all(axis=1))
            percentage = (pattern_count / total_count) * 100 if total_count > 0 else 0
            print(f"  {pattern_name}: {pattern_count} states ({percentage:.2f}%)")

print("\nDBSCAN Clustering Patterns:")
valid_dbscan_labels = dbscan_labels[dbscan_labels != -1]
valid_dbscan_states = states[dbscan_labels != -1]
calculate_patterns(valid_dbscan_states, valid_dbscan_labels)

'''
Silhouette Coefficient 
'''
from sklearn.metrics import silhouette_score


valid_dbscan_labels = dbscan_labels[dbscan_labels != -1]
valid_dbscan_representations = representations[dbscan_labels != -1]


if len(valid_dbscan_labels) > 1: 
    dbscan_silhouette_score = silhouette_score(valid_dbscan_representations, valid_dbscan_labels)
    print(f"Silhouette Score for DBSCAN: {dbscan_silhouette_score:.4f}")
else:
    print("Not enough clusters to calculate a silhouette score for DBSCAN.")

def find_min_max_distances_all_clusters(states, labels, clusters):
    min_max_results = {}

    
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            cluster1 = clusters[i]
            cluster2 = clusters[j]

            
            if cluster1 != cluster2:
                cluster1_states = states[labels == cluster1]
                cluster2_states = states[labels == cluster2]

               
                distances = cdist(cluster1_states, cluster2_states, metric='euclidean')

               
                min_dist_idx = np.unravel_index(distances.argmin(), distances.shape)
                max_dist_idx = np.unravel_index(distances.argmax(), distances.shape)

                min_dist = distances[min_dist_idx]
                max_dist = distances[max_dist_idx]

                min_dist_state1 = cluster1_states[min_dist_idx[0]]
                min_dist_state2 = cluster2_states[min_dist_idx[1]]

                max_dist_state1 = cluster1_states[max_dist_idx[0]]
                max_dist_state2 = cluster2_states[max_dist_idx[1]]

                print(f"Minimum distance ({min_dist:.4f}) between clusters {cluster1} and {cluster2}:")
                print(f"State from Cluster {cluster1}: {min_dist_state1}")
                print(f"State from Cluster {cluster2}: {min_dist_state2}\n")

                print(f"Maximum distance ({max_dist:.4f}) between clusters {cluster1} and {cluster2}:")
                print(f"State from Cluster {cluster1}: {max_dist_state1}")
                print(f"State from Cluster {cluster2}: {max_dist_state2}\n")

           
                min_max_results[(cluster1, cluster2)] = {
                    'min_distance': min_dist,
                    'min_states': (min_dist_state1, min_dist_state2),
                    'max_distance': max_dist,
                    'max_states': (max_dist_state1, max_dist_state2)
                }

    return min_max_results


clusters = list(set(valid_dbscan_labels))
min_max_distances = find_min_max_distances_all_clusters(states, dbscan_labels, clusters)


if dim_red:
   
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    reduced_representations = tsne.fit_transform(representations)

   
    smallest_min_dist = float('inf')
    largest_max_dist = float('-inf')
    smallest_pair = None
    largest_pair = None

    for (cluster1, cluster2), result in min_max_distances.items():
        min_dist = result['min_distance']
        max_dist = result['max_distance']

        if min_dist < smallest_min_dist:
            smallest_min_dist = min_dist
            smallest_pair = (cluster1, cluster2)

        if max_dist > largest_max_dist:
            largest_max_dist = max_dist
            largest_pair = (cluster1, cluster2)

    
    plt.figure(figsize=(14, 6))

  
    plt.scatter(reduced_representations[:, 0], reduced_representations[:, 1], color='grey', alpha=0.3)

  
    for cluster_id in set(valid_dbscan_labels):
       
        cluster_points = reduced_representations[valid_dbscan_labels == cluster_id]
        centroid = cluster_points.mean(axis=0)
        plt.text(centroid[0], centroid[1], str(cluster_id), fontsize=20, color='black', ha='center')


    
    if smallest_pair:
        min_states = min_max_distances[smallest_pair]['min_states']
        min_state_indices = [np.where((states == min_state).all(axis=1))[0][0] for min_state in min_states]

        print(f"Smallest minimum distance states between clusters {smallest_pair[0]} and {smallest_pair[1]}:")
        print(f"State from Cluster {smallest_pair[0]}: {min_states[0]}")
        print(f"State from Cluster {smallest_pair[1]}: {min_states[1]}")

        plt.scatter(reduced_representations[min_state_indices, 0], reduced_representations[min_state_indices, 1],
                    color='blue', marker='o', s=100, label=f'Smallest Min Distance (Clusters {smallest_pair[0]} & {smallest_pair[1]})')
        
        plt.plot(reduced_representations[min_state_indices, 0], reduced_representations[min_state_indices, 1],
                 color='blue', linestyle='--', linewidth=2)
        
   
    if largest_pair:
        max_states = min_max_distances[largest_pair]['max_states']
        max_state_indices = [np.where((states == max_state).all(axis=1))[0][0] for max_state in max_states]

        print(f"Largest maximum distance states between clusters {largest_pair[0]} and {largest_pair[1]}:")
        print(f"State from Cluster {largest_pair[0]}: {max_states[0]}")
        print(f"State from Cluster {largest_pair[1]}: {max_states[1]}")

        plt.scatter(reduced_representations[max_state_indices, 0], reduced_representations[max_state_indices, 1],
                    color='red', marker='x', s=100, label=f'Largest Max Distance (Clusters {largest_pair[0]} & {largest_pair[1]})')
      
        plt.plot(reduced_representations[max_state_indices, 0], reduced_representations[max_state_indices, 1],
                 color='red', linestyle='--', linewidth=2)
        
    
    plt.title("t-SNE of State Representations with Smallest and Largest Distances")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(loc='best')
    plt.show()

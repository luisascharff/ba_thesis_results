import numpy as np
import matplotlib.pyplot as plt


algorithms = ["PPO", "Adapted Rewards PPO", "SAC", "TD3"]
colors = {"PPO": "blue", "Adapted Rewards PPO": "blue", "SAC": "red", "TD3": "green"}


reward_files = {
    "PPO": 'Key_Door_Goal_Environment/agents/PPO/rewards/all_rewards_PPO_Key_dist.npy',
    "Adapted Rewards PPO": 'Key_Door_Goal_Environment/Sparse_Key_Door_Goal/PPO/rewards/all_rewards_PPO_Key_rew.npy',
    "SAC": 'Key_Door_Goal_Environment/agents/SAC/rewards/all_rewards_SAC_Key_dist.npy',
    "TD3": 'Key_Door_Goal_Environment/agents/TD3/rewards/all_rewards_TD3_Key_rew.npy'
}


for algo in algorithms:
    
    reward_file = reward_files[algo]
    all_rewards_test = np.load(reward_file, allow_pickle=True).item()

    
    if algo == "Adapted Rewards PPO":
        rewards_array = np.array(all_rewards_test['PPO']) 
    else:
        rewards_array = np.array(all_rewards_test[algo])

    
    mean_rewards = np.mean(rewards_array, axis=0)
    std_rewards = np.std(rewards_array, axis=0)
    n_agents = rewards_array.shape[0]
    standard_error = std_rewards / np.sqrt(n_agents)
    confidence_interval = 1.96 * standard_error

    global_min_y = np.min(mean_rewards - confidence_interval)
    global_max_y = np.max(mean_rewards + confidence_interval)

   
    plt.figure(figsize=(10, 5))
    plt.plot(mean_rewards, label=f'{algo} Average Reward', color=colors[algo])
    plt.fill_between(range(len(mean_rewards)), mean_rewards - confidence_interval, mean_rewards + confidence_interval, color=colors[algo], alpha=0.1)

    plt.title(f'{algo} Learning Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    
    plt.ylim(global_min_y - 10, global_max_y + 10)
    plt.ylabel('Average Reward')
    plt.xlabel('Training Steps (x 10,000 timesteps)')

   
    plt.show()

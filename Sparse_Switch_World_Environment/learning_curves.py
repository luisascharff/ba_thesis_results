import numpy as np
import matplotlib.pyplot as plt


algorithms = ["PPO", "Adapted Rewards PPO", "SAC", "TD3"]
colors = {"PPO": "blue", "Adapted Rewards PPO": "blue", "SAC": "red", "TD3": "green"}


reward_files = {
    "PPO": 'SB3_SW/agents/PPO_agents/rewards/all_rewards_PPO.npy',
    "Adapted Rewards PPO": 'SB3_SW/changed_rewards/PPO/rewards/all_rewards_PPO_Key_rew.npy',
    "SAC": 'SB3_SW/agents/SAC_agents/rewards/all_rewards_SAC.npy',
    "TD3": 'SB3_SW/agents/TD3_agents/rewards/all_rewards_TD3.npy'
}


agents_to_remove_adapted_ppo = [5, 13, 21, 19]  


for algo in algorithms:
    
    reward_file = reward_files[algo]
    all_rewards_test = np.load(reward_file, allow_pickle=True).item()

   
    if algo == "Adapted Rewards PPO":
        rewards_array = np.array(all_rewards_test['PPO'])  

        
        agents_to_remove_adapted_ppo.sort(reverse=True)  
        for agent_index in agents_to_remove_adapted_ppo:
            if agent_index < len(rewards_array):
                rewards_array = np.delete(rewards_array, agent_index, axis=0)
            else:
                print(f"Agent index {agent_index} is out of range for {algo}.")
    else:
        rewards_array = np.array(all_rewards_test[algo])

    
    mean_rewards = np.mean(rewards_array, axis=0)
    std_rewards = np.std(rewards_array, axis=0)
    n_agents = rewards_array.shape[0]
    standard_error = std_rewards / np.sqrt(n_agents)
    confidence_interval = 1.96 * standard_error

    global_min_y = np.min(mean_rewards - confidence_interval) - 1000
    global_max_y = np.max(mean_rewards + confidence_interval) + 1000

    
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

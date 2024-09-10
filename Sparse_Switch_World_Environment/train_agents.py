import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from tqdm import trange
from Sparse_Switch_World_MDP import SwitchWorldContinuousSA


# Make environment: set render_mode to True to visualize
env = SwitchWorldContinuousSA(render_mode=False)
check_env(env)


algorithms = ["PPO", "TD3", "SAC"]
num_agents = 3 
total_timesteps_per_agent = 1_000_000
increment_timesteps = 10_000
num_increments = total_timesteps_per_agent // increment_timesteps
all_rewards = {algo: [] for algo in algorithms}

for algo in algorithms:
    
    base_folder_name = f"{algo}-trunc1000_CSCA"
    base_models_dir = f"SB3_SW/changed_distances/{algo}_test/models/{base_folder_name}"
    base_logdir = f"SB3_SW/changed_distances/{algo}_test/logs/{base_folder_name}"
    
    for i in trange(num_agents, desc=f"{algo} agents"):
        
        agent_folder_name = f"agent_{i+1}"
        models_dir = os.path.join(base_models_dir, agent_folder_name)
        logdir = os.path.join(base_logdir, agent_folder_name)
        
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(logdir, exist_ok=True)
        
        
        monitored_env = Monitor(env)
        
        # Create the agent
        if algo == "SAC":
            model = SAC("MlpPolicy", monitored_env, verbose=0, tensorboard_log=logdir)
        elif algo == "PPO":
            model = PPO("MlpPolicy", monitored_env, verbose=0, tensorboard_log=logdir)
        elif algo == "TD3":
            model = TD3("MlpPolicy", monitored_env, verbose=0, tensorboard_log=logdir)
        else:
            raise ValueError(f"Unknown algorithm: {algo}")
        
        agent_rewards = []

        for k in range(num_increments):
            # Train the agent
            model.learn(total_timesteps=increment_timesteps, reset_num_timesteps=False, tb_log_name=f"{agent_folder_name}")
            
            # Save the model after each increment
            model.save(f"{models_dir}/{algo}_step_{increment_timesteps * (k + 1)}")
            
            # Record the rewards
            rewards = monitored_env.get_episode_rewards()
            agent_rewards.append(np.mean(rewards) if rewards else 0)
        
        all_rewards[algo].append(agent_rewards)
        np.save(f'SB3_SW/changed_distances/{algo}_test/rewards/all_rewards_{algo}_agent_{i+1}_dist.npy', all_rewards)

# Convert rewards to numpy arrays for easier manipulation
for algo in algorithms:
    all_rewards[algo] = np.array(all_rewards[algo])


np.save(f'Sparse_Switch_World_Environment/agents/{algo}/rewards/all_rewards_{algo}.npy', all_rewards)

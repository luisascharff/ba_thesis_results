import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from tqdm import trange
from Key_Door_Goal_MDP import KeyDoorGoalContinuousEnv


env = KeyDoorGoalContinuousEnv(render_mode=False)
check_env(env)


algorithms = ["PPO", "SAC", "TD3"]
num_agents = 20
total_timesteps_per_agent = 1_000_000
increment_timesteps = 10_000  
num_increments = total_timesteps_per_agent // increment_timesteps


all_rewards = {algo: [] for algo in algorithms}  

for algo in algorithms:
    base_folder_name = f"{algo}-trunc1500_CSCA"
    base_models_dir = f"Key_Door_Goal_Environment/Sparse_Key_Door_Goal/{algo}_new/models/{base_folder_name}"
    base_logdir = f"Key_Door_Goal_Environment/Sparse_Key_Door_Goal/{algo}_new/logs/{base_folder_name}"
    
    for i in trange(num_agents, desc=f"{algo} agents"):
       
        agent_folder_name = f"agent_{i+1}"
        models_dir = os.path.join(base_models_dir, agent_folder_name)
        logdir = os.path.join(base_logdir, agent_folder_name)
        
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(logdir, exist_ok=True)
        
        
        monitored_env = Monitor(env)
        
      
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
           
            model.learn(total_timesteps=increment_timesteps, reset_num_timesteps=False, tb_log_name=f"{agent_folder_name}")
            
          
            model.save(f"{models_dir}/{algo}_step_{increment_timesteps * (k + 1)}")
            
          
            rewards = monitored_env.get_episode_rewards()
            agent_rewards.append(np.mean(rewards) if rewards else 0)
        
        all_rewards[algo].append(agent_rewards)
        np.save(f'Key_Door_Goal_Environment/Sparse_Key_Door_Goal/{algo}_new/rewards/all_rewards_{algo}_agent_{i+1}_Key_rew.npy', all_rewards)


for algo in algorithms:
    all_rewards[algo] = np.array(all_rewards[algo])


np.save(f'Key_Door_Goal_Environment/Sparse_Key_Door_Goal/{algo}_new/rewards/all_rewards_{algo}_Key_rew.npy', all_rewards)


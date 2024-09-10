import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from stable_baselines3 import PPO, SAC, TD3
from Sparse_Switch_World_Environment.Sparse_Switch_World_MDP import SwitchWorldContinuousSA
from stable_baselines3.common.evaluation import evaluate_policy


algorithms = ["PPO", "SAC", "TD3"]


trajectories = {}
steps_list = {}


initial_state = np.array([0.3, 1.4, 0, 0])  

for algo in algorithms:
    if algo == "PPO":
        model_path = "SB3_SW/changed_rewards/PPO/models/PPO-trunc1000_CSCA_rew/agent_11/PPO_step_1000000"
        model = PPO.load(model_path)
    elif algo == "SAC":
        model_path = "SB3_SW/agents/SAC_agents/models/SAC-trunc1000_CSCA/agent_9/SAC_step_1000000"
        model = SAC.load(model_path)
    elif algo == "TD3":
        model_path = "SB3_SW/agents/TD3_agents/models/TD3-trunc1000_CSCA/agent_14/TD3_step_1000000"
        model = TD3.load(model_path)
    

    env = SwitchWorldContinuousSA(render_mode=False)
    
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, deterministic=True)
    print(f"Mean reward for {algo}: {mean_reward} +/- {std_reward}")

    
    obs = env.reset(state=initial_state)[0]
    trajectory = [obs[:2]] 
    done = False
    
    steps = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        steps += 1
        trajectory.append(obs[:2])  

    trajectories[algo] = np.array(trajectory)
    steps_list[algo] = steps 
    print (f"Number of steps taken by {algo}: {steps}")


plt.figure(figsize=(8, 8))
plt.style.use('seaborn-v0_8-whitegrid')  


colors = {'PPO': '#FF6347', 'SAC': '#4682B4', 'TD3': '#32CD32'}  

for algo, traj in trajectories.items():
    plt.plot(traj[:, 0], traj[:, 1], label=f"{algo}", ls='--', color=colors[algo], linewidth=2.5, alpha=0.8)


for i, loc in enumerate(env._switch_locations):
    color = 'black' if i == 0 else 'darkgrey'  
    plt.scatter(loc[0], loc[1], color=color, s=300, label=f'Switch {i+1}')
    
    
    circle = Circle(loc, radius=0.3, color='black', fill=False, linestyle='dotted', linewidth=2)
    plt.gca().add_patch(circle)


plt.xlim(0, env._size)
plt.ylim(0, env._size)
plt.xlabel('x', fontsize=14, color='black')
plt.ylabel('y', fontsize=14, color='black')
plt.title('Agent Trajectories', fontsize=18, color='black')


plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=14, loc='lower right')
plt.gca().set_facecolor('#f9f9f9') 


for spine in plt.gca().spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.5)

plt.show()

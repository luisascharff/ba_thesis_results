import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from stable_baselines3 import PPO, SAC, TD3
from Key_Door_Goal_Environment.Key_Door_Goal_MDP import KeyDoorGoalContinuousEnv
from stable_baselines3.common.evaluation import evaluate_policy


algorithms = ["PPO", "TD3", "SAC"]


trajectories = {}
steps_lst = {}


initial_state = np.array([1.7, 0.5, 0, 0, 0])  

for algo in algorithms:
    if algo == "PPO":
        model_path = "Key_Door_Goal_Environment/agents/PPO/models/PPO-trunc1000_CSCA_dist/agent_8/PPO_step_1000000"
        model = PPO.load(model_path)
    elif algo == "SAC":
        model_path = "Key_Door_Goal_Environment/agents/SAC/models/SAC-trunc1000_CSCA_dist/agent_19/SAC_step_1000000"
        model = SAC.load(model_path)
    elif algo == "TD3":
        model_path = "Key_Door_Goal_Environment/agents/TD3/models/TD3-trunc1000_CSCA_dist/agent_4/TD3_step_1000000"
        model = TD3.load(model_path)
    

    env = KeyDoorGoalContinuousEnv(render_mode=False)
    
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, deterministic=True)
    print(f"Mean reward for {algo}: {mean_reward} +/- {std_reward}")

    
    obs = env.reset(state=initial_state)[0]
    trajectory = [obs[:2]]  
    done = False
    truncated = False
    
    step = 0
    while not (done or truncated):  
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        trajectory.append(obs[:2])  
        step += 1

   
    if done and not truncated:
        steps_lst[algo] = step
        trajectories[algo] = np.array(trajectory)
        print(f"Number of steps taken by {algo} to solve the environment: {step}")
    else:
        print(f"{algo} did not solve the environment in this episode.")


plt.figure(figsize=(8, 8))
plt.style.use('seaborn-v0_8-whitegrid')  


colors = {'PPO': '#FF6347', 'SAC': '#4682B4', 'TD3': '#32CD32'}  

for algo, traj in trajectories.items():
    plt.plot(traj[:, 0], traj[:, 1], label=f"{algo}", ls='--', color=colors[algo], linewidth=2.5, alpha=0.8)


plt.scatter(*env.key_position, color='#1E90FF', s=300, label='Key')  
plt.scatter(*env.door_position, color='grey', s=300, label='Door')   
plt.scatter(*env.goal_position, color='#000080', s=300, label='Goal')  


circle_radius = 0.2
circle_kwargs = {'fill': False, 'linestyle': 'dotted', 'linewidth': 2.0, 'edgecolor': 'black'}

key_circle = Circle(env.key_position, circle_radius, **circle_kwargs)
door_circle = Circle(env.door_position, circle_radius, **circle_kwargs)
goal_circle = Circle(env.goal_position, circle_radius, **circle_kwargs)

plt.gca().add_patch(key_circle)
plt.gca().add_patch(door_circle)
plt.gca().add_patch(goal_circle)


plt.xlim(0, env.size)
plt.ylim(0, env.size)
plt.xlabel('x', fontsize=14, color='black')
plt.ylabel('y', fontsize=14, color='black')
plt.title('Agent Trajectories (Successful Episodes)', fontsize=18, color='black')

plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=14, loc='lower right')
plt.gca().set_facecolor('#f9f9f9') 


for spine in plt.gca().spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.5)

plt.show()

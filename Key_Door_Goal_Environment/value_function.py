import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC, TD3
import torch
from Key_Door_Goal_Environment.Key_Door_Goal_MDP import KeyDoorGoalContinuousEnv
from matplotlib.patches import Circle


plot_3d = False #
algorithm = "SAC"  


if algorithm == "PPO":
    
    model = PPO.load("SB3_Key/Changed_Distance/PPO/models/PPO-trunc1000_CSCA_dist/agent_8/PPO_step_1000000")
elif algorithm == "SAC":
   
    model = SAC.load("SB3_Key/Changed_Distance/SAC/models/SAC-trunc1000_CSCA_dist/agent_19/SAC_step_1000000")
elif algorithm == "TD3":
    
    model = TD3.load("SB3_Key/Changed_Distance/TD3/models/TD3-trunc1000_CSCA_dist/agent_4/TD3_step_1000000")
else:
    raise ValueError("Unsupported algorithm. Choose 'PPO', 'SAC', or 'TD3'.")


env = KeyDoorGoalContinuousEnv(render_mode=False)


x_vals = np.linspace(0, env.size, 100)
y_vals = np.linspace(0, env.size, 100)
X, Y = np.meshgrid(x_vals, y_vals)


switch_states = [
    np.array([0, 0, 0]),
    np.array([1, 0, 0]),
    np.array([1, 1, 0])
]

key_location = env.key_position
door_location = env.door_position
goal_location = env.goal_position


n_plots = len(switch_states)


key_colors = ['#000000', '#FFD700'] 
door_colors = ['#00FFFF', '#FF4500']
goal_colors = ['#4B0082', '#32CD32']  

if plot_3d:
    plt.style.use('_mpl-gallery')
    fig, axs = plt.subplots(1, n_plots, subplot_kw={"projection": "3d"}, figsize=(5 * n_plots, 6))

    for idx, switch_state in enumerate(switch_states):
        Z = np.zeros(X.shape)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                
                state = np.array([X[i, j], Y[i, j], 0.0, *switch_state]) 

                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                
                if algorithm == "PPO":
                    value = model.policy.predict_values(state_tensor)
                    Z[i, j] = value.item()
                
                if algorithm in ["SAC", "TD3"]:
                    action = torch.tensor([0.0, 0.0, 0.0, 0.0]).unsqueeze(0)
                    q_value1, q_value2 = model.critic(state_tensor, action)
                    Z[i, j] = torch.min(q_value1, q_value2).item()

        ax = axs[idx] if n_plots > 1 else axs
        wire = ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10, color='black', linewidth=0.5)
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('Value Function')
       

        max_z = Z.max()
        ax.scatter(key_location[0], key_location[1], max_z, color=key_colors[switch_state[0]], s=100, label='Key')
        ax.scatter(door_location[0], door_location[1], max_z, color=door_colors[switch_state[1]], s=100, label='Door')
        ax.scatter(goal_location[0], goal_location[1], max_z, color=goal_colors[switch_state[2]], s=100, label='Goal')

        ax.legend()
        ax.set(xticklabels=[], yticklabels=[], zticklabels=[])

    plt.show()

else:
    fig, axs = plt.subplots(1, n_plots, figsize=(5 * n_plots, 6))

    for idx, switch_state in enumerate(switch_states):
        Z = np.zeros(X.shape)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                
                state = np.array([X[i, j], Y[i, j], switch_state[0], switch_state[1], switch_state[2]])
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                
                if algorithm == "PPO":
                    value = model.policy.predict_values(state_tensor)
                    Z[i, j] = value.item()
                
                if algorithm in ["SAC", "TD3"]:
                    action = torch.tensor([0.0, 0.0, 0.0, 0.0]).unsqueeze(0)
                    q_value1, q_value2 = model.critic(state_tensor, action)
                    Z[i, j] = torch.min(q_value1, q_value2).item()
        
        ax = axs[idx] if n_plots > 1 else axs
        im = ax.imshow(Z, extent=(0, env.size, 0, env.size), origin='lower', cmap='viridis')
        
        ax.scatter(key_location[0], key_location[1], color=key_colors[switch_state[0]], s=100, label='Key')
        ax.scatter(door_location[0], door_location[1], color=door_colors[switch_state[1]], s=100, label='Door')
        ax.scatter(goal_location[0], goal_location[1], color=goal_colors[switch_state[2]], s=100, label='Goal')
        
        circle = Circle([key_location[0], key_location[1]], radius=0.2, color='black', fill=False, linestyle='dotted')
        ax.add_patch(circle)
        
        circle = Circle([door_location[0], door_location[1]], radius=0.2, color='black', fill=False, linestyle='dotted')
        ax.add_patch(circle)
        
        circle = Circle([goal_location[0], goal_location[1]], radius=0.2, color='black', fill=False, linestyle='dotted')
        ax.add_patch(circle)
        
        
        ax.legend()

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Value Function')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        
    plt.show()

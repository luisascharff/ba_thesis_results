import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from matplotlib.patches import Circle
from Sparse_Switch_World_Environment.Sparse_Switch_World_MDP import SwitchWorldContinuousSA
import torch
from matplotlib.patches import Rectangle
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import PPO, SAC, TD3

# Configuration flags
plot_3d = False 
algorithm = "TD3"  

# Load the chosen algorithm's model
if algorithm == "PPO":
    model = PPO.load("Sparse_Switch_World_Environment/agents/PPO/models/PPO-trunc1000_CSCA/agent_3/PPO_step_1000000.zip")
elif algorithm == "SAC":
    model = SAC.load("Sparse_Switch_World_Environment/agents/SAC/models/SAC-trunc1000_CSCA/agent_1/SAC_step_1000000.zip")
elif algorithm == "TD3":
    model = TD3.load("Sparse_Switch_World_Environment/agents/TD3/models/TD3-trunc1000_CSCA/agent_18/TD3_step_1000000")
else:
    raise ValueError("Unsupported algorithm. Choose 'PPO', 'SAC', or 'TD3'.")

# Create the environment
env = SwitchWorldContinuousSA(render_mode=False, switch_width=0.2)

# Define grid resolution
x_vals = np.linspace(0, env._size, 100)
y_vals = np.linspace(0, env._size, 100)
X, Y = np.meshgrid(x_vals, y_vals)

# Define the switch states to analyze
switch_states = [np.array([0, 0]), np.array([1, 0])]

if plot_3d:
    plt.style.use('_mpl-gallery')
    fig, axs = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(14, 6))

    for idx, switch_state in enumerate(switch_states):
        Z = np.zeros(X.shape)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                state = np.array([X[i, j], Y[i, j], *switch_state])
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                
                if algorithm == "PPO":
                    value = model.policy.predict_values(state_tensor)
                    Z[i, j] = value.item()
                if algorithm in ["SAC", "TD3"]:
                    action = torch.tensor([0.0, 0.0, 0.0]).unsqueeze(0)
                    q_value1, q_value2 = model.critic(state_tensor, action)
                    Z[i, j] = torch.min(q_value1, q_value2).item()

        ax = axs[idx]
        ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10, color='black', linewidth=0.5)
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('Value Function')
        ax.set_title(f'Switch State {switch_state} - {algorithm}')

        max_z = Z.max()
        i = 1
        for loc, switch_state_val in zip(env._switch_locations, switch_state):
            color = 'green' if switch_state_val else 'black'
            ax.scatter(loc[0], loc[1], max_z, color=color, s=100, label=f'Switch {i} at {loc}')
            i += 1

        ax.legend()
        ax.set(xticklabels=[], yticklabels=[], zticklabels=[])

    plt.show()

else:
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    for idx, switch_state in enumerate(switch_states):
        Z = np.zeros(X.shape)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                state = np.array([X[i, j], Y[i, j], *switch_state])
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                
                if algorithm == "PPO":
                    value = model.policy.predict_values(state_tensor)
                    Z[i, j] = value.item()
                    
                if algorithm in ["SAC", "TD3"]:
                    action = torch.tensor([0.0, 0.0, 0.0]).unsqueeze(0)
                    q_value1, q_value2 = model.critic(state_tensor, action)
                    Z[i, j] = torch.min(q_value1, q_value2).item()

        ax = axs[idx]
        im = ax.imshow(Z, extent=(0, env._size, 0, env._size), origin='lower', cmap='viridis')
        
        i = 1
        for loc, switch_state_val in zip(env._switch_locations, switch_state):
            color = 'green' if switch_state_val else 'black' if i == 1 else 'darkgrey'
            ax.scatter(loc[0], loc[1], color=color, s=100, label=f'Switch {i} at {loc}')
            
            # Add a dotted circle around the switch location
            circle = Circle(loc, radius=0.2, color='black', fill=False, linestyle='dotted')
            ax.add_patch(circle)
            
            i += 1
        
        ax.legend()
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Value Function')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Switch State {switch_state} - {algorithm}')
        
    plt.show()
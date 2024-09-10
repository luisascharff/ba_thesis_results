import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import matplotlib.pyplot as plt
import seaborn as sns

class KeyDoorGoalContinuousEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, size=2.0, key_position=[0.2, 0.2], door_position=(1.6, 1.6), goal_position=(0.2, 1.5), render_mode=False):
        super(KeyDoorGoalContinuousEnv, self).__init__()
        
        self.size = size
        self.key_position = np.array(key_position)
        self.door_position = np.array(door_position)
        self.goal_position = np.array(goal_position)
        self.render_mode = render_mode

        
        self.action_space = spaces.Box(low=np.array([-1, -1, 0, 0]), high=np.array([1, 1, 1, 1]), dtype=np.float32)

        
        low = np.array([0.0, 0.0, 0, 0, 0])
        high = np.array([self.size, self.size, 1, 1, 1])
        self.observation_space = spaces.Box(low=low, high=high, dtype=float)

        self.state = None
        self.has_key = False
        self.door_opened = False
        self.end_goal_reached = False
        self._steps = 0

      
        self.frames = []
        self.clock = None
        self.screen = None
        self.isopen = True
        
      
        self.key_pickup_timer = 0
        self.door_opened_timer = 0
        self.color_change_duration = 20  

    def reset(self, seed=None, state=None):
        super().reset(seed=seed)

        self._steps = 0
        self.has_key = False
        self.door_opened = False
        self.end_goal_reached = False
        self.key_pickup_timer = 0
        self.door_opened_timer = 0

        if state is None:
            self.state = np.array([np.random.uniform(0, self.size), np.random.uniform(0, self.size), 0, 0, 0])
        else:
            assert len(state) == 5, "State must include position, key possession, door status, and goal status."
            self.state = np.array(state)

        if self.render_mode and self.screen is None:
            pygame.init()
            screen_size = 600
            self.screen = pygame.display.set_mode((screen_size, screen_size))
            self.clock = pygame.time.Clock()

        if self.render_mode:
            self.render()

        return self.state, {}


    def step(self, action):
        if len(action) != 4:
            raise ValueError("Action must be a 4-dimensional vector.")
        
        x, y, has_key, door_opened, goal_reached = self.state
        truncated = False
        reward = 0

        movement = action[:2] * 0.1
        pick_up_key = action[2]
        open_door = action[3]

        x = np.clip(x + movement[0], 0, self.size)
        y = np.clip(y + movement[1], 0, self.size)

        if not self.has_key and pick_up_key > 0.5 and np.linalg.norm([x - self.key_position[0], y - self.key_position[1]]) < 0.2:
            self.has_key = True
            has_key = 1
            reward += 2
            self.key_pickup_timer = self.color_change_duration

        if self.has_key and not self.door_opened and open_door > 0.5 and np.linalg.norm([x - self.door_position[0], y - self.door_position[1]]) < 0.2:
            self.door_opened = True
            door_opened = 1
            reward += 2
            self.door_opened_timer = self.color_change_duration

        if self.door_opened and not self.end_goal_reached and np.linalg.norm([x - self.goal_position[0], y - self.goal_position[1]]) < 0.2:
            self.end_goal_reached = True
            goal_reached = 1
            reward += 50

        new_state = np.array([x, y, has_key, door_opened, goal_reached])
        done = self.end_goal_reached

        self.state = new_state
        self._steps += 1
        
        reward -= 0.1
        
        if self._steps >= 1500:
            truncated = True
            
        if self.render_mode:
            self.render()

        return self.state, reward, done, truncated, {}

    def render(self, mode='human'):
        if not self.render_mode:
            return

        screen_size = 600 
        scale = screen_size / self.size

        if not self.isopen:
            return

        self.screen.fill((255, 255, 255))  

        
        agent_pos = (self.state[:2] * scale).astype(int)
        pygame.draw.circle(self.screen, (255, 0, 0), agent_pos, 10)

      
        key_color = (0, 200, 0) if self.key_pickup_timer == 0 else (255, 255, 0) 
        key_pos = (self.key_position * scale).astype(int)
        pygame.draw.circle(self.screen, key_color, key_pos, 10)

     
        door_color = (0, 0, 255) if self.door_opened_timer == 0 else (0, 255, 255)  
        door_pos = (self.door_position * scale).astype(int)
        pygame.draw.rect(self.screen, door_color, (*door_pos, 20, 20)) 

       
        goal_pos = (self.goal_position * scale).astype(int)
        pygame.draw.circle(self.screen, (255, 255, 0), goal_pos, 10)  

        if mode == 'human':
            pygame.display.flip()
            self.clock.tick(10) 

        
        if self.key_pickup_timer > 0:
            self.key_pickup_timer -= 1
        if self.door_opened_timer > 0:
            self.door_opened_timer -= 1

       
        frame = pygame.surfarray.array3d(self.screen)
        frame = np.transpose(frame, (1, 0, 2))
        self.frames.append(frame)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False
    
    def sample(env):
        """
        Function to sample a valid state from the KeyDoorGoalContinuousEnv environment.
        Valid states are limited to the following patterns:
        [x, y, 0, 0, 0], [x, y, 1, 0, 0], [x, y, 1, 1, 0], [x, y, 1, 1, 1].
        
        Args:
            env (KeyDoorGoalContinuousEnv): The environment instance from which to sample the state.
            
        Returns:
            np.array: A sampled valid state within the bounds of the environment's observation space.
        """
        
        x = np.random.uniform(0, env.size)
        y = np.random.uniform(0, env.size)
        
        
        valid_patterns = [
            [0, 0, 0],  
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1]
        ]
        
        
        selected_pattern = valid_patterns[np.random.choice(len(valid_patterns))]
        
       
        sampled_state = np.array([x, y] + selected_pattern)
        
        return sampled_state

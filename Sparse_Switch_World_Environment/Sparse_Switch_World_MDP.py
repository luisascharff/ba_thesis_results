import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import matplotlib.pyplot as plt

class SwitchWorldContinuousSA(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, size=1.5, num_switches=2, switch_width=0.3, switch_locations=[np.array([0.3, 0.3]), np.array([1.0, 1.0])], render_mode=False):
        super(SwitchWorldContinuousSA, self).__init__()

        self._size = size
        self._num_switches = num_switches
        self._switch_width = switch_width
        self._switch_locations = switch_locations
        self._switch_states = np.zeros(num_switches, dtype=float)

        # Pygame 
        self.frames = []
        self.clock = None
        self.screen = None
        self.isopen = True
        self.render_mode = render_mode

        
        self.action_space = spaces.Box(low=np.array([-1, -1, 0]), high=np.array([1, 1, 1]), dtype=float) 

        
        low = np.array([0, 0] + [0] * num_switches)
        high = np.array([size, size] + [1] * num_switches)
        self.observation_space = spaces.Box(low=low, high=high, dtype=float)

        self._state = None
        self._steps = 0
        self._episode_end = True

    def reset(self, seed=None, state=None):
        self._steps = 0
        self._switch_states = np.zeros(self._num_switches, dtype=int)

        if state is None:
            self._state = np.concatenate([np.random.rand(2) * self._size, self._switch_states])
        else:
            assert len(state) == 2 + self._num_switches, "State must include position and switch states."
            assert np.all(state[:2] >= 0) and np.all(state[:2] <= self._size), "Position must be within the defined space."
            assert np.all(np.isin(state[2:], [0, 1])), "Switch states must be either 0 or 1."
            self._state = state

        
        if self.render_mode and self.screen is None:
            pygame.init()
            screen_size = 600  
            self.screen = pygame.display.set_mode((screen_size, screen_size))
            self.clock = pygame.time.Clock()

        if self.render_mode:
            self.render()  

        return self._state, {}

    def step(self, action):
        
        if len(action) != 3:
            raise ValueError("Action must be a 3-dimensional vector.")
        
        pos = self._state[:2]
        switches = np.round(self._state[2:]).astype(int)
        truncated = False

        
        movement = action[:2] * 0.1  
        toggle = action[2]  

        pos += movement

        if toggle > 0.5:  
            for i, loc in enumerate(self._switch_locations):
                if np.linalg.norm(pos - loc) < self._switch_width and self._can_toggle(i, switches):
                    switches[i] = 1 - switches[i]  
                    

        self._state = np.concatenate((pos, switches))
        self._state[:2] = np.clip(self._state[:2], self.observation_space.low[:2], self.observation_space.high[:2]) 

        self._steps += 1
        reward = self._steps * -0.1
        terminated = False

        if all(switches == 1):
            reward += 50
            terminated = True

        if self._steps >= 1000:
            truncated = True

        if self.render_mode:
            self.render()

        return self._state, reward, terminated, truncated, {}

    def render(self, mode='human'):
        if not self.render_mode:
            return

        screen_size = 600  
        scale = screen_size / self._size

        if not self.isopen:
            return

        self.screen.fill((255, 255, 255))  

        
        agent_pos = (self._state[:2] * scale).astype(int)
        pygame.draw.circle(self.screen, (255, 0, 0), agent_pos, 10)

        
        for loc, switch in zip(self._switch_locations, self._state[2:]):
            color = (0, 255, 0) if switch else (0, 0, 255)  
            switch_pos = (loc * scale).astype(int)
            pygame.draw.rect(self.screen, color, (*switch_pos, int(self._switch_width * scale), int(self._switch_width * scale)))

        if mode == 'human':
            pygame.display.flip()
            self.clock.tick(10)  

        
        frame = pygame.surfarray.array3d(self.screen)
        frame = np.transpose(frame, (1, 0, 2))
        self.frames.append(frame)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def _can_toggle(self, switch_index, switches):
        if switches[switch_index] == 0:
            return all(switches[i] == 1 for i in range(switch_index))
        else:
            return all(switches[i] == 0 for i in range(switch_index + 1, self._num_switches))

    def _generate_valid_switch_state(self):
        switches = np.zeros(self._num_switches, dtype=int)
        for i in range(self._num_switches):
            if np.random.rand() < 0.5:
                switches[i] = 1
                for j in range(i):
                    switches[j] = 1
            else:
                break
        return switches
    
    def sample(self):
        pos = np.random.uniform(self.observation_space.low[:2], self.observation_space.high[:2])
        switches = self._generate_valid_switch_state()
        state = np.concatenate([pos, switches])
        return state

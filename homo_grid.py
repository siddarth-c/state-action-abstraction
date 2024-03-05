from __future__ import annotations

import os
import sys

repo_path = os.path.join(os.path.dirname(__file__), 'Minigrid')
sys.path.append(repo_path)

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava, Ball, Floor
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import Wrapper, ObservationWrapper
from minigrid.core.constants import COLOR_NAMES
from gymnasium import spaces
import numpy as np
import cv2


class CustomRGBImgObsWrapper(ObservationWrapper):

    def __init__(self, env, tile_size = 8):
        super().__init__(env)

        self.tile_size = tile_size

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.env.unwrapped.width * tile_size, self.env.unwrapped.height * tile_size, 1),
            dtype="float64",
        )

    def observation(self, obs):
        rgb_img = self.get_frame(highlight=False, tile_size=self.tile_size)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        
        return np.expand_dims(rgb_img, -1) / (255.0)
    
    
class HomoEnv(MiniGridEnv):
    def __init__(
        self,
        size=10,
        max_steps: int | None = None,
        stoch = True,
        seed = 0,
        **kwargs,
    ):
        
        self.rng = np.random.RandomState(seed)
        
        self.size = size
        self.stoch = stoch
        self.agent_start_pos = (1, 8)
        self.agent_start_dir = 0
        
        if self.stoch:
            # randint[low, high)
            self.agent_start_dir = self.rng.randint(0, 4)
            self.agent_start_pos = (self.rng.randint(1, 3), self.rng.randint(7, 9))
            

        mission_space = MissionSpace(mission_func = self._gen_mission)
        
        if max_steps is None:
            max_steps = 4 * (self.size**2)

        super().__init__(
            mission_space = mission_space,
            grid_size = size,
            see_through_walls = True,
            max_steps = max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.unwrapped.grid = Grid(width, height)

        # Generate the surrounding walls
        self.unwrapped.grid.wall_rect(0, 0, width, height)
        
        # Place a goal square in the top-right corner
        self.put_obj(Goal(), 8, 1)
        
        # (i, j) & (9 - j, 9 - i)
        
        self.lava_positions = [[1, 6], [3, 8], 
                                [3, 6], 
                                [2, 4], [5, 7], 
                                [3, 2], [7, 6], 
                                [5, 3], [6, 4], [5, 4], 
                                [6, 1], [8, 3]]
        
        
        for pos in self.lava_positions:
            self.put_obj(Lava(), pos[0], pos[1])
        
        # Place the agent
        if self.unwrapped.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
            
        self.mission = "Reach Green, Avoid Orange!"

    def reset(
        self,
        seed = None,
        options = None):
        super().reset(seed=seed)
        
        self.agent_start_pos = (1, 8)
        self.agent_start_dir = 0
        
        if self.stoch:
            # randint[low, high)
            self.agent_start_dir = self.rng.randint(0, 4)
            self.agent_start_pos = (self.rng.randint(1, 3), self.rng.randint(7, 9))
            
        # Generate a new random grid at the start of each episode
        self._gen_grid(self.width, self.height)
        
        # Item picked up, being carried, initially nothing
        self.carrying = None
        
        # Step count since episode start
        self.unwrapped.step_count = 0
        
        if self.render_mode == "human":
            self.render()
            
        # Return first observation
        obs = self.gen_obs()
            
        return obs, {}
    
class CustomRewardAndTransition(Wrapper):

    def __init__(self, env, prob = 1.0):
        super().__init__(env)
        
        self.env = env
        self.prob = prob

    def step(self, action):
        
        if self.rng.random_sample() > self.prob:
            action = self.env.action_space.sample()
            
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = -1
        
        x, y = self.env.unwrapped.agent_pos
        
        # reward = (-(y - 1) - (self.env.size - 2 - x)) / 10
        
        if [x, y] in self.env.unwrapped.lava_positions:
            terminated = False
            reward = -10

        if terminated:
            reward = +10
        
        return obs, reward, terminated, truncated, info


# def main():

#     env = HomoEnv(render_mode = "human", stoch = True, seed = 42)
#     env = CustomRGBImgObsWrapper(env)
#     env = CustomRewardAndTransition(env, 0.8)

#     # enable manual control for testing
#     manual_control = ManualControl(env, seed=42)
#     manual_control.start()
    

# if __name__ == "__main__":
#     main()

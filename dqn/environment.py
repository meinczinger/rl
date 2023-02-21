import gymnasium as gym
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics, TimeLimit
import numpy as np
from dqn.wrappers import MaxAndSkipEnv, FireResetEnv, ProcessFrame84, ImageToPyTorch, BufferWrapper, ScaledFloatFrame

class DQNEnvironment:
    def __init__(self, name: str, max_episode_step: int = 0, atari_game: bool = False):
        self.env = gym.make(name)
        if max_episode_step > 0:
            self.env = TimeLimit(self.env, max_episode_steps=max_episode_step)
        # # env = RecordVideo(env, video_folder='./videos', episode_trigger=lambda x: x % 50 == 0)
        
        if atari_game:
            self.env = gym.wrappers.AtariPreprocessing(self.env)
            self.env = gym.wrappers.TransformObservation(self.env, lambda x: x.swapaxes(-1, 0))
            self.env.observation_space = gym.spaces.Box(low=0, high=255, shape=(1, 84, 84), dtype=np.float32)
            # self.env = MaxAndSkipEnv(self.env)  
            # self.env = FireResetEnv(self.env)
            # self.env = ProcessFrame84(self.env)
            # self.env = ImageToPyTorch(self.env)
            # self.env = BufferWrapper(self.env, 4)
            # self.env = ScaledFloatFrame(self.env)        

        self.env = RecordEpisodeStatistics(self.env)

    def observation_size(self):
        return self.env.observation_space.shape[0]

    def number_of_actions(self):
        return self.env.action_space.n

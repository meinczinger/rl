import gymnasium as gym
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics, TimeLimit
import abc


class DQNEnvironment:
    def __init__(self, name: str, max_episode_step: int = 0, atari_game: bool = False):
        self.env = gym.make(name)
        if max_episode_step > 0:
            self.env = TimeLimit(self.env, max_episode_steps=max_episode_step)
        # # env = RecordVideo(env, video_folder='./videos', episode_trigger=lambda x: x % 50 == 0)
        self.env = RecordEpisodeStatistics(self.env)
        if atari_game:
            self.env = gym.wrappers.AtariPreprocessing(self.env)

    def observation_size(self):
        return self.env.observation_space.shape[0]

    def number_of_actions(self):
        return self.env.action_space.n

import gymnasium as gym
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics, TimeLimit



class DQNEnvironment:
    def __init__(self, name:str):
        self.env = gym.make(name)
        self.env = TimeLimit(self.env, max_episode_steps=400)
        # # env = RecordVideo(env, video_folder='./videos', episode_trigger=lambda x: x % 50 == 0)
        self.env = RecordEpisodeStatistics(self.env)

    def observation_size(self):
        return self.env.observation_space.shape[0]

    def number_of_actions(self):    
        return self.env.action_space.n
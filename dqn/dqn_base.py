import torch
import gymnasium as gym
import abc
import math
import os
from torch.utils.tensorboard import SummaryWriter
from dqn.neural_net import NeuralNetwork
from dqn.memory import ExperienceReplay

writer = SummaryWriter("runs/pong")

class ConcreteDQN:
    def __init__(
        self,
        env: gym.Env,
        neural_net: NeuralNetwork,
        lr: float,
        device,
        gamma: float,
        clip_error: bool = True,
        double_dqn: bool = True,
        target_net: NeuralNetwork = None,
        update_target_frequency: int = 1000,
        experience_memory_size: int = 1,
        batch_size: int = 32,
        egreedy: float = 0.9,
        egreedy_final: float = 0.01,
        egreedy_decay: float = 500,
        file2save: str = "pong_save.pth",
        save_model_frequency: int = 10000,
        resume_previous_training: bool = False,
        preprocess_state_func=None,
    ):
        self.env = env
        self.neural_net = neural_net
        self.lr = lr
        self.device = device
        self.gamma = gamma
        self.clip_error = clip_error
        self.double_dqn = double_dqn
        self.egreedy = egreedy
        self.egreedy_final = egreedy_final
        self.egreedy_decay = egreedy_decay
        self.file2save = file2save
        self.save_model_frequency = save_model_frequency
        self.preprocess_state_func = preprocess_state_func


        if target_net is None:
            self.target_net = neural_net            
            self.use_target_net = False
        else:
            self.target_net = target_net
            self.update_target_frequency = update_target_frequency
            self.use_target_net = True

        if resume_previous_training and os.path.exists(self.file2save):
            self.neural_net.load_state_dict(self.load_model())
            self.target_net.load_state_dict(self.load_model())

        self.experience_memory_size = experience_memory_size
        self.experience_replay = ExperienceReplay(self.experience_memory_size)
        self.batch_size = batch_size

        self.loss_func = torch.nn.MSELoss()

        self.optimizer = torch.optim.Adam(
            params=self.neural_net.parameters(), lr=self.lr
        )


        self.number_of_steps = 0

    def preprocess_state(self, state):
        if self.preprocess_state_func is not None:
            return self.preprocess_state_func(state, self.device)
        else:
            return torch.Tensor(state).to(self.device)

    def select_action(self, state, epsilon):
        random_for_egreedy = torch.rand(1)[0]

        if random_for_egreedy > epsilon:
            with torch.no_grad():
                state = self.preprocess_state(state)
                state = torch.cat([state])
                action_from_nn = self.neural_net(state)
                action = torch.max(action_from_nn, 1)[1]
                action = action.item()
        else:
            action = self.env.action_space.sample()

        return action


    def optimize(self):
        if len(self.experience_replay) < self.batch_size:
            return

        state, action, new_state, reward, done = self.experience_replay.sample(self.batch_size)

        state = [self.preprocess_state(frame) for frame in state]
        state = torch.cat(state)

        new_state = [self.preprocess_state(frame) for frame in new_state]
        new_state = torch.cat(new_state)

        reward = torch.Tensor(reward).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        done = torch.Tensor(done).to(self.device)
        
        if self.double_dqn:
            new_state_values = self.neural_net(new_state).detach()
            max_new_state_indexes = torch.max(new_state_values, 1)[1]  
            
            new_state_values = self.target_net(new_state).detach()
            max_new_state_values = new_state_values.gather(1, max_new_state_indexes.unsqueeze(1)).squeeze(1)            
        else:
            new_state_values = self.target_net(new_state).detach()
            max_new_state_values = torch.max(new_state_values)
            
        target_value = reward + (1 - done) * self.gamma * max_new_state_values

        predicted_value = self.neural_net(state).gather(1, action.unsqueeze(1)).squeeze(1)

        loss = self.loss_func(predicted_value, target_value)

        self.optimizer.zero_grad()

        loss.backward()

        if self.clip_error:
            for param in self.neural_net.parameters():
                param.grad.data.clamp_(-1,1)

        self.optimizer.step()

    def calculate_epsilon(self, steps_done):
        epsilon = self.egreedy_final + (self.egreedy - self.egreedy_final) * math.exp(
            -1.0 * steps_done / self.egreedy_decay
        )
        return epsilon


    def after_optimize(self):
        if (self.number_of_steps > 0) and (
            self.number_of_steps % self.save_model_frequency == 0
        ):
            self.save_model(self.neural_net)
        if self.use_target_net and ((self.number_of_steps % self.update_target_frequency) == 0):
            self.target_net.load_state_dict(self.neural_net.state_dict())


    def reset_env(self, seed_value):
        if seed_value == 0:
            state, _ = self.env.reset()
        else:
            state, _ = self.env.reset(seed=seed_value)
        return state

    def load_model(self):
        print("Loading model")
        return torch.load(self.file2save)

    def save_model(self, model):
        print("Saving model")
        torch.save(model.state_dict(), self.file2save)

    def learn(
        self,
        nr_of_episodes: int,
        seed_value: int = 0,
        report_interval: int = 100,
        metrics_callback=None,
    ):
        rewards_total = []

        for i_episode in range(nr_of_episodes):
            state = self.reset_env(seed_value)
            # print("--- State ---")
            # print(state.shape)
            # print(type(state))
            # print(state.dtype)

            score = 0
            done = False
            while not done:
                epsilon = self.calculate_epsilon(self.number_of_steps)

                action = self.select_action(state, epsilon)

                new_state, reward, done, truncated, _ = self.env.step(action)

                done = done or truncated

                score += reward

                self.experience_replay.push(state, action, new_state, reward, 1 if done else 0)

                self.optimize()

                self.number_of_steps += 1

                self.after_optimize()

                state = new_state

                if done:
                    print(i_episode, score, epsilon)
                    rewards_total.append(score)

                    if (i_episode % report_interval == 0) and (
                        metrics_callback is not None
                    ):
                        metrics_callback(rewards_total)
                    
                    writer.add_scalar('reward', score, i_episode)
                    writer.add_scalar('epsilon', epsilon, i_episode)
                    

        return rewards_total


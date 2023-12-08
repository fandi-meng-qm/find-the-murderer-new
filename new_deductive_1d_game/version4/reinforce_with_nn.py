import itertools as it

import torch
from matplotlib import pyplot as plt
import numpy as np
import pickle
from game_core import MurderGame, MurderParams
from open_spiel.python.algorithms import exploitability
from open_spiel.python import policy as policy_lib
import gym
import torch
from open_spiel.python import rl_environment
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 device):
        self.policy_net = PolicyNet(state_dim, hidden_dim,
                                    action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=learning_rate)  # Using the Adam optimizer
        self.gamma = gamma  # discount factor
        self.device = device

    def take_action(self, state):  # Random sampling based on action probability distribution
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):  # Starting from the last step
            reward = reward_list[i]
            state = torch.tensor([state_list[i]],
                                 dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = self.gamma * G + reward
            loss = -log_prob * G
            loss.backward()
        self.optimizer.step()


learning_rate = 1e-3
num_episodes = 10000
hidden_dim = 128
gamma = 1
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

params = MurderParams(4, 1, 1)
game = MurderGame(game_params=params)

env = rl_environment.Environment(game, include_full_state=True)
state_dim = env._game.information_state_tensor_size()
action_dim = env.action_spec()["num_actions"]
agent = REINFORCE(state_dim, hidden_dim, action_dim, learning_rate, gamma,
                      device)

def test(agent) -> float:
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': []
                }
                time_step = env.reset()

                while not time_step.last():
                    action = agent.take_action(time_step.observations['info_state'][0])
                    transition_dict['states'].append(time_step.observations['info_state'][0])
                    time_step = env.step([action])
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(time_step.observations['info_state'][0])
                    transition_dict['rewards'].append(time_step.rewards[0])
                    transition_dict['dones'].append(time_step.last())
                    episode_return += time_step.rewards[0]
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 100 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-100:])
                    })
                pbar.update(1)
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('REINFORCE on 1-D Simple Game')
    plt.show()


if __name__ == "__main__":
    start_time = time.time()
    test(agent)
    end_time = time.time()
    print("time cost: {:.2f}s".format(end_time - start_time))



import argparse
import numpy as np
from itertools import count
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from envs.GridWorldEnv import GridWorldEnv
import time

from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
# Constants representing the state of each cell
S = 0
I = 1
R = 2
V = 3

# Assuming the GridWorldEnv class definition is already provided above this code

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--reward-threshold', type=float, default=195.0, metavar='RT',
                    help='reward threshold for solving the environment (default: 195.0)')

parser.add_argument('--num_episodes', type=int, default=75000)
args = parser.parse_args()

env = GridWorldEnv(render_mode='human' if args.render else None)
env.reset(seed=args.seed)
torch.manual_seed(args.seed)

class Policy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(input_dim, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, output_dim)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

input_dim = np.prod(env.observation_space.shape)
output_dim = env.action_space.n
policy = Policy(input_dim, output_dim)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()

def select_action(state):
    state = torch.from_numpy(state.flatten()).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()

def finish_episode():
    R = 0
    policy_loss = []
    returns = deque()
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.appendleft(R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]

def plot_results(rewards, infected, vaccinated, susceptible):
    episodes = range(1, len(rewards) + 1)
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(episodes, rewards, label='Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward over Episodes')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(episodes, infected, label='Infected')
    plt.plot(episodes, vaccinated, label='Vaccinated')
    plt.plot(episodes, susceptible, label='Susceptible')
    plt.xlabel('Episode')
    plt.ylabel('Proportion of Population')
    plt.title('Population States over Episodes')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    global_step=0
    import wandb
    run_name = f"a__a__a__{int(time.time())}"

    wandb.init(
        project="rl-vaccination", entity="rl-project-lma",
        sync_tensorboard=True,
        config=vars(args),
        name=run_name,
        monitor_gym=True,
        save_code=True
    )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    running_reward = 10
    rewards_over_episodes = []
    infected_over_episodes = []
    vaccinated_over_episodes = []
    susceptible_over_episodes = []

    for i_episode in tqdm(range(args.num_episodes), total=args.num_episodes):
        state, _ = env.reset()
        ep_reward = 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            global_step += 1
            done = terminated or truncated
            if args.render:
                env.render()
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()
        rewards_over_episodes.append(ep_reward)
        infected_over_episodes.append(info['state_counts'][I])
        vaccinated_over_episodes.append(info['state_counts'][V])
        susceptible_over_episodes.append(info['state_counts'][S])
    
                    
        if 'state_counts' in info:
            state_counts = info['state_counts']
            writer.add_scalar("counts/Susceptible", state_counts[S], global_step)
            writer.add_scalar("counts/Infected", state_counts[I], global_step)
            writer.add_scalar("counts/Recovered", state_counts[R], global_step)
            writer.add_scalar("counts/Vaccinated", state_counts[V], global_step)
            writer.add_scalar("counts/Available_Vaccines", info['num_vaccine'], global_step)

        if i_episode % 100000 == 99999:
            pass
                #log_final_state(info, global_step)
        
        
        
    # Plotting the results
    #plot_results(rewards_over_episodes, infected_over_episodes, vaccinated_over_episodes, susceptible_over_episodes)

if __name__ == '__main__':
    main()


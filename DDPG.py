import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from uav_2d import EnvWrapper
import torch as T
import draw as d
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math as m
import ssa as s


TAU = 0.05
GAMMA = 0.95
DELAY = 2
STD = 0.1
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
BATCH_SIZE = 128
START_UPD_SAMPLES = 2000
TARGET_STD = 0.1
PI=3.14159265358979323846

class Point2D:
    def __init__(self):
        self.x = complex()
        self.y = complex()

lost_uav = Point2D()
lost_uav.x = 10.0
lost_uav.y = 45  # 失联无人机的位置

neigh_uav = {}

def sel(uavx,uavy,num):
    k = 0
    for i in range(num):
        if ((uavx[i]-lost_uav.x) * (uavx[i]-lost_uav.x) + (uavy[i]-lost_uav.y) * (uavy[i]-lost_uav.y)) < 625:
            neigh_uav[i] = Point2D()
            neigh_uav[k].x = uavx[i]
            neigh_uav[k].y = uavy[i]
            k = k + 1
    return neigh_uav

def l2norm (x,y):
    return m.sqrt(x * x + y * y)

def real_done(done):
    for v in done.values():
        if not v:
            return False
    return True

def includedangle(x1,y1,x2,y2):#目标点 无人机
    w = 0
    if(x1 >= x2 and y1 >= y2):
        y = l2norm(x1 - x2, y1 - y2)
        cosx = (x1 - x2 ) / y
        w = m.acos(cosx)
    if(x1 > x2 and y1 < y2):
        y = l2norm(x1 - x2, y1 - y2)
        cosx = (x1 - x2) / y
        w = m.acos(cosx)
        w = 2*PI-w
    if(x1 < x2 and y1 >= y2):
        y = l2norm(x1 - x2, y1 - y2)
        cosx = (x1 - x2) / y
        w = m.acos(cosx)
        w = PI + w
    if (x1 < x2 and y1 <= y2):
        y = l2norm(x1 - x2, y1 - y2)
        cosx = (x1 - x2) / y
        w = m.acos(cosx)
        w = PI  - w
    if w > PI:
        w = w -2*PI
    return w

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Input n_agents and main folder')
parser.add_argument('--agents', type=int, default=5)
parser.add_argument('--folder', type=str)
parser.add_argument('--global_', type=str, default="GLOBAL")
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/TD3/')

args = parser.parse_args()

N_AGENTS = args.agents
MAIN_FOLDER = args.folder


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


class ActorNetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.ln1 = nn.LayerNorm(fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)
        self.action = nn.Linear(fc2_dim, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.apply(weight_init)
        self.to(device)

    def forward(self, state):
        x = T.relu(self.ln1(self.fc1(state)))
        x = T.relu(self.ln2(self.fc2(x)))
        action = T.tanh(self.action(x))

        return action

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, beta, state_dim, action_dim, fc1_dim, fc2_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.ln1 = nn.LayerNorm(fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)
        self.fc3 = nn.Linear(action_dim, fc2_dim)
        self.q = nn.Linear(fc2_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta, weight_decay=0.001)
        self.apply(weight_init)
        self.to(device)

    def forward(self, state, action):
        x_s = T.relu(self.ln1(self.fc1(state)))
        x_s = self.ln2(self.fc2(x_s))
        x_a = self.fc3(action)
        x = T.relu(x_s + x_a)
        q = self.q(x)

        return q

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file))


class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim, batch_size):
        self.mem_size = max_size
        self.batch_size = batch_size
        self.mem_cnt = 0

        self.state_memory = np.zeros((self.mem_size, state_dim))
        self.action_memory = np.zeros((self.mem_size, action_dim))
        self.reward_memory = np.zeros((self.mem_size,))
        self.next_state_memory = np.zeros((self.mem_size, state_dim))
        self.terminal_memory = np.zeros((self.mem_size,), dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        mem_idx = self.mem_cnt % self.mem_size

        self.state_memory[mem_idx] = state
        self.action_memory[mem_idx] = action
        self.reward_memory[mem_idx] = reward
        self.next_state_memory[mem_idx] = state_
        self.terminal_memory[mem_idx] = done

        self.mem_cnt += 1

    def sample_buffer(self):
        mem_len = min(self.mem_size, self.mem_cnt)
        batch = np.random.choice(mem_len, self.batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.next_state_memory[batch]
        terminals = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminals

    def n_samples(self):
        return self.mem_size if not self.ready else self.mem_cnt

    def ready(self):
        return self.mem_cnt >= self.batch_size

    def ready(self):
        return self.mem_cnt >= self.batch_size





class DDPG:
    def __init__(self, alpha, beta, state_dim, action_dim, actor_fc1_dim,
                 actor_fc2_dim, critic_fc1_dim, critic_fc2_dim, ckpt_dir,
                 gamma=0.99, tau=0.005, action_noise=0.05, max_size=1000000,
                 batch_size=256):
        self.gamma = gamma
        self.tau = tau
        self.action_noise = action_noise
        self.checkpoint_dir = ckpt_dir

        self.actor = ActorNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                  fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim)
        self.target_actor = ActorNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                         fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim)
        self.critic = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
                                    fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)
        self.target_critic = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
                                           fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)

        self.memory = ReplayBuffer(max_size=max_size, state_dim=state_dim, action_dim=action_dim,
                                   batch_size=batch_size)

        self.update_network_parameters(tau=1.0)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for actor_params, target_actor_params in zip(self.actor.parameters(),
                                                     self.target_actor.parameters()):
            target_actor_params.data.copy_(tau * actor_params + (1 - tau) * target_actor_params)

        for critic_params, target_critic_params in zip(self.critic.parameters(),
                                                       self.target_critic.parameters()):
            target_critic_params.data.copy_(tau * critic_params + (1 - tau) * target_critic_params)

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def choose_action(self, observation, train=True):
        self.actor.eval()
        state = T.tensor([observation], dtype=T.float).to(device)
        action = self.actor.forward(state).squeeze()

        if train:
            noise = T.tensor(np.random.normal(loc=0.0, scale=self.action_noise),
                             dtype=T.float).to(device)
            action = T.clamp(action + noise, -1, 1)
        self.actor.train()

        return action.detach().cpu().numpy()

    def learn(self):
        if not self.memory.ready():
            return

        states, actions, reward, states_, terminals = self.memory.sample_buffer()
        states_tensor = T.tensor(states, dtype=T.float).to(device)
        actions_tensor = T.tensor(actions, dtype=T.float).to(device)
        rewards_tensor = T.tensor(reward, dtype=T.float).to(device)
        next_states_tensor = T.tensor(states_, dtype=T.float).to(device)
        terminals_tensor = T.tensor(terminals).to(device)

        with T.no_grad():
            next_actions_tensor = self.target_actor.forward(next_states_tensor)
            q_ = self.target_critic.forward(next_states_tensor, next_actions_tensor).view(-1)
            q_[terminals_tensor] = 0.0
            target = rewards_tensor + self.gamma * q_
        q = self.critic.forward(states_tensor, actions_tensor).view(-1)

        critic_loss = F.mse_loss(q, target.detach())
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        new_actions_tensor = self.actor.forward(states_tensor)
        actor_loss = -T.mean(self.critic(states_tensor, new_actions_tensor))
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()


class TD3Trainer:

    def __init__(self, n_agents):
        self._n_agents = n_agents
        self._obs_dim = 48
        self._action_dim = 2
        self._agent =  DDPG(alpha=0.0003, beta=0.0003, state_dim=48,
                 action_dim=2, actor_fc1_dim=128, actor_fc2_dim=64,
                 critic_fc1_dim=128, critic_fc2_dim=64, ckpt_dir=args.checkpoint_dir)
        self._env = EnvWrapper(n_agents)
        self._now_ep = 0
        self._sw = SummaryWriter(f'./{MAIN_FOLDER}/logs/trainer')
        self._step = 0
        self.sumway = [0.0] *  n_agents
        self.endstep = [0.0] *  n_agents
        self.rew = [0.0] *  n_agents
        self.zhui_num = 0
        self.dao_num = 0


    def _sample_global(self):
        self._env.set_global_center(True)

    def train_one_episode(self):

        if args.global_ == 'GLOBAL':
            self._env.set_global_center(True)
        elif args.global_ == 'LOCAL':
            self._env.set_global_center(False)
        elif args.global_ == 'ANNEAL':
            self._sample_global()
        else:
            assert False

        self._now_ep += 1


        enum_seq = [f'uav_{i}' for i in range(self._n_agents)]  # 里面是字符串

        states = self._env.reset()  # 返回初始化的场景
        done = {n: False for n in enum_seq}
        total_rew = {n: 0 for n in enum_seq}

        while not real_done(done):
            actions = {}
            in_states = []
            for seq in enum_seq:
                in_states.append(states[seq])
            uavs = self._env._env._cpp_env.getUavs()
            w = self._env._env._cpp_env.getuavW()
            # if k>0:
            #     n = len(uavs)
            #     uavx = [0.0] * n
            #     uavy = [0.0] * n
            #     #print(uavs)
            #     for i in range(n):
            #         uavx[i] = uavs[i][0]
            #         uavy[i] = uavs[i][1]
            #     neigh_uav = sel(uavx, uavy, n)
            #     #print(neigh_uav)
            #     next = s.Tent_SSA(100, 2, lb, ub, 50, neigh_uav, lost_uav)
            #     #print( next )
            #     lost_uav.x = next[0][0]
            #     lost_uav.y = next[0][1]
                #print(lost_uav)
            v = self._env._env._cpp_env.getuavV()
            out_actions = self._agent.choose_action(in_states,True)
            for i, seq in enumerate(enum_seq):
                actions[seq] = out_actions[i]
            die = self._env._env._cpp_env.getStatus()
            choices = []
            for i in range(self._n_agents):
                if not die[i]:
                    choices.append(i)

            next_states, rewards, done, info = self._env.step(actions)

            self._step += 1
            # if self._step % 2 == 0:
            #     for u in uavs:
            #       d.draw_uav(uavs[u][0], uavs[u][1],'red')
            buffer_index = np.random.choice(choices)
            buffer_name = enum_seq[buffer_index]
            self._agent.memory.store_transition(states[buffer_name], actions[buffer_name], rewards[buffer_name],
                                    next_states[buffer_name],done)
            if self._step % 50 == 0 and self._agent.memory.n_samples() > START_UPD_SAMPLES:
                for _ in range(20):
                    self._agent.learn()
            for seq in enum_seq:
                total_rew[seq] += rewards[seq]
            states = next_states
            if self._now_ep % 200 == 0:
                self._sw.add_scalar(f'train_rew/0', total_rew['uav_0'], self._now_ep)
            # k = k + 1
        return total_rew
    def test_one_episode(self):
        self._env.set_global_center(True)
        enum_seq = [f'uav_{i}' for i in range(self._n_agents)]
        avg_rewards = 0.0

        self.sumway= [0.0]*self._n_agents
        self.endstep = [0.0]*self._n_agents
        self.rew = [0.0] * self._n_agents

        lost_uav.x = 10.0
        lost_uav.y = 45  # 失联无人机的位置
        k = 0
        lb = [-10, -10]
        ub = [10, 10]
        lost_v = 6
        lost_w = PI/4
        a = 0
        b = 0
        states = self._env.reset()
        done = {n: False for n in enum_seq}
        total_rew = {n: 0 for n in enum_seq}
        d.drawback()
        target_pos = self._env._env._cpp_env.getTarget()
        d.drawtarget(target_pos.x, target_pos.y)
        obs = self._env._env._cpp_env.getObstacles()
        for i in range(4):
            d.drawobs(obs[i].x, obs[i].y,i)
        while not real_done(done):
            actions = {}
            in_states = []
            for seq in enum_seq:
                in_states.append(states[seq])

            way = self._env._env._cpp_env.getWay()
            for i in range (self._n_agents):
                self.sumway[i] = way[i]+self.sumway[i]
            uavs = self._env._env._cpp_env.getUavs()
            w = self._env._env._cpp_env.getuavW()
            v = self._env._env._cpp_env.getuavV()
            if k>0:
                n = len(uavs)
                uavx = [0.0] * n
                uavy = [0.0] * n
                #print(uavs)
                for i in range(n):
                    uavx[i] = uavs[i][0]
                    uavy[i] = uavs[i][1]
                neigh_uav = sel(uavx, uavy, n)
                next = s.Tent_SSA(100, 2, lb, ub, 50, neigh_uav, lost_uav,obs)
                lost_uav.x = next[0][0]
                lost_uav.y = next[0][1]
                d.drawnext(lost_uav.x,lost_uav.y)
                collisiono = False
                for k in range(4):
                    if (l2norm(lost_uav.x - obs[k].x,lost_uav.y - obs[k].y) < 30 + k * 5):
                        collisiono = True
                        break
                if (collisiono and a==0):
                    self.zhui_num = self.zhui_num + 1
                    a = a + 1
                if (l2norm(lost_uav.x - target_pos.x, lost_uav.y - target_pos.y) < 20 and b == 0):
                    self.dao_num = self.dao_num+1
                    b= b + 1

            out_actions = self._agent.choose_action(in_states,False)
            for i, seq in enumerate(enum_seq):
                actions[seq] = out_actions[i]

            next_states, rewards, done, info = self._env.step(actions)

            self._step += 1
            #if self._step % 2 == 0:
            for u in uavs:
                d.draw_uav(uavs[u][0], uavs[u][1],'red')
            for seq in enum_seq:
                total_rew[seq] += rewards[seq]/100

            states = next_states
            #k = k + 1
        self.rew = total_rew
        self.End= self._env._env._cpp_env.getEnd()
        for i, seq in enumerate(enum_seq):
            avg_rewards = avg_rewards + total_rew[seq]
        avg_rewards = avg_rewards / self._n_agents
        for i, seq in enumerate(enum_seq):
            self._sw.add_scalar(f'test_rew/{i}', total_rew[seq], self._now_ep)
        self._sw.add_scalar(f'test_avg_rew', avg_rewards, self._now_ep)
        d.savepic(self._now_ep)
        print(avg_rewards,self.zhui_num,self.dao_num)
        return total_rew

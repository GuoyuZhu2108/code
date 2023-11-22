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
past_uav = Point2D()
lost_uav.x = 10.0
lost_uav.y = 45  # 失联无人机的位置

neigh_uav = {}

def sel(uavx,uavy,num):
    k = 0
    for i in range(num):
        if ((uavx[i]-lost_uav.x) * (uavx[i]-lost_uav.x) + (uavy[i]-lost_uav.y) * (uavy[i]-lost_uav.y)) < 625:
            neigh_uav[k] = Point2D()
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
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/TD3/')

args = parser.parse_args()

N_AGENTS = args.agents
MAIN_FOLDER = args.folder


class ActorNetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.ln1 = nn.LayerNorm(fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)
        self.action = nn.Linear(fc2_dim, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(device)

    def forward(self, state):
        x = T.relu(self.ln1(self.fc1(state)))
        x = T.relu(self.ln2(self.fc2(x)))
        action = T.tanh(self.action(x))
        return action

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, beta, state_dim, action_dim, fc1_dim, fc2_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, fc1_dim)
        self.ln1 = nn.LayerNorm(fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)
        self.q = nn.Linear(fc2_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.to(device)

    def forward(self, state, action):
        x = T.cat([state, action], dim=-1)
        x = T.relu(self.ln1(self.fc1(x)))
        x = T.relu(self.ln2(self.fc2(x)))
        q = self.q(x)

        return q

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file))


class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim, batch_size):
        self.mem_size = max_size
        self.batch_size = batch_size
        self.mem_cnt = 0

        self.state_memory = np.zeros((max_size, state_dim))
        self.action_memory = np.zeros((max_size, action_dim))
        self.reward_memory = np.zeros((max_size,))
        self.next_state_memory = np.zeros((max_size, state_dim))
        self.terminal_memory = np.zeros((max_size,), dtype=np.bool)


    def store_transition(self, state, action, reward, state_, done):
        mem_idx = self.mem_cnt % self.mem_size

        self.state_memory[mem_idx] = state
        self.action_memory[mem_idx] = action
        self.reward_memory[mem_idx] = reward
        self.next_state_memory[mem_idx] = state_
        self.terminal_memory[mem_idx] = done

        self.mem_cnt += 1

    def sample_buffer(self):
        mem_len = min(self.mem_cnt, self.mem_size)
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


class TD3:
    def __init__(self, alpha, beta, state_dim, action_dim, actor_fc1_dim, actor_fc2_dim,
                 critic_fc1_dim, critic_fc2_dim, ckpt_dir, gamma=0.99, tau=0.005, action_noise=0.05,
                 policy_noise=0.1, policy_noise_clip=0.5, delay_time=2, max_size=1000000,
                 batch_size=128):
        self.gamma = gamma
        self.tau = tau
        self.action_noise = action_noise
        self.policy_noise = policy_noise
        self.policy_noise_clip = policy_noise_clip
        self.delay_time = delay_time
        self.update_time = 0
        self.checkpoint_dir = ckpt_dir

        self.actor = ActorNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                  fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim)
        self.critic1 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
                                     fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)
        self.critic2 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
                                     fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)

        self.target_actor = ActorNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                         fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim)
        self.target_critic1 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
                                            fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)
        self.target_critic2 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
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

        for critic1_params, target_critic1_params in zip(self.critic1.parameters(),
                                                         self.target_critic1.parameters()):
            target_critic1_params.data.copy_(tau * critic1_params + (1 - tau) * target_critic1_params)

        for critic2_params, target_critic2_params in zip(self.critic2.parameters(),
                                                         self.target_critic2.parameters()):
            target_critic2_params.data.copy_(tau * critic2_params + (1 - tau) * target_critic2_params)

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def change(self,n):
        k = int((n-2000)/1000)
        self.action_noise = self.action_noise-k*0.01
        self.policy_noise = 2*self.action_noise
        if(self.action_noise < 0.05):
            self.action_noise = 0.05
            self.policy_noise = 2 * self.action_noise



    def choose_action(self, observation, uavs,obs,esp,tar,uavw,uavv,n,train):#角速度 飞行速度
        self.actor.eval()
        state = T.tensor([observation], dtype=T.float).to(device)
        r = random.uniform(0,1)
        if(r < 0.3 and esp<2000 and n==len(uavs)):
            avgv = 0
            wt = [0.0] * n
            wo = [0.0]* n
            lo = [0.0]* n
            action = [0.0] * n
            turn = False
            a = [0.0]*n
            b = [0.0]*n
            c = [0.0]*n
            for i in range(len(uavs)):
                wt[i] = includedangle(tar.x, tar.y, uavs[i][0], uavs[i][1])
                if(uavw[i] > wt[i]):
                    a[i]=(wt[i]-uavw[i] )/(PI)#负数

                else:
                    a[i] = (wt[i]-uavw[i] )/(PI)#正数

            for i in range(len(uavs)):
                lo[i] = [0.0] * 4
                for j in range (4):
                    lo[i][j] = l2norm(obs[j].x-uavs[i][0], obs[j].x-uavs[i][1])
                    if(lo[i][j] < 50):
                        turn = True
            #print(lo)
            if (turn):
                for i in range(len(uavs)):
                    wo[i] = [0.0]*4
                    for j in range (4):
                        wo[i][j] = includedangle(obs[j].x, obs[j].x, uavs[i][0], uavs[i][1])
                        if (uavw[i] > wo[i][j]):
                            k =  (uavw[i] -wo[i][j])
                            b[i] =  (PI-k)/(PI)
                        else:
                            k = ((wo[i][j]) - uavw[i] )#正数
                            b[i] =  - (PI-k)/(PI)#越接近越小
            avguavw = 0.0
            for i in range(len(uavs)):
                avgv = avgv + uavv[i]
                avguavw = avguavw +uavw[i]
            avguavw = avguavw/len(uavs)
            avgv = avgv/len(uavs)
            d = [0.0] * len(uavs)
            for i in range(len(uavs)):
                if (uavw[i] >avguavw):
                    c[i] =  (avguavw - uavw[i]) / (PI)
                else:
                    c[i] =  ( avguavw -uavw[i]) / (PI)
            for i in range(len(uavs)):
                if (uavv[i] >avgv):
                    d[i] =  (avgv - uavv[i])
                else:
                    d[i] =  (avgv - uavv[i])#负数
            sum = [0.0]*n
            #print(d)

            for i in range(len(uavs)):
                if(d[i]>1):
                    d[i] = 1
                if(d[i]<-1):
                    d[i] = -1

            for i in range(len(uavs)):
                sum[i] = a[i]+b[i]+c[i]
                if sum[i] >1:
                    sum[i] = 1
                if sum[i] < -1:
                    sum[i] = -1
                #print(sum[i])

            for i in range(len(uavs)):
                action[i] = [sum[i],d[i]]
            #print(action)
            action = T.tensor(action)
            #print(action)

        else:
            action = self.actor.forward(state)
            if train:
                # exploration noise
                noise = T.tensor(np.random.normal(loc=0.0, scale=self.action_noise),
                                 dtype=T.float).to(device)
                action = T.clamp(action + noise, -1, 1)
            self.actor.train()
        #print(action)
        return action.squeeze().detach().cpu().numpy()

    def learn(self):
        if not self.memory.ready():
            return

        states, actions, rewards, states_, terminals = self.memory.sample_buffer()
        states_tensor = T.tensor(states, dtype=T.float).to(device)
        actions_tensor = T.tensor(actions, dtype=T.float).to(device)
        rewards_tensor = T.tensor(rewards, dtype=T.float).to(device)
        next_states_tensor = T.tensor(states_, dtype=T.float).to(device)
        terminals_tensor = T.tensor(terminals).to(device)

        with T.no_grad():
            next_actions_tensor = self.target_actor.forward(next_states_tensor)
            action_noise = T.tensor(np.random.normal(loc=0.0, scale=self.policy_noise),
                                    dtype=T.float).to(device)
            # smooth noise
            action_noise = T.clamp(action_noise, -self.policy_noise_clip, self.policy_noise_clip)
            next_actions_tensor = T.clamp(next_actions_tensor + action_noise, -1, 1)
            q1_ = self.target_critic1.forward(next_states_tensor, next_actions_tensor).view(-1)
            q2_ = self.target_critic2.forward(next_states_tensor, next_actions_tensor).view(-1)
            q1_[terminals_tensor] = 0.0
            q2_[terminals_tensor] = 0.0
            critic_val = T.min(q1_, q2_)
            target = rewards_tensor + self.gamma * critic_val
        q1 = self.critic1.forward(states_tensor, actions_tensor).view(-1)
        q2 = self.critic2.forward(states_tensor, actions_tensor).view(-1)

        critic1_loss = F.mse_loss(q1, target.detach())
        critic2_loss = F.mse_loss(q2, target.detach())
        critic_loss = critic1_loss + critic2_loss
        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        self.update_time += 1
        if self.update_time % self.delay_time != 0:
            return

        new_actions_tensor = self.actor.forward(states_tensor)
        q1 = self.critic1.forward(states_tensor, new_actions_tensor)
        actor_loss = -T.mean(q1)
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def save_models(self, episode):
        self.actor.save_checkpoint(self.checkpoint_dir + 'Actor/TD3_actor_{}.pth'.format(episode))
        print('Saving actor network successfully!')
        self.target_actor.save_checkpoint(self.checkpoint_dir +
                                          'Target_actor/TD3_target_actor_{}.pth'.format(episode))
        print('Saving target_actor network successfully!')
        self.critic1.save_checkpoint(self.checkpoint_dir + 'Critic1/TD3_critic1_{}.pth'.format(episode))
        print('Saving critic1 network successfully!')
        self.target_critic1.save_checkpoint(self.checkpoint_dir +
                                            'Target_critic1/TD3_target_critic1_{}.pth'.format(episode))
        print('Saving target critic1 network successfully!')
        self.critic2.save_checkpoint(self.checkpoint_dir + 'Critic2/TD3_critic2_{}.pth'.format(episode))
        print('Saving critic2 network successfully!')
        self.target_critic2.save_checkpoint(self.checkpoint_dir +
                                            'Target_critic2/TD3_target_critic2_{}.pth'.format(episode))
        print('Saving target critic2 network successfully!')

    def load_models(self, episode):
        self.actor.load_checkpoint(self.checkpoint_dir + 'Actor/TD3_actor_{}.pth'.format(episode))
        print('Loading actor network successfully!')
        self.target_actor.load_checkpoint(self.checkpoint_dir +
                                          'Target_actor/TD3_target_actor_{}.pth'.format(episode))
        print('Loading target_actor network successfully!')
        self.critic1.load_checkpoint(self.checkpoint_dir + 'Critic1/TD3_critic1_{}.pth'.format(episode))
        print('Loading critic1 network successfully!')
        self.target_critic1.load_checkpoint(self.checkpoint_dir +
                                            'Target_critic1/TD3_target_critic1_{}.pth'.format(episode))
        print('Loading target critic1 network successfully!')
        self.critic2.load_checkpoint(self.checkpoint_dir + 'Critic2/TD3_critic2_{}.pth'.format(episode))
        print('Loading critic2 network successfully!')
        self.target_critic2.load_checkpoint(self.checkpoint_dir +
                                            'Target_critic2/TD3_target_critic2_{}.pth'.format(episode))
        print('Loading target critic2 network successfully!')

zhui_num = 0
dao_num = 0

class TD3Trainer:

    def __init__(self, n_agents):
        self._n_agents = n_agents
        self._obs_dim = 48
        self._action_dim = 2

        self._agent = TD3(alpha=0.001, beta=0.005, state_dim=48,
                action_dim=2, actor_fc1_dim=128, actor_fc2_dim=64,
                critic_fc1_dim=128, critic_fc2_dim=64, ckpt_dir=args.ckpt_dir, gamma=0.99,
                tau=0.005, action_noise=0.05, policy_noise=0.1, policy_noise_clip=0.5,
                delay_time=2, max_size=1000000, batch_size=256)
        self._env = EnvWrapper(n_agents)
        self._now_ep = 0
        self._sw = SummaryWriter(f'./{MAIN_FOLDER}/logs/trainer')
        self._step = 0
        self.sumway = [0.0] *  n_agents
        self.endstep = [0.0] *  n_agents
        self.rew = [0.0] *  n_agents
        self.zhui_num = 0
        self.dao_num = 0

        self.lostend = 0
        self.lostway = 0
        self.losttime = 0

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

        self._agent.change(self._now_ep)

        enum_seq = [f'uav_{i}' for i in range(self._n_agents)]  # 里面是字符串

        states = self._env.reset()  # 返回初始化的场景
        done = {n: False for n in enum_seq}
        total_rew = {n: 0 for n in enum_seq}

        target_pos = self._env._env._cpp_env.getTarget()
        obs = self._env._env._cpp_env.getObstacles()

        while not real_done(done):
            actions = {}
            in_states = []
            for seq in enum_seq:
                in_states.append(states[seq])
            uavs = self._env._env._cpp_env.getUavs()
            w = self._env._env._cpp_env.getuavW()

            v = self._env._env._cpp_env.getuavV()
            out_actions = self._agent.choose_action(in_states,uavs,obs,self._now_ep,target_pos,w,v,self._n_agents,True)
            for i, seq in enumerate(enum_seq):
                actions[seq] = out_actions[i]
            die = self._env._env._cpp_env.getStatus()
            choices = []
            for i in range(self._n_agents):
                if not die[i]:
                    choices.append(i)

            next_states, rewards, done, info = self._env.step(actions)

            self._step += 1
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
        past_uav.x = lost_uav.x
        past_uav.y = lost_uav.y

        k = 0
        lb = [-10, -10]
        ub = [10, 10]
        a = 0
        b = 0
        num = 0
        long = 0
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
            num = num +1
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
                #print(neigh_uav)
                next = s.Tent_SSA(100, 2, lb, ub, 50, neigh_uav, lost_uav,obs)
                past_uav.x = lost_uav.x
                past_uav.y = lost_uav.y
                lost_uav.x = next[0][0]
                lost_uav.y = next[0][1]
                long = l2norm(lost_uav.x - past_uav.x, lost_uav.y - past_uav.y) +long
                if l2norm(lost_uav.x - target_pos.x, lost_uav.y - target_pos.y) <20:
                    self.lostend = 1
                    self.losttime = num
                    self.lostway = long
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

            out_actions = self._agent.choose_action(in_states,uavs,obs,self._now_ep,target_pos,w,v,self._n_agents,False)
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

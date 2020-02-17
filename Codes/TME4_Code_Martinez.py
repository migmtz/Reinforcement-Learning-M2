import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
from matplotlib import pyplot as plt

## Agents

import torch
import torch.nn as nn
from collections import deque

class NN(nn.Module):
    def __init__(self, inSize, outSize, layers=[]):
        super(NN, self).__init__()
        self.layers = nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize, x))
            inSize = x
        self.layers.append(nn.Linear(inSize, outSize))

    def forward(self, x):
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = torch.nn.functional.leaky_relu(x)
            x = self.layers[i](x)
        return x

class Memory():
    def __init__(self,max_size = 100000):
        self.memory = deque(maxlen = max_size)

    def __len__(self):
        return(len(self.memory))

    def sample(self,batch_size = 100):
        idx = np.random.randint(0,len(self.memory),batch_size)
        batch = [self.memory[i] for i in idx]
        return(batch)

    def store(self,obs):
        self.memory.append(obs)

class DQN(object):
    def __init__(self,action_space,epsilon,gamma,l_r,inSize,outSize,layers=[]):
        self.action_space = action_space
        self.Q = NN(inSize,outSize,layers)
        self.gamma = gamma
        self.epsilon = epsilon
        self.optim = torch.optim.Adam(self.Q.parameters(),l_r)
        self.loss = nn.MSELoss(reduction = "mean")

    def act(self,obs,reward,done):
        limit = (self.epsilon/self.action_space.n + 1 - self.epsilon)
        chance = np.random.uniform(0,1)
        if(chance < limit):
            action = torch.argmax(self.Q.forward(torch.tensor(obs).float())).item()
        else:
            action = self.action_space.sample()
        return(action)

    def update(self,obs,action,reward,done,obs_p):
        self.optim.zero_grad()
        target = reward + (1-done)*self.gamma*self.Q.forward(torch.tensor(obs).float())
        loss = self.loss(self.Q.forward(torch.tensor(obs).float()),target)
        loss.backward()
        self.optim.step()


class DQN_replay_target(object):
    def __init__(self,action_space,epsilon,gamma,l_r,inSize,outSize,layers=[]):
        self.action_space = action_space
        self.Q = NN(inSize,outSize,layers)
        self.Q_targ = copy.deepcopy(self.Q)
        self.gamma = gamma
        self.epsilon = epsilon
        self.optim = torch.optim.Adam(self.Q.parameters(),l_r)
        self.loss = nn.MSELoss(reduction = "mean")
        self.memory = Memory()
        self.reset_count = 0
        self.reset_n = 100

    def act(self,obs,reward,done):
        limit = (self.epsilon/self.action_space.n + 1 - self.epsilon)
        chance = np.random.uniform(0,1)
        if(chance < limit):
            action = torch.argmax(self.Q_targ.forward(torch.tensor(obs).float())).item()
        else:
            action = self.action_space.sample()
        return(action)

    def update(self,obs,action,reward,done,obs_p):
        self.optim.zero_grad()
        self.memory.store((torch.tensor(obs).float(),action,reward,done,torch.tensor(obs_p).float()))
        replay_list = self.memory.sample(100)
        aux,aux_a = torch.stack([s for (s,_,_,_,_) in replay_list]), torch.stack([torch.tensor(a) for (_,a,_,_,_) in replay_list]).reshape(-1,1)
        pred = self.Q.forward(aux)
        pred = pred.gather(1,aux_a)
        target = torch.tensor([[r + float(1-d)*self.gamma*self.Q_targ.forward(s_p).max().item()] for (_,_,r,d,s_p) in replay_list],requires_grad = True)
        #aux = torch.tensor([self.Q.forward(s)[a].item() for (s,a,_,_,_) in replay_list],requires_grad = True)
        loss = self.loss(pred,target)
        loss.backward()
        self.optim.step()
        self.reset_count += 1
        if self.reset_count == self.reset_n:
            self.Q_targ = copy.deepcopy(self.Q)
            self.reset_count = 0

## Apprentissage

if __name__ == '__main__':

    # Enregistrement des agents
    #      Cartpole

    env = gym.make('CartPole-v1') # obs of 4

    outdir = 'cartpole-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed(0)

    agent_dqn = DQN(env.action_space,epsilon=0.05,gamma = 0.99,l_r=1e-3,inSize=4,outSize=2,layers=[100,100])
    agent_rep_tar = DQN_replay_target(env.action_space,epsilon=0.05,gamma = 0.99,l_r=1e-3,inSize=4,outSize=2,layers=[100,100])

    #      LunarLander

    # env = gym.make('LunarLander-v2')
    #
    # outdir = 'LunarLander-v2/results'
    # envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    # env.seed(0)
    #
    # agent_dqn = DQN(env.action_space,epsilon=0.05,gamma = 0.99,l_r=1e-3,inSize=8,outSize=4,layers=[100,100])
    # agent_rep_tar = DQN_replay_target(env.action_space,epsilon=0.05,gamma = 0.99,l_r=1e-3,inSize=8,outSize=4,layers=[100,100])


    list_agent = [agent_dqn,agent_rep_tar]
    name = ["without replay/target","with replay/target"]

    episode_count = 150
    for k,agent in enumerate(list_agent):
        reward = 0
        done = False
        env.verbose = True
        np.random.seed(5)
        r_tot = 0
        r_graph = []
        r_epi = []
        rsum = 0
        a = list(agent.Q.parameters())[0].clone()
        for i in range(episode_count):
            obs = envm.reset()
            env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
            if env.verbose:
                env.render()
            j = 0
            rsum = 0
            while True:
                action = agent.act(obs, reward, done)
                obs_prec = obs
                obs, reward, done, _ = envm.step(action)
                agent.update(obs_prec,action,reward,done,obs)
                rsum += reward
                j += 1
                if env.verbose:
                    env.render()
                if done:
                    r_tot += rsum
                    r_graph += [r_tot/(i+1)]
                    if i%20 == 0:
                        r_epi += [rsum]
                        print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                    break

        print("done")
        plt.figure(1)
        #plt.plot(20*np.arange(len(r_epi)),r_epi,label ="episodic")
        plt.plot(np.arange(len(r_graph)),r_graph,label = "%s"%(name[k]))
        plt.title("DQN, Environment : CartPole")
        b = list(agent.Q.parameters())[0].clone()
        print(torch.equal(a.data, b.data))
    plt.legend()
    plt.grid()
    plt.show()
    env.close()

## Pour regarder la performance une fois entraîné

if __name__ == '__main__':

    # Enregistrement des agents
    #      Cartpole

    env = gym.make('CartPole-v1') # obs of 4

    outdir = 'cartpole-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed(0)

    #      LunarLander

    # env = gym.make('LunarLander-v2')
    #
    # outdir = 'LunarLander-v2/results'
    # envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    # env.seed(0)

    list_agent = [agent_dqn,agent_rep_tar]
    name = ["without replay/target","with replay/target"]

    episode_count = 10
    for k,agent in enumerate(list_agent):
        reward = 0
        done = False
        env.verbose = True
        np.random.seed(5)
        r_tot = 0
        r_graph = []
        r_epi = []
        rsum = 0
        for i in range(episode_count):
            obs = envm.reset()
            env.verbose = (i % 2 == 0 and i > 0)  # afficher 1 episode sur 100
            if env.verbose:
                env.render()
            j = 0
            rsum = 0
            while True:
                action = agent.act(obs, reward, done)
                obs_prec = obs
                obs, reward, done, _ = envm.step(action)
                rsum += reward
                j += 1
                if env.verbose:
                    env.render()
                if done:
                    r_tot += rsum
                    r_graph += [r_tot/(i+1)]
                    if i%20 == 0:
                        r_epi += [rsum]
                        print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                    break
        print("done")
        plt.figure(14)
        #plt.plot(20*np.arange(len(r_epi)),r_epi,label ="episodic")
        plt.plot(np.arange(len(r_graph)),r_graph,label = "%s"%(name[k]))
        plt.title("DQN, Environment : CartPole")
    plt.legend()
    plt.grid()
    plt.show()
    env.close()
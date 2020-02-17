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

class online_a2c(object):
    def __init__(self,action_space,epsilon,gamma,l_r_V,l_r_pi,inSize,outSize,layers_V,layers_pi):
        self.action_space = action_space
        self.pi = NN(inSize,outSize,layers_pi)
        self.V = NN(inSize,1,layers_V)
        self.optim_V = torch.optim.Adam(self.V.parameters(),l_r_V)
        self.optim_pi = torch.optim.Adam(self.pi.parameters(),l_r_pi)
        self.gamma = gamma
        self.epsilon = epsilon
        self.loss_V = nn.SmoothL1Loss(reduction="mean")

    def act(self,obs,reward,done):
        limit = (self.epsilon/self.action_space.n + 1 - self.epsilon)
        chance = np.random.uniform(0,1)
        if(chance < limit):
            action = torch.argmax(self.pi.forward(torch.tensor(obs).float())).item()
        else:
            action = self.action_space.sample()
        return(action)

    def update(self,obs,action,reward,done,obs_p):
        # Update V
        self.optim_V.zero_grad()
        target_V = reward + float(1-done)*self.V.forward(torch.tensor(obs_p).float())
        loss_V = self.loss_V(self.V.forward(torch.tensor(obs).float()),target_V)
        loss_V.backward()
        self.optim_V.step()

        # Update Pi
        self.optim_pi.zero_grad()
        A = reward + self.gamma*self.V.forward(torch.tensor(obs_p).float()) - self.V.forward(torch.tensor(obs).float())
        aux = self.pi.forward(torch.tensor(obs).float())[action]
        loss_pi = -torch.log(aux)*A
        loss_pi.backward()
        self.optim_pi.step()

class batch_a2c(object):
    def __init__(self,action_space,epsilon,gamma,l_r_V,l_r_pi,inSize,outSize,layers_V,layers_pi):
        self.action_space = action_space
        self.pi = NN(inSize,outSize,layers_pi)
        self.V = NN(inSize,1,layers_V)
        self.optim_V = torch.optim.Adam(self.V.parameters(),l_r_V)
        self.optim_pi = torch.optim.Adam(self.pi.parameters(),l_r_pi)
        self.gamma = gamma
        self.epsilon = epsilon
        self.loss_V = nn.SmoothL1Loss(reduction="mean")
        self.episode = []

    def act(self,obs,reward,done):
        limit = (self.epsilon/self.action_space.n + 1 - self.epsilon)
        chance = np.random.uniform(0,1)
        if(chance < limit):
            action = torch.argmax(self.pi.forward(torch.tensor(obs).float())).item()
        else:
            action = self.action_space.sample()
        return(action)

    def update(self,obs,action,reward,done,obs_p):
        self.episode += [(obs,action,reward,done,obs_p)]
        if done:
            self.update_weights()
            self.episode = []

    def update_weights(self):
        # Update V
        self.optim_V.zero_grad()
        R = 0
        Vloss = torch.zeros(1,requires_grad = True)
        for (s,_,r,_,_) in reversed(self.episode):
            R = r + self.gamma*R
            Vloss = Vloss + self.loss_V(self.V.forward(torch.tensor(s).float()),torch.tensor(R))
        Vloss.backward()
        self.optim_V.step()

        # Update pi
        self.optim_pi.zero_grad()
        R = 0
        piloss = torch.zeros(1,requires_grad = True)
        for (s,a,r,_,s_p) in self.episode:
            A = r + self.gamma*self.V.forward(torch.tensor(s_p).float()) - self.V.forward(torch.tensor(s).float())
            aux = self.pi.forward(torch.tensor(s).float())[a]
            piloss = piloss - torch.log(aux)*A
        piloss.backward()
        self.optim_pi.step()

## Apprentissage

if __name__ == '__main__':

    # Enregistrement des agents
    #      Cartpole

    # env = gym.make('CartPole-v1') # obs of 4
    #
    # outdir = 'cartpole-v0/random-agent-results'
    # envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    # env.seed(0)
    #
    # agent_online = online_a2c(env.action_space,epsilon=0.05,gamma = 0.999,l_r=1e-3,inSize=4,outSize=2,layers_V = [100,100],layers_pi=[100,100])
    # agent_batch = batch_a2c(env.action_space,epsilon=0.05,gamma = 0.999,l_r=1e-3,inSize=4,outSize=2,layers_V = [100,100],layers_pi=[100,100])
    # envir = "CartPole"

    #      LunarLander

    env = gym.make('LunarLander-v2')

    outdir = 'LunarLander-v2/results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed(0)

    agent_online = online_a2c(env.action_space,epsilon=0.1,gamma = 0.999,l_r_V=1e-3,l_r_pi=1e-3,inSize=8,outSize=4,layers_V = [20,20],layers_pi=[40,40,40])
    agent_batch = batch_a2c(env.action_space,epsilon=0.1,gamma = 0.999,l_r_V=1e-3,l_r_pi=1e-3,inSize=8,outSize=4,layers_V = [20,20],layers_pi=[40,40,40])
    envir = "LunarLander"


    list_agent = [agent_online,agent_batch]
    name = ["Online","Batch"]

    episode_count = 500
    for k,agent in enumerate(list_agent):
        reward = 0
        done = False
        env.verbose = True
        np.random.seed(5)
        r_tot = 0
        r_graph = []
        r_epi = []
        rsum = 0
        a = list(agent.pi.parameters())[0].clone()
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
        b = list(agent.pi.parameters())[0].clone()
        print(torch.equal(a.data, b.data))
    plt.title("A2C, Environment : "+ envir)
    plt.legend()
    plt.grid()
    plt.show()
    env.close()





## Pour regarder la performance une fois entraîné

if __name__ == '__main__':

    # Enregistrement des agents
    #      Cartpole

    # env = gym.make('CartPole-v1') # obs of 4
    #
    # outdir = 'cartpole-v0/random-agent-results'
    # envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    # env.seed(0)

    #      LunarLander

    env = gym.make('LunarLander-v2')

    outdir = 'LunarLander-v2/results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed(0)

    list_agent = [agent_online,agent_batch]
    name = ["Online","Batch"]

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
        plt.title("DQN, Environment : "+ envir)
    plt.legend()
    plt.grid()
    plt.show()
    env.close()
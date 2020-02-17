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
    def __init__(self, inSize, outSize, layers=[],softmax = False):
        super(NN, self).__init__()
        self.layers = nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize, x))
            inSize = x
        self.layers.append(nn.Linear(inSize, outSize))
        self.soft_flag = softmax

    def forward(self, x):
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = torch.nn.functional.leaky_relu(x)
            x = self.layers[i](x)
        if self.soft_flag:
            x = torch.nn.functional.softmax(x,dim=-1)
        return x

class KL_PPO(object):
    def __init__(self,action_space,epsilon,beta,delta,gamma,K,l_r,inSize,outSize,layers):
        self.action_space = action_space
        self.beta = beta
        self.delta = delta
        self.epsilon = epsilon
        self.gamma = gamma
        self.K = K
        self.pi = NN(inSize,outSize,layers,softmax = True)
        self.pi_aux = copy.deepcopy(self.pi)
        self.V = NN(inSize,1,layers)
        self.optim_V = torch.optim.Adam(self.V.parameters(),l_r)
        self.optim_pi = torch.optim.Adam(self.pi_aux.parameters(),l_r)
        self.loss_V = nn.SmoothL1Loss(reduction="mean")
        self.loss_kl = nn.KLDivLoss(reduction="sum")
        self.batch = []
        self.episodes = []

    def act(self,obs,reward,done):
        limit = (self.epsilon/self.action_space.n + 1 - self.epsilon)
        chance = np.random.uniform(0,1)
        if(chance < limit):
            action = torch.argmax(self.pi.forward(torch.tensor(obs).float())).item()
        else:
            action = self.action_space.sample()
        return(action)

    def update(self,obs,action,reward,done,obs_p):
        self.episodes += [(obs,action,reward,done,obs_p)]
        if done:
            self.batch += [self.episodes]
            self.episodes = []
            if self.K == len(self.batch):
                self.update_policy()
                self.batch = []

    def update_policy(self):
        # Update V
        self.optim_V.zero_grad()
        for episode in self.batch:
            reward_exp = torch.zeros(1,requires_grad = True)
            loss_v_tot = torch.zeros(1,requires_grad = True)
            for s,a,r,d,s_p in reversed(episode):
                reward_exp = self.gamma*reward_exp + r
                loss_v_tot = loss_v_tot + self.loss_V(self.V.forward(torch.tensor(s).float()),reward_exp.float())
        loss_v_tot = loss_v_tot
        loss_v_tot.backward()
        self.optim_V.step()

        # Update pi
        for episode in self.batch:
            kl = torch.zeros(1,requires_grad = False)
            self.optim_pi.zero_grad()
            R = torch.zeros(1,requires_grad = True)
            loss_pi_ep = torch.zeros(1,requires_grad = True)
            for s,a,r,d,s_p in reversed(episode):
                R = self.gamma*R + r
                A = R - self.V.forward(torch.tensor(s).float())
                aux = self.pi_aux.forward(torch.tensor(s).float())
                old = self.pi.forward(torch.tensor(s).float())
                kl_loss = self.loss_kl(torch.log(aux),old)
                kl += kl_loss
                loss_pi_ep = loss_pi_ep - A*(aux[a]/old[a]) + self.beta*kl_loss
            loss_pi_ep.backward()
            self.optim_pi.step()
            if kl >= 1.5*self.delta:
                self.beta = 2*self.beta
            if kl <= self.delta/1.5:
                self.beta = 0.5*self.beta
        self.pi = copy.deepcopy(self.pi_aux)

class Clip_PPO(object):
    def __init__(self,action_space,epsilon,threshold,gamma,K,l_r,inSize,outSize,layers):
        self.action_space = action_space
        self.threshold = threshold
        self.epsilon = epsilon
        self.gamma = gamma
        self.K = K
        self.pi = NN(inSize,outSize,layers,softmax = True)
        self.pi_aux = copy.deepcopy(self.pi)
        self.V = NN(inSize,1,layers)
        self.optim_V = torch.optim.Adam(self.V.parameters(),l_r)
        self.optim_pi = torch.optim.Adam(self.pi_aux.parameters(),l_r)
        self.loss_V = nn.SmoothL1Loss(reduction="mean")
        self.batch = []
        self.episodes = []

    def act(self,obs,reward,done):
        limit = (self.epsilon/self.action_space.n + 1 - self.epsilon)
        chance = np.random.uniform(0,1)
        if(chance < limit):
            action = torch.argmax(self.pi.forward(torch.tensor(obs).float())).item()
        else:
            action = self.action_space.sample()
        return(action)

    def update(self,obs,action,reward,done,obs_p):
        self.episodes += [(obs,action,reward,done,obs_p)]
        if done:
            self.batch += [self.episodes]
            self.episodes = []
            if self.K == len(self.batch):
                self.update_policy()
                self.batch = []

    def update_policy(self):
        # Update V
        self.optim_V.zero_grad()
        for episode in self.batch:
            reward_exp = torch.zeros(1,requires_grad = True)
            loss_v_tot = torch.zeros(1,requires_grad = True)
            for s,a,r,d,s_p in reversed(episode):
                reward_exp = self.gamma*reward_exp + r
                loss_v_tot = loss_v_tot + self.loss_V(self.V.forward(torch.tensor(s).float()),reward_exp.float())
        loss_v_tot = loss_v_tot
        loss_v_tot.backward()
        self.optim_V.step()

        # Update pi
        for episode in self.batch:
            self.optim_pi.zero_grad()
            R = torch.zeros(1,requires_grad = True)
            loss_pi_ep = torch.zeros(1,requires_grad = True)
            for s,a,r,d,s_p in reversed(episode):
                R = self.gamma*R + r
                A = R - self.V.forward(torch.tensor(s).float())
                aux = self.pi_aux.forward(torch.tensor(s).float())
                old = self.pi.forward(torch.tensor(s).float())
                r_t = aux[a]/old[a]
                clipped = torch.clamp(r_t,1-self.threshold,1+self.threshold)
                loss_pi_ep = loss_pi_ep - torch.min(A*r_t, A*clipped)
            loss_pi_ep.backward()
            self.optim_pi.step()
        self.pi = copy.deepcopy(self.pi_aux)


## Apprentissage

if __name__ == '__main__':


    env = gym.make('CartPole-v1') # obs of 4

    # Enregistrement de l'Agent

    outdir = 'cartpole-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed(0)

    agent_kl = KL_PPO(env.action_space,epsilon=0.05,beta=0.8,delta=250,gamma = 0.999,K=250,l_r=1e-3,inSize=4,outSize=2,layers=[200,200])
    # agent_clip = Clip_PPO(env.action_space,epsilon=0.1,threshold=0.1,gamma = 0.999,K=250,l_r=1e-4,inSize=4,outSize=2,layers=[50,50])
    #
    # list_agent = [agent_kl,agent_clip]
    # name = ["kl","clip"]
    list_agent = [agent_kl]
    name = ["kl"]

    episode_count = 1300
    for k,agent in enumerate(list_agent):
        reward = 0
        done = False
        env.verbose = True
        np.random.seed(5)
        r_tot = 0
        r_graph = []
        r_epi = []
        rsum = 0
        # a = list(agent.V.parameters())[0].clone()
        #print(list(agent.pi.parameters())[0].clone())
        for i in range(episode_count):
            obs = envm.reset()
            env.verbose = (i % 200 == 0 and i > 0)  # afficher 1 episode sur 100
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
                    if i%10 == 0:
                        r_epi += [rsum]
                    print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                    break

        print("done")
        # b = list(agent.V.parameters())[0].clone()
        # print(torch.equal(a.data, b.data))
        #print(list(agent.pi.parameters())[0].clone())
        plt.figure(1)
        #plt.plot(10*np.arange(len(r_epi)),r_epi,label ="episodic")
        plt.plot(np.arange(len(r_graph)),r_graph,label = "%s"%name[k])
    plt.title("KL PPO, Environment : CartPole")
    plt.legend()
    plt.grid()
    plt.show()
    env.close()
import matplotlib

matplotlib.use("TkAgg")
import gym
import multiagent
import multiagent.scenarios
import multiagent.scenarios.simple_tag as simple_tag
import multiagent.scenarios.simple_tag as simple_spread
import multiagent.scenarios.simple_tag as simple_adversary
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from gym import wrappers, logger
import numpy as np
import copy
from matplotlib import pyplot as plt

"""
Code for creating a multiagent environment with one of the scenarios listed
in ./scenarios/.
Can be called by using, for example:
    env = make_env('simple_speaker_listener')
After producing the env object, can be used similarly to an OpenAI gym
environment.

A policy using this environment must output actions in the form of a list
for all agents. Each element of the list should be a numpy array,
of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
communication actions in this array. See environment.py for more details.
"""
## Agents

import numpy as np
from torch import nn
import torch
from collections import deque

class NN(nn.Module):
    def __init__(self, inSize, outSize, layers=[],softmax = False):
        super(NN, self).__init__()
        self.layers = nn.ModuleList([])
        self.batch = nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize, x))
            inSize = x
            self.batch.append(nn.BatchNorm1d(inSize))
        self.layers.append(nn.Linear(inSize, outSize))
        self.soft_flag = softmax

    def forward(self, x):
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = torch.nn.functional.leaky_relu(x)
            x = self.batch[i-1](x)
            x = self.layers[i](x)
        if self.soft_flag:
            x = torch.nn.functional.softmax(x,dim=-1)
        return x

class Memory():
    def __init__(self,i1,i2,i3,max_size = 100000):
        self.memory = deque(maxlen = max_size)
        self.idx = [i1,i2,i3]

    def sample(self,batch_size = 10):
        idx = np.random.randint(0,len(self.memory),batch_size)
        batch = np.vstack([self.memory[i] for i in idx])
        return(batch)

    def store(self,o1,a,r,o2):
        res = np.concatenate((np.hstack(o1),np.hstack(a),np.array([r[0]]),np.hstack(o2)))
        self.memory.append(res) ## has to store by () tuple

class MADDPG_spread(object):
    def __init__(self,q_size = 48,obs_size = 14,gamma = 0.999,tau = 1e-3,C = 10, l_r = 1e-4,layer = [40,40,40]):
        self.Q = [NN(q_size,1,layer),NN(q_size,1,layer),NN(q_size,1,layer)]
        self.Q_hat = [NN(q_size,1,layer),NN(q_size,1,layer),NN(q_size,1,layer)]

        self.mu = [NN(obs_size,2,layer),NN(obs_size,2,layer),NN(obs_size,2,layer)]
        self.mu_hat = [NN(obs_size,2,layer),NN(obs_size,2,layer),NN(obs_size,2,layer)]

        self.memory = Memory(14,2,1)
        self.gamma = gamma
        self.tau = tau
        self.C = C
        self.compteur = 0

        self.loss_critic = nn.SmoothL1Loss()

        self.optim_Q = [torch.optim.Adam(params = self.Q[i].parameters(),lr = l_r) for i in range(3)]
        self.optim_mu = [torch.optim.Adam(params = self.mu[i].parameters(),lr = l_r) for i in range(3)]

    def act(self,o):
        with torch.no_grad():
            a = []
            for i,obs in enumerate(o):
                obs = torch.tensor(obs).float().reshape(1,14)
                a.append(((self.mu[i].eval()(obs)).detach().numpy() + (np.random.rand(2)-0.5)).reshape(2))
        return(a)

    def store(self,o1,a,r,o2):
        self.memory.store(o1,a,r,o2)

    def update(self):
        sample_val = 1000
        for i,_ in enumerate(self.Q):

            # Sampled replays (transform every sample in tensors with sample_val number of lines)

            sample = self.memory.sample(sample_val)
            x = [torch.tensor(sample[:,0:14]).float(),torch.tensor(sample[:,14:28]).float(),torch.tensor(sample[:,28:42]).float()]
            a = torch.tensor(sample[:,42:48]).float()
            r = torch.tensor(sample[:,48:49]).float()
            x_p = [torch.tensor(sample[:,49:63]).float(),torch.tensor(sample[:,63:77]).float(),torch.tensor(sample[:,77:91]).float()]

            a_p = torch.cat([self.mu_hat[i](o) for i,o in enumerate(x_p)],1)
            mu_oj = torch.cat([self.mu[i](o) for i,o in enumerate(x)],1)
            x = torch.cat(x,1)
            x_p = torch.cat(x_p,1)

            # Critic update

            loss_crit = self.loss_critic(r + self.gamma*self.Q_hat[i](torch.cat((x_p,a_p),1)),self.Q[i](torch.cat((x,a),1)))
            l1 = loss_crit.item()

            self.optim_Q[i].zero_grad()
            loss_crit.backward()
            self.optim_Q[i].step()

            # Actor update

            loss_actor = -torch.mean((self.Q[i](torch.cat((x,mu_oj),1))))
            l2 = loss_actor.item()

            self.optim_mu[i].zero_grad()
            loss_actor.backward()
            self.optim_mu[i].step()

         # Target Update

        if self.compteur == self.C:
            for i,_ in enumerate(self.Q):
                for theta,theta_p in zip(self.Q[i].parameters(),self.Q_hat[i].parameters()):
                    theta_p.data.copy_(self.tau*theta.data + (1- self.tau)*theta_p.data)
                for theta,theta_p in zip(self.mu[i].parameters(),self.mu_hat[i].parameters()):
                    theta_p.data.copy_(self.tau*theta.data + (1- self.tau)*theta_p.data)
            self.compteur = 0
        else:
            self.compteur += 1
        return(l1,l2)

## Apprentissage
def make_env(scenario_name, benchmark=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    world.dim_c = 0
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    env.discrete_action_space = False
    env.discrete_action_input = False
    scenario.reset_world(world)
    return env,scenario,world

if __name__ == '__main__':
    agent = MADDPG_spread(gamma = 0.99,tau = 0.5,C = 30, l_r = 1e-4,layer = [200,200])

    env,scenario,world = make_env('simple_spread')
    r_tot = []
    for episode in range(21):
        o = env.reset()
        r_tot = []
        for k in range(1,71):
            o_aux = o
            a = agent.act(o)
            o, r, d, i = env.step(a)
            agent.store(o_aux,a,r,o)
            l1,l2 = agent.update()
            env.render(mode="none")
            if k%20 == 0:
                print(k,r[0])
            r_tot += [r[0]]
        if episode%4 == 0:
            print("Episode :", episode)
            plt.figure(0)
            plt.plot(r_tot,label = "Episode : %s"%(episode))
    plt.title("MADDPG: Spread")
    plt.legend()
    plt.grid()
    plt.show()
    env.close()

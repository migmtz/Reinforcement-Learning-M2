import matplotlib


matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy

from matplotlib import pyplot as plt

## Tracer la référence avec l'agent randomisé

class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

if __name__ == '__main__':

    agent_rand = RandomAgent(env.action_space)
    episode_count = 1000
    reward = 0
    done = False
    rsum = 0
    reward_total = []
    FPS = 0.0001
    for i in range(episode_count):
        obs = envm.reset()
        env.verbose = (i % 1000 == 0 and i > 0)  # afficher 1 episode sur 100
        # if env.verbose:
        #     env.render(FPS)
        j = 0
        rsum = 0
        while True:
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = envm.step(action)
            rsum += reward
            j += 1
            # if env.verbose:
            #     env.render(FPS)
            if done:
                reward_total += [rsum]
                # if i%100 == 0:
                #     print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    plt.figure(1)
    plt.plot(np.arange(episode_count),np.cumsum(reward_total)/np.arange(1,episode_count+1),label="Random")
    env.close()


## Agents:

import numpy as np

class Q_Learning_agent(object):
    def __init__(self,action_space,statedic,alpha,gamma):
        self.action_space = action_space
        self.statedic = statedic
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros((len(statedic),action_space.n)) #state x action

    def act(self,obs,reward,done):
        i = self.statedic[str(obs.tolist())]
        return(np.argmax(self.Q[i]))

    def update(self,obs,action,reward,done,obs_p):
        i = self.statedic[str(obs.tolist())]
        i_1 = self.statedic[str(obs_p.tolist())]
        self.Q[i,action] = self.Q[i,action] + self.alpha*(reward + self.gamma*np.max(self.Q[i_1])  - self.Q[i,action])


class Sarsa_agent(object):
    def __init__(self,action_space,statedic,alpha,gamma,epsilon):
        self.action_space = action_space
        self.statedic = statedic
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros((len(statedic),action_space.n)) #state x action
        self.epsilon = epsilon

    def act(self,obs,reward,done):
        i = self.statedic[str(obs.tolist())]
        chance = np.random.uniform(0,1)
        if chance < self.epsilon:
            return(env.action_space.sample())
        else:
            return(np.argmax(self.Q[i]))

    def update(self,obs,action,reward,done,obs_p):
        i = self.statedic[str(obs.tolist())]
        i_1 = self.statedic[str(obs_p.tolist())]
        a_1 = np.argmax(self.Q[i_1])
        self.Q[i,action] = self.Q[i,action] + self.alpha*(reward + self.gamma*self.Q[i_1,a_1]  - self.Q[i,action])


class Dyna_Q_agent(object):
    def __init__(self,action_space,statedic,alpha,gamma,epsilon,alpha_R,k = 10):
        self.action_space = action_space
        self.statedic = statedic
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros((len(statedic),action_space.n)) #state x action
        self.epsilon = epsilon
        self.R = np.zeros((len(statedic),action_space.n,len(statedic)))
        self.P = np.zeros((len(statedic),action_space.n,len(statedic)))
        self.alpha_R = alpha_R
        self.k = k

    def act(self,obs,reward,done):
        i = self.statedic[str(obs.tolist())]
        chance = np.random.uniform(0,1)
        if chance < self.epsilon:
            return(self.action_space.sample())
        else:
            return(np.argmax(self.Q[i]))

    def update(self,obs,action,reward,done,obs_p):
        #Update Q
        i = self.statedic[str(obs.tolist())]
        i_1 = self.statedic[str(obs_p.tolist())]
        self.Q[i,action] = self.Q[i,action] + self.alpha*(reward + self.gamma*np.max(self.Q[i_1])  - self.Q[i,action])
        #Update MDP
        self.R[i,action,i_1] += self.alpha_R*(reward - self.R[i,action,i_1])
        for j in range(len(statedic)):
            self.P[i,action,j] += self.alpha_R*(float(j == i_1) - self.P[i,action,i_1])
        #Sampled Update
        rand_state = np.random.randint(0,len(statedic),self.k)
        rand_action = np.random.randint(0,self.action_space.n,self.k)
        for s,a in zip(rand_state,rand_action):
            aux = [self.P[s,a,s_p]*(self.R[s,a,s_p] + self.gamma*np.max(self.Q[s_p])) for s_p in range(len(statedic))]
            self.Q[s,a] += self.alpha*(np.sum(aux) - self.Q[s,a])


## Boucle pour les agents Q-learning

if __name__ == '__main__':


    env = gym.make("gridworld-v0")

    # Enregistrement de l'Agent
    # Faire un fichier de log sur plusieurs scenarios
    outdir = 'gridworld-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.setPlan("gridworldPlans/plan1.txt", {0: -0.01, 3: 1, 4: 1, 5: -1, 6: -1})
    statedic,_ = env.getMDP()

    agent_Q = Q_Learning_agent(env.action_space,statedic,alpha = 1e-1,gamma = 0.999)
    agent_sarsa = Sarsa_agent(env.action_space,statedic,alpha = 1e-1,gamma = 0.999,epsilon = 0.05)
    agent_dyna = Dyna_Q_agent(env.action_space,statedic,alpha = 1e-1,gamma = 0.999,epsilon = 0.05,alpha_R =1e-3, k = 10)

    list_agent = [agent_Q,agent_sarsa,agent_dyna]
    name = ["Q_learning","Sarsa","Dyna-Q"]

    for k,agent in enumerate(list_agent):
        env.seed(0)  # Initialiser le pseudo aleatoire
        episode_count = 1000
        reward = 0
        done = False
        r_tot = 0
        r_graph = []
        r_epi = []
        FPS = 0.0001
        for i in range(episode_count):
            obs = envm.reset()
            env.verbose = (i % 1000 == 0 and i > 0)  # afficher 1 episode sur 100
            if env.verbose:
                env.render(FPS)
            j = 0
            rsum = 0
            while True:
                action = agent.act(obs, reward, done)
                prev_obs = obs
                obs, reward, done, _ = envm.step(action)
                agent.update(prev_obs,action,reward,done,obs)
                rsum += reward
                j += 1
                if env.verbose:
                    env.render(FPS)
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
    env.close()
    plt.title("Q learning, Environment : plan1")
    plt.legend()
    plt.grid()
    plt.show()

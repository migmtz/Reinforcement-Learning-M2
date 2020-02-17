##
import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
from matplotlib import pyplot as plt

## Agents

class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

class policy_iteration_agent(object):
    def __init__(self, action_space, statedic, mdp, epsilon = 0.01, gamma = 0.9):
        self.action_space = action_space
        self.statedic = statedic
        self.mdp = mdp
        self.epsilon = epsilon
        self.gamma = gamma
        self.pi = np.random.randint(action_space.n, size = len(statedic))

    def act(self, obs, reward, done):
        obs = self.statedic[str(obs.tolist())]
        return(self.pi[obs])

    def fit(self):
        flag_pi = False
        while not(flag_pi):
            self.V = np.random.rand(len(self.statedic))
            flag_V = False
            while not(flag_V):
                V_precedent = self.V
                for s in self.mdp.keys():
                    i = self.statedic[s]
                    action_pi = self.mdp[s][self.pi[i]]
                    for (p, s_p, r, d) in action_pi:
                        self.V[i] += p*(r + self.gamma*V_precedent[self.statedic[s_p]])
                        if d:
                            self.V[self.statedic[s_p]] = r
                    #self.V[i] = np.sum([ p*(r + self.gamma*V_precedent[self.statedic[s_p]]) for (p, s_p, r, d) in action_pi])

                flag_V = np.linalg.norm(self.V - V_precedent) <= self.epsilon
                print(self.V)
            pi_precedent = self.pi
            for s in mdp.keys():
                aux = [np.sum([p*(r + self.gamma*self.V[self.statedic[s_p]]) for (p, s_p, r, d) in self.mdp[s][a]])  for a in range(self.action_space.n)]
                self.pi[self.statedic[s]] = np.argmax(aux)
            #aux = np.array([ [ np.sum([p*(r + self.gamma*self.V[self.statedic[s_p]]) for (p, s_p, r, d) in self.mdp[s][a]])  for a in range(self.action_space.n)]   for i,s in enumerate(self.statedic.keys())]) # 9 lignes et 4 colonnes
            #self.pi = aux.argmax(axis=1)
            flag_pi = np.array_equal(self.pi,pi_precedent)
        return('done fitting')


class value_iteration_agent(object):

    def __init__(self, action_space, statedic, mdp, epsilon = 0.01, gamma = 0.9):
        self.action_space = action_space
        self.statedic = statedic
        self.mdp = mdp
        self.epsilon = epsilon
        self.gamma = gamma
        self.pi = np.random.randint(action_space.n, size = len(statedic))

    def act(self, obs, reward, done):
        obs = self.statedic[str(obs.tolist())]
        return(self.pi[obs])

    def fit(self):
        self.V = np.random.rand(len(statedic))
        flag_V = False
        while not(flag_V):
            old_V = self.V
            for s in self.mdp.keys():
                i = self.statedic[s]
                max_sum = [np.sum([p*(r + self.gamma*old_V[self.statedic[s_p]]) for (p,s_p,r,d) in self.mdp[s][a]]) for a in range(self.action_space.n)]
                self.V[i] = np.argmax(max_sum)
            flag_V = np.linalg.norm(self.V-old_V) <= self.epsilon
        for s in self.mdp.keys():
            self.pi[self.statedic[s]] = np.argmax([np.sum([p*(r + self.gamma*self.V[self.statedic[s_p]]) for (p,s_p,r,d) in self.mdp[s][a]]) for a in range(self.action_space.n)])
        return('done fitting')

## Apprentissage

if __name__ == '__main__':


    env = gym.make("gridworld-v0")
    env.seed(0)  # Initialisele seed du pseudo-random
    print(env.action_space)  # Quelles sont les actions possibles
    print(env.step(3))  # faire action 1 et retourne l'observation, le reward, et un done un booleen (jeu fini ou pas)
    env.render()  # permet de visualiser la grille du jeu
    env.render(mode="human") #visualisation sur la console
    statedic, mdp = env.getMDP()  # recupere le mdp : statedic
    print("Nombre d'etats : ",len(statedic))  # nombre d'etats ,statedic : etat-> numero de l'etat
    state, transitions = list(mdp.items())[0]
    print(state)  # un etat du mdp
    print(transitions)  # dictionnaire des transitions pour l'etat :  {action-> [proba,etat,reward,done]}

    # Execution avec un Agent
# Faire un fichier de log sur plusieurs scenarios
    outdir = 'gridworld-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.setPlan("gridworldPlans/plan0.txt", {0: -0.01, 3: 1, 4: 1, 5: -2, 6: -2})
    env.seed()  # Initialiser le pseudo aleatoire
    statedic, mdp = env.getMDP()

    # # # # # Random Agent # # # # #

    agent_rand = RandomAgent(env.action_space)

    # # # # # Policy Iteration # # # # #

    agent_pol = policy_iteration_agent(env.action_space,statedic,mdp,1e-6,0.99)
    agent_pol.fit()

    # # # # # Value Iteration # # # # #

    agent_val = value_iteration_agent(env.action_space,statedic,mdp,1e-6,0.99)
    agent_val.fit()

    list_agent = [agent_rand, agent_pol, agent_val]

    name = ["random","policy","value"]

    # # # # # # # # # # # # # # #
    for k,agent in enumerate(list_agent):
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
        plt.figure(0)
        plt.plot(np.arange(episode_count),np.cumsum(reward_total)/np.arange(1,episode_count+1),label="%s"%(name[k]))
    plt.title("TME 2")
    plt.grid()
    plt.legend()
    plt.show()
    env.close()

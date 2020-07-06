import numpy as np 
import gym
from gym import spaces
from gym.utils import seeding
import random

class GrnEnv(gym.Env):
    def __init__(self,data,booldata,dis,eng):
        # data: a struct of some matrix
        self.numGene = data.shape[1]
        self.numCell = data.shape[0]
        self.action_space = spaces.Discrete(self.numGene*self.numGene)
        # -- improve representation
        ### -- self.observation_space = spaces.Box(np.zeros(numGene),np.zeros(numGene))
        ## observation -- cell state ++ changed gene...
        # -- increase features
        self.numGene = self.numGene 
        self.seed()
        self.state = booldata[0]
        self.data = data
        self.dis = dis
        self.eng = eng
        self.booldata = booldata
        self.start_point = booldata[0]
        self.data_pair = self.map_data()
        self.dict = self.map_bool()
        self.dict_en = self.calcu_mean_energy()
    def seed(self,seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def step(self,action):
        # action is an array
        action = action.reshape(self.numGene,self.numGene)
        if(sum(sum(action))==0):
            a = random.randint(0,self.numGene-1)
            b = random.randint(0,self.numGene-1)
            action[a][b] = 1
        next_state = self.get_next_state(action)
        r1 = len(self.dict[tuple(next_state)])  ## frequence
        mean_energy2 = self.dict_en[tuple(next_state)]#[x[1] for x in self.dict[tuple(next_state)]]
        englist1 = self.dict_en[tuple(self.state)]
        mean_energy1 = np.mean(englist1)  ## mean energy
        r2 = mean_energy1-mean_energy2
        #mean_dist = 
        #r3 = 
        reward = 0.001*(r2*10+r1)
        done = False
        self.state = next_state
        return next_state,reward,done,{}
    def _get_obs(self):
        pass
    def reset(self):
        return self.start_point
    ## for _step function
    ## action: a matrix or some other
    def get_next_state(self,action):
        state = self.state
        next_state = state
        for i in range(self.numGene):
            idx = np.where(action[:,i]==1)[0]
            if(len(idx)==0):
                continue
            for id in idx:
                if(state[id]==1):
                    next_state[id]=1
        return next_state
    def map_data(self):
        data_pair = []
        for i in range(self.data.shape[0]):
            data_pair.append((self.data[i],self.eng[i]))
        return data_pair
    def map_bool(self):
        dict = {}
        for i in range(self.booldata.shape[0]):
            dict[tuple(self.booldata[i])] = []
        for i in range(self.data.shape[0]):
            dict[tuple(self.booldata[i])].append(self.data_pair[i])
        return dict
    def calcu_mean_energy(self):
        dict_en = {}
        for i in range(self.booldata.shape[0]):
            englist = [x[1] for x in self.dict[tuple(self.booldata[i])]]
            mean_energy = np.mean(englist)  ## mean energy
            dict_en[tuple(self.booldata[i])] = mean_energy
        return dict_en

"""
The code of paper titled "AI-enabled Priority and Auction-Based Spectrum Management for 6G", is published in IEEE WCNC 2024, Dubai.
@author: Farshad (zeinalifarshad1375@gmail.com) 
"""
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import time
from Env import Environment
from ddpg_torch import Agent
import os

np.random.seed(1375)
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#____ Environment parameters ____
n_vsp = 5
delta = np.zeros([n_vsp])
delta[0] = np.random.randint(1,4)
delta[1] = delta[0] * 1
delta[2] = delta[0] * 2
delta[3] = delta[0] * 3
delta[4] = delta[0] * 4
phi = np.array([delta[0]*1, delta[1]*2, delta[2]*3, delta[3]*4, delta[4]*5])

env = Environment(n_vsp,delta,phi)

#%% Learning parameter
alpha=0.0000001   #Actor networks learning rate
beta=0.0000001    #Critic networks learning rate
gamma=0.9     #discount factor
tau=0.01      #Target networks soft update parameter
lr = 0.001
max_size=1000000     #replay buffer size
fc1_dims=400     #first layer of neural network norons numbers
fc2_dims=300      #second layer of neural network norons numbers
batch_size=32   
n_input = 3 * n_vsp    #states (inputs of the network)
n_actions = n_vsp      #actions (outputs of the network)
#%% 
agent = Agent(alpha, beta, n_input, tau, n_actions, gamma,
              max_size, fc1_dims, fc2_dims, batch_size)


score_history = [] 
r = []
n_episode = 5
n_step = 250

vsp_win = np.zeros([n_vsp,n_step,n_episode])
vsp_winning = np.zeros([n_vsp,n_episode])

reward_per_episode = np.zeros ([n_episode])
i_episode_matrix = np.zeros ([n_episode])

for i_episode in range(n_episode):
    i_episode_matrix[i_episode] = i_episode
    state = env.reset()
    state = state.reshape((n_input))
    done = False
    
    score = 0
    agent.noise.reset()
    
    for i_step in range(n_step):
        #------------------
        state=state/(1+max(state))
        #------------------
        action = agent.choose_action(state)
        new_state, reward, vsp_win[:,i_step,i_episode] = env.step(action)
        #------------------
        new_state = new_state.reshape((n_input))
        new_state=new_state/(1+max(new_state))
        #------------------
        agent.remember(state, action, reward, new_state, done)
        agent.learn()
        score += reward
             
        state = new_state.copy()
        
    reward_per_episode[i_episode] = score
    print('episode:',i_episode, 'reward:',reward_per_episode[i_episode])
    for i_vsp in range(n_vsp):
        vsp_winning[i_vsp,i_episode] = np.sum(vsp_win[i_vsp,:,i_episode])
        
agent.save_models()                          

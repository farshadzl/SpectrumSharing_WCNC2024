
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 13:31:08 2023

@author: Fari
"""
import numpy as np
import random
import math
import time

np.random.seed(1375)

class Environment:
    
    def __init__(self, n_vsp,delta,phi):
        
        self.n_vsp = n_vsp
        self.delta = delta
        self.phi = phi
        
        self.n_uav = 5
    
    def knapsack_alg(self, vsp_req, n_block, original):
        
        selected_mno = 0
        selected_mno = np.where(vsp_req[:, 2]!=0)[0] # number of mnos which have requested blocks
     
        values = np.zeros(self.n_vsp)
        weights = np.zeros(self.n_vsp)
        
 
        for i in range(self.n_vsp):
             values[i] = vsp_req[i, 0] # bids
             weights[i] = vsp_req[i, 2] # number of requested blocks
             
        max_weights = n_block
                
        rows = int(self.n_vsp + 1)
        cols = int(max_weights + 1)
        #_______ numpy to list conversion _____
        values = values.tolist()
        weights = weights.tolist()
        values = [0] + values
        weights = [0] + weights
        
        dp_array = [[0 for i in range(cols)] for j in range(rows)]
        
        # values
        for i in range(1, rows):
            # weights
            for j in range(1, cols):
                # if this weight exceeds max_weight at that point
                if j - weights[i] < 0:
                    dp_array[i][j] = dp_array[i - 1][j]

                # max of -> last ele taken | this ele taken + max of previous values possible
                else:
                    dp_array[i][j] = max(dp_array[i - 1][j], (values[i])+ dp_array[i - 1][j - int(weights[i])])
        
        values_chosen = []
        mno_chosen = []
        i = rows - 1
        j = cols - 1

        # Get the items to be picked
        while i > 0 and j > 0:

            # ith element is added
            if dp_array[i][j] != dp_array[i - 1][j]:
                # add the value
                values_chosen.append((values[i]))
                mno_chosen.append(i)
                # decrease the weight possible (j)
                j = j - int(weights[i])
                # go to previous row
                i = i - 1

            else:
                i = i - 1    
                
        mno_chosen = [x-1 for x in mno_chosen]

        return dp_array[rows - 1][cols - 1], values_chosen, mno_chosen


    def reset(self):
        n_block = np.random.randint(1,60) 
        min_vsp_rate = np.zeros([self.n_vsp])
        vsp_rate = np.zeros([self.n_vsp])
        vsp_req = np.zeros([self.n_vsp,3])
        min_vsp_rate[0] = 66.44 #bps
        min_vsp_rate[1] = 66.44 
        min_vsp_rate[2] = 199.32
        min_vsp_rate[3] = 66.44
        min_vsp_rate[4] = 66.44

        for i_vsp in range(self.n_vsp):
            vsp_rate[i_vsp] = np.random.randint(min_vsp_rate[i_vsp],500)
            
        for i_vsp in range(self.n_vsp):
            vsp_req[i_vsp, 0] =  self.delta[i_vsp] * vsp_rate[i_vsp]
        #________ Values __________
            vsp_req[i_vsp, 1] = self.phi[i_vsp] * vsp_rate[i_vsp]
        #________ Request for blocks __________
            vsp_req[i_vsp, 2] = np.random.randint(0,(n_block/self.n_vsp)+self.n_vsp)
        
        return vsp_req
    
    def step(self, action):
        n_block = np.random.randint(1,60)
        min_vsp_rate = np.zeros([self.n_vsp])
        vsp_rate = np.zeros([self.n_vsp])
        weights = np.zeros([self.n_vsp])
        vsp_req = np.zeros([self.n_vsp,3])
        
        for i_vsp in range(self.n_vsp):
            weights[i_vsp] = action[i_vsp]
            
        min_vsp_rate[0] = 66.44 #bps
        min_vsp_rate[1] = 66.44 
        min_vsp_rate[2] = 199.32
        min_vsp_rate[3] = 66.44
        min_vsp_rate[4] = 66.44
   
        for i_vsp in range(self.n_vsp):
            vsp_rate[i_vsp] = np.random.randint(min_vsp_rate[i_vsp],500)
        
        for i_vsp in range(self.n_vsp):
        #________ Bid __________
            vsp_req[i_vsp, 0] = weights[i_vsp] * self.delta[i_vsp] * vsp_rate[i_vsp]
        #________ Values __________
            vsp_req[i_vsp, 1] = self.phi[i_vsp] * vsp_rate[i_vsp]
        #________ Request for blocks __________
            vsp_req[i_vsp, 2] = np.random.randint(0,(n_block/self.n_vsp)+self.n_vsp)
            
        table_filled, chosen_bids, winners = self.knapsack_alg(vsp_req, n_block, original=True)
        
        vsp_win = np.zeros([self.n_vsp])  #binary varible --> equal 1 if vsp win
        for i in range(len(winners)):
            vsp_win[winners[i]] = 1
            
        utility = 0
        for i in range(self.n_vsp):
            utility += (vsp_win[i_vsp] * vsp_rate[i_vsp]) - min_vsp_rate[i_vsp]
        
                
        return vsp_req, utility, vsp_win
        
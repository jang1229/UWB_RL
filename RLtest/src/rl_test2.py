#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import rospy
from collections import deque
from std_msgs.msg import String
import re
import time
import math

class DDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(DDQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Agent:
    def __init__(self, state_dim, action_dim, hidden_dim, lr, gamma, epsilon, epsilon_decay):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=10000)
        self.timestep = 0
        self.batch_size = 32
        self.policy_net = DDQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = DDQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        self.target_update_freq = 100
	self.epsilon_min = 0.01
        self.uwb_distance = 0
        self.camera_distance = 0
        self.rssi_distance = 0
	self.start_step = 0
	self.pre_uwb_distance=0
	self.pre_rssi_distance=0
	self.pre_rssi_distance_no=0
	self.rssi_distanc_noise=0
        #rospy.init_node('distance_node', anonymous=True)
        #rospy.Subscriber('serial_data_distance', String, self.callback)


    def new_environment(self):
  
	
        try:
		self.rssi_distanc =rospy.wait_for_message('/RF_data_recoding', String)
		self.uwb_distance =rospy.wait_for_message('/serial_data_recoding', String)
	 	self.rssi_distanc_noise =rospy.wait_for_message('/RFnoise_data_recoding', String)
	except:
		print("wait")		
#	self.rssi_distance = 0

	
	if self.rssi_distanc.data == '':
    		self.rssi_distanc = self.pre_rssi_distance

	self.rssi_distanc = int(self.rssi_distanc.data)
	self.uwb_distance = int(self.uwb_distance.data)
	self.rssi_distanc_noise = 10
	self.pre_rssi_distance =self.rssi_distanc
	#self.camera_distance =self.rssi_distanc

        return (self.uwb_distance, self.rssi_distanc, self.rssi_distanc_noise)



    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_dim)
        state = torch.FloatTensor(state).to(self.device)
        q_values = self.policy_net(state)
        return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def reward_function(self, current_state, next_state,action , next_action):
	action=round(action, 5)
        action = action*0.01
     	next_action= next_action*0.01
	tx_power =22
        uwb_distance, rssi, rssi_no = current_state
        next_uwb_distance, next_rssi, next_rssi_no = next_state
  
	print("action ", action)
	#print ("rssi",rssi)
	#print ("uwb_distance",uwb_distance)

        p_signal = 10 ** (rssi / 10)  
        p_noise = 10 ** (rssi_no / 10)
	d_0 = 1.0  #
	n_factor = 10 * 2 * math.log10(1000/d_0)
	path_loss = tx_power - rssi - n_factor
    	distance = 1000000*(math.sqrt((p_signal / p_noise) / (10 ** (path_loss / 10))))
	#print ("distance",distance)
	
##
        p_signal_n = 10 ** (next_rssi / 10)  
        p_noise_n = 10 ** (next_rssi_no / 10)
	n_factor_n = 10 * 2 * math.log10(1000/d_0)
	path_loss_n = tx_power - next_rssi - n_factor_n

	distance_next = 1000000*(math.sqrt((p_signal_n / p_noise_n) / (10 ** (path_loss_n / 10))))

        distance_diff = abs( distance*action- uwb_distance)
        next_distance_diff = abs(next_rssi*next_action - next_uwb_distance)

        reward_next = 1 / (next_distance_diff + 1e-6)
        reward_per = 1 / (distance_diff + 1e-6)
  
        reward = reward_next + reward_per

	self.start_step= self.start_step+1

        if self.start_step >= 10:
            done = True
	    self.start_step= 0
        else:
            done = False

        return reward,done

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)

        q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        next_actions = torch.argmax(self.policy_net(next_state_batch), dim=1)
        next_q_values = self.target_net(next_state_batch).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        expected_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))

        loss = self.loss_fn(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        if self.timestep % 100 == 0:
            self.timestep += 1

        # Update target network
        if self.timestep % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    def reset(self):
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)

            except:
                pass



if __name__ == "__main__":
    rospy.init_node('DDQN_D', anonymous=True)
    #print("Episode: ", "Total Reward: ")
    #tgent = Agent()
    episodes = 1000
    agent = Agent(state_dim=3, action_dim=200, hidden_dim=64, lr=0.01, gamma=0.95, epsilon=1.0, epsilon_decay=0.997)
    for episode in range(episodes):
        state = agent.new_environment()
        action = agent.choose_action(state)
	action=round(action, 5)
        done = False
        total_reward = 0
	#state = env.reset()
        while not done:
            next_action = agent.choose_action(state)
            next_state = agent.new_environment()

            reward,done = agent.reward_function(state, next_state,action ,next_action)
	    #reward=round(reward, 5)
            agent.store_transition(state, next_action, reward, next_state, done)
            state = next_state
            action =next_action
            total_reward += reward
            agent.train()
	if done:
  	  print("episode:", episode, "  score:", total_reward," epsilon:",agent.epsilon)


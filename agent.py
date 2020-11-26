import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.categorical import Categorical
from collections import deque

from PIL import Image
import PIL

import numpy as np
import cv2
from skimage import transform
from skimage.color import rgb2gray  # grayscale image


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space, frames=4):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 512
        #self.fc1 = torch.nn.Linear(state_space, self.hidden)  # CNN is now first layer
        self.fc2_mean = torch.nn.Linear(self.hidden, 3)  # neural network for Q (actor) (action-value)
        self.fc3 = torch.nn.Linear(self.hidden, 1)  # neural network for V (critic) (state-value)
        self.sigma = torch.nn.Parameter(torch.tensor([10.]))  # Implement learned variance during gradient update of NN's
        self.init_weights()
        # create Convolutional Neural Network: we input 4 frames of dimension of 80x80
        self.cnn = nn.Sequential(
            nn.Conv2d(frames, 32, 8, stride=4),  # (number of layers, number of filters, kernel_size e.g. 8x8, stride)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU()
        )

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        # Common part: Convolutional Neural Network
        x = self.cnn(x)

        # Actor part
        action_mean = self.fc2_mean(x)
        sigma = self.sigma

        # Critic part
        state_val = self.fc3(x)
        # Instantiate and return a normal distribution with mean mu and std of sigma
        #action_dist = Normal(action_mean, sigma)
        action_dist = Categorical(logits=action_mean)

        # Return state value in addition to the distribution
        return action_dist, state_val


class Agent(object):
    def __init__(self, policy):
        self.train_device = "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.RMSprop(policy.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.clip_range = 0.2
        self.states = []
        self.action_probs = []
        self.rewards = []
        self.next_states = []
        self.done = []
        self.name = "BeschdePong"
        self.number_stacked_imgs = 4  # we stack up to for imgs to get information of motion
        self.img_collection = []
        #self.img_collection = [np.zeros((80,80), dtype=np.int) for i in range(self.number_stacked_imgs)]



    def update_policy(self, episode_number):
        # Convert buffers to Torch tensors
        action_probs = torch.stack(self.action_probs, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        # Clear state transition buffers
        self.states, self.action_probs, self.rewards = [], [], []
        self.next_states, self.done = [], []
        self.reset()  # reset all remaining state transition buffers

        #print("action_probs: ", action_probs)
        #print("rewards: ", rewards)
        #print("states: ", states)
        #print("next_states: ", next_states)
        #print("done: ", done)

        #print("action_probs shape: ", action_probs.shape)
        #print("rewards shape: ", rewards.shape)
        #print("states shape: ", states.shape)
        #print("next_states shape: ", next_states.shape)
        #print("done shape: ", done.shape)

        # Bring states in right order to be processed
        states = states.permute(0, 3, 1, 2)
        next_states = next_states.permute(0, 3, 1, 2)

        # Compute state values (NO NEED FOR THE DISTRIBUTION)
        action_distr, pred_value_states = self.policy.forward(states)
        nextaction_distribution, valueprediction_next_states = self.policy.forward(next_states)

        # Delete 1 dimensionality
        valueprediction_next_states = (valueprediction_next_states).squeeze(-1)
        pred_value_states = (pred_value_states).squeeze(-1)

        # Handle terminal states
        valueprediction_next_states = torch.mul(valueprediction_next_states, 1-done)

        #Critic Loss:
        critic_loss = F.mse_loss(pred_value_states, rewards+self.gamma*valueprediction_next_states.detach())
        print('target: ', rewards+self.gamma*valueprediction_next_states)
        print('estimation: ', pred_value_states)

        # Compute advantage estimates
        advantage = rewards + self.gamma * valueprediction_next_states - pred_value_states
        # Calculate actor loss (very similar to PG)
        actor_loss = (-action_probs * advantage.detach()).mean()

        # Compute the gradients of loss w.r.t. network parameters
        loss = critic_loss + actor_loss
        loss.backward()

        # Update network parameters using self.optimizer and zero gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

    def preprocessing(self, observation):
        """ Preprocess the received information: 1) Grayscaling 2) Reducing quality (resizing)
        Params:
            observation: image of pong
        """
        # Grayscaling
        img_gray = np.dot(observation, [0.2989, 0.5870, 0.1140]).astype(np.uint8)

        # Normalize pixel values
        img_norm = img_gray / 255.0

        # Downsampling: we receive squared image (e.g. 200x200) and downsample by x2.5 to (80x80)
        img_resized = cv2.resize(img_norm, dsize=(80, 80))

        return img_resized

    def stack_images(self, observation):
        """ Stack up to four frames together
        Params:
            observation: raw 200x200 image
        """
        # image preprocessing
        img_preprocessed = self.preprocessing(observation)

        if (len(self.img_collection) == 0):  # start of new episode, use len() instead of timestep to stay Markovian
            # img_collection get filled with zeros
            self.img_collection = deque([np.zeros((80,80), dtype=np.int) for i in range(4)], maxlen=4)
            # fill img_collection 4x with the first frame
            self.img_collection.append(img_preprocessed)
            self.img_collection.append(img_preprocessed)
            self.img_collection.append(img_preprocessed)
            self.img_collection.append(img_preprocessed)
            # Stack the images in img_collection
            img_stacked = np.stack(self.img_collection, axis=2)
        else:
            # Delete first/oldest entry and append new image
            self.img_collection.append(img_preprocessed)

            # Stack the images in img_collection
            img_stacked = np.stack(self.img_collection, axis=2)

        return img_stacked


    def get_action(self, img_stacked, timestep, evaluation=False):
        # create torch out from numpy array
        x = torch.from_numpy(img_stacked).float().to(self.train_device)

        #Add one more dimension, batch_size=1, for the conv2d to read it
        x = x.unsqueeze(0)

        # Change the order, so that the channels are at the beginning is expected: (1*4*80*80) = (batch size, number of channels, height, width)
        x = x.permute(0, 3, 1, 2)

        # Pass state x through the policy network
        action_distribution, __ = self.policy.forward(x)

        # Get action: Return mean if evaluation, else sample from the distribution (returned by the policy)
        if evaluation:
            action = action_distribution.mean()
        else:
            action = action_distribution.sample()

        # Calculate the log probability of the action
        act_log_prob = action_distribution.log_prob(action)

        return action, act_log_prob


    def store_outcome(self, state, next_state, action_prob, reward, done):
        # Now we need to store some more information than with PG
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)

    def load_model(self):
        """ Load already created model
        """
        #load_path = '/home/isaac/codes/autonomous_driving/highway-env/data/2020_09_03/Intersection_egoattention_dqn_ego_attention_1_22:00:25/models'
        policy.load_state_dict(torch.load("./model50000ep_WimblepongVisualSimpleAI-v0_0.mdl"))


    def get_name(self):
        """ Interface function to retrieve the agents name
        """
        return self.name

    def reset(self):
        """ Resets all memories and buffers
        """
        self.img_collection = []

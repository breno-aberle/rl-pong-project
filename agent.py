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
        #self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2_mean = torch.nn.Linear(self.hidden, 3)  # neural network for Q
        # TODO: Add another linear layer for the critic
        #Entry the hidden layers, output the value.
        self.fc3 = torch.nn.Linear(self.hidden, 1)  # neural network for V
        self.sigma = torch.nn.Parameter(torch.tensor([10.]))  # TODO: Implement learned variance (or copy from Ex5)
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
        # Common part
        #x = self.fc1(x)
        #x = F.relu(x)
        x = self.cnn(x)
        #x.squeeze(0)
        #print("forward x: ", x.shape)

        # Actor part
        action_mean = self.fc2_mean(x)
        sigma = self.sigma  # TODO: Implement (or copy from Ex5)

        # Critic part
        # TODO: Implement
        state_val = self.fc3(x)
        # TODO: Instantiate and return a normal distribution
        # with mean mu and std of sigma
        # Implement or copy from Ex5
        #action_dist = Normal(action_mean, sigma)
        action_dist = Categorical(logits=action_mean)
        # TODO: Return state value in addition to the distribution

        return action_dist , state_val


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
        #self.img_collection = [np.zeros((80,80), dtype=np.int) for i in range(self.number_stacked_imgs)]



    def update_policy(self, episode_number):
        # Convert buffers to Torch tensors
        action_probs = torch.stack(self.action_probs, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        #print('states hape',states.shape)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)
        # Clear state transition buffers
        self.states, self.action_probs, self.rewards = [], [], []
        self.next_states, self.done = [], []

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


        #print("states: ", states.shape)

        # TODO: Compute state values (NO NEED FOR THE DISTRIBUTION)
        states = states.permute(0, 3, 1, 2)
        #print(states.shape)
        next_states = next_states.permute(0,3,1,2)
        action_distr, pred_value_states = self.policy.forward(states)
        nextaction_distribution, valueprediction_next_states = self.policy.forward(next_states)  ##### COMPUTED using the forward function

        #Critic Loss:
        valueprediction_next_states = (valueprediction_next_states).squeeze(-1)
        pred_value_states = (pred_value_states).squeeze(-1)
        #print(valueprediction_next_states.shape)
        #print(done.shape)
        valueprediction_next_states = torch.mul(valueprediction_next_states, 1-done)
        #valueprediction_next_states = valueprediction_next_states*(1-done.T)
        print('target',rewards+self.gamma*valueprediction_next_states)
        print('estimation',pred_value_states)
        critic_loss = F.mse_loss(pred_value_states, rewards+self.gamma*valueprediction_next_states.detach())

        # Advantage estimates
        # TODO: Compute advantage estimates
        advantage = rewards + self.gamma * valueprediction_next_states - pred_value_states
        # TODO: Calculate actor loss (very similar to PG)
        actor_loss = (-action_probs * advantage.detach()).mean()

        # TODO: Compute the gradients of loss w.r.t. network parameters
        # Or copy from Ex5
        loss = critic_loss + actor_loss
        #print(loss)
        loss.backward()

        # TODO: Update network parameters using self.optimizer and zero gradients
        # Or copy from Ex5
        self.optimizer.step()
        self.optimizer.zero_grad()

    def preprocessing(self, observation):
        """ Preprocess the received information: 1) Grayscaling 2) Reducing quality (resizing)
        Params:
            observation: image of pong
        """
        # Grayscaling
        #img_gray = rgb2gray(observation)
        img_gray = np.dot(observation, [0.2989, 0.5870, 0.1140]).astype(np.uint8)

        # Normalize pixel values
        img_norm = img_gray / 255.0

        # Downsampling: we receive squared image (e.g. 200x200) and downsample by x2.5 to (80x80)
        img_resized = cv2.resize(img_norm, dsize=(80, 80))
        #img_resized = img_norm[::2.5,::2.5]
        return img_resized

    def stack_images(self, observation, img_collection, timestep):
        """ Stack up to four frames together
        """
        # image preprocessing
        img_preprocessed = self.preprocessing(observation)

        if (timestep == 0):  # start of new episode
            # img_collection get filled with zeros again
            img_collection =  deque([np.zeros((80,80), dtype=np.int) for i in range(4)], maxlen=4)
            # fill img_collection 4x with the first frame
            img_collection.append(img_preprocessed)
            img_collection.append(img_preprocessed)
            img_collection.append(img_preprocessed)
            img_collection.append(img_preprocessed)
            # Stack the images in img_collection
            img_stacked = np.stack(img_collection, axis=2)
        else:
            # Delete first/oldest entry and append new image
            #img_collection.pop(0)
            img_collection.append(img_preprocessed)

            # Stack the images in img_collection
            img_stacked = np.stack(img_collection, axis=2) # TODO: right axis??

        return img_stacked, img_collection


    def get_action(self, img_stacked, timestep, evaluation=False):
        # stack Image




        # create torch out from numpy array
        x = torch.from_numpy(img_stacked).float().to(self.train_device)
        #print("stacked image", x)
        #print("stacked image shape", x.shape)

        #Add one more dimension, batch_size=1, for the conv2d to read it
        x = x.unsqueeze(0)
        #print("dimension of batch added", x.shape)

        # Change the order, so that the channels are at the beginning is expected: (1*4*80*80) = (batch size, number of channels, height, width)
        x = x.permute(0, 3, 1, 2)
        #print("After permutation: ", x.shape)

        # TODO: Pass state x through the policy network
        # Or copy from Ex5
        action_distribution, __ = self.policy.forward(x)
        # TODO: Return mean if evaluation, else sample from the distribution
        # returned by the policy
        #print("action_distribution", action_distribution)

        # Or copy from Ex5
        if evaluation:
            action = action_distribution.mean()
        else:
            action = action_distribution.sample()

        # TODO: Calculate the log probability of the action
        # Or copy from Ex5
        act_log_prob = action_distribution.log_prob(action)
        #print("action from dist: ", action)
        #print(action_distribution)

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
        return:
            none
        """

    def get_name(self):
        """ Interface function to retrieve the agents name
        """
        return self.name

    def reset(self):
        """ Resets the agent’s state after an episode is finished
        return:
            none
        """
        # TODO: Reset the after one point to the middle


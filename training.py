import torch
import gym
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
from agent import Agent, Policy
#from cp_cont import CartPoleEnv  # importing cartpole environment from exercise session
from wimblepong import Wimblepong # import wimblepong environment
import pandas as pd
from PIL import Image
from collections import deque


# Policy training function
def train(env_name, print_things=True, train_run_id=0, train_episodes=5000):
    # Create a Gym environment
    env = gym.make(env_name)

    # Get dimensionalities of actions and observations
    action_space_dim = env.action_space.shape
    observation_space_dim = env.observation_space.shape

    # Instantiate agent and its policy
    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy)

    # Arrays to keep track of rewards
    reward_history, timestep_history = [], []
    average_reward_history = []
    counter=0
    # Run actual training
    for episode_number in range(train_episodes):
        reward_sum, timesteps = 0, 0
        done = False
        # Reset the environment and observe the initial state
        observation = env.reset()
        observation = np.array(observation)
        #observation = cv2.cvtColor(np.array(observation), cv2.COLOR_RGB2GRAY)
        #print(observation)
        # TODO: call first time stack_frames(observation/state) ?
        #Make an array of the dimension 200*200, and 4 elements, full of zeros.
        img_collection =  deque([np.zeros((80,80), dtype=np.int) for i in range(4)], maxlen=4)
        #We send the first one, will the full of zeros, and the initial observation which is our 'state'.
        state_images, img_collection = agent.stack_images(observation,img_collection, timestep=timesteps)
        # Loop until the episode is over
        while not done:

            # Get action from the agent, an action gets chosen based on the img_stacked processed.
            action, action_probabilities = agent.get_action(state_images, timestep=timesteps)
            #We save to previous observation, the img_stacked corresponding to the state before taking the action
            #previous_observation = img_stacked
            #State_images

            print("action: ", action)
            print("action_probabilities: ", action_probabilities)

            # Perform the action on the environment, get new state and reward
            #We do a new action
            #Now we perform a new step, to see what happens with the action in our current state, the result is the enxt state
            observation, reward, done, info = env.step(action.detach().numpy())

            # TODO: call second time stack_frames(observation/next_state) ?

            next_state_images, img_collection = agent.stack_images(observation,img_collection, timestep=timesteps)
            # TODO: BEGIN: start of if done: (no need)

            # Store action's outcome (so that the agent can improve its policy)
            #agent.store_outcome(previous_observation, observation, action_probabilities, reward, done)
            #With the other naming
            if not done:
                agent.store_outcome(state_images, next_state_images, action_probabilities, reward, done)
            # TODO: END
                state_images=next_state_images
            else:
                observation = observation*0
                # We process the images to get the proper stat rather than just the observation.
                next_state_images, img_collection = agent.stack_images(observation,img_collection, timestep=timesteps)
                agent.store_outcome(state_images, next_state_images, action_probabilities, reward, done)


            # Store total episode reward
            reward_sum += reward
            timesteps += 1
            counter += 1
            ###COMMENT FOR TASK 1(NEXT two lines)-UNCOMMENT FOR TASK 2-3
            #if counter%50==0:
            #    agent.update_policy(episode_number)

        #We have to send a image all black/all white, so that the NN knows when that happend the point has finished.
        # Either sending the function doen and getting that as an if inside the preprocessing, or something like this.


        if print_things:
            print("Episode {} finished. Total reward: {:.3g} ({} timesteps)"
                  .format(episode_number, reward_sum, timesteps))

        # Bookkeeping (mainly for generating plots)
        reward_history.append(reward_sum)
        timestep_history.append(timesteps)
        if episode_number > 100:
            avg = np.mean(reward_history[-100:])
        else:
            avg = np.mean(reward_history)
        average_reward_history.append(avg)

        # Let the agent do its magic (update the policy)
        #COMMENT FOR TASK 2-3 NEXT LINE - UNCOMMENT TASK 1
        agent.update_policy(episode_number)

    # Training is finished - plot rewards
    if print_things:
        plt.plot(reward_history)
        plt.plot(average_reward_history)
        plt.legend(["Reward", "100-episode average"])
        plt.title("AC reward history (non-episodic)")
        plt.show()
        print("Training finished.")
    data = pd.DataFrame({"episode": np.arange(len(reward_history)),
                         "train_run_id": [train_run_id]*len(reward_history),
                         # TODO: Change algorithm name for plots, if you want
                         "algorithm": ["Non-Episodic AC"]*len(reward_history),
                         "reward": reward_history})
    torch.save(agent.policy.state_dict(), "model_%s_%d.mdl" % (env_name, train_run_id))
    return data


# Function to test a trained policy
def test(env_name, episodes, params, render):
    # Create a Gym environment
    env = gym.make(env_name)

    # Get dimensionalities of actions and observations
    action_space_dim = env.action_space.shape[-1]
    observation_space_dim = env.observation_space.shape[-1]

    # Instantiate agent and its policy
    policy = Policy(observation_space_dim, action_space_dim)
    policy.load_state_dict(params)
    agent = Agent(policy)

    test_reward, test_len = 0, 0
    for ep in range(episodes):
        done = False
        observation = env.reset()
        while not done:
            # Similar to the training loop above -
            # get the action, act on the environment, save total reward
            # (evaluation=True makes the agent always return what it thinks to be
            # the best action - there is no exploration at this point)
            action, _ = agent.get_action(observation, evaluation=True)
            observation, reward, done, info = env.step(action.detach().cpu().numpy())

            if render:
                env.render()
            test_reward += reward
            test_len += 1
    print("Average test reward:", test_reward/episodes, "episode length:", test_len/episodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None, help="Model to be tested")
    parser.add_argument("--env", type=str, default="WimblepongVisualSimpleAI-v0", help="Environment to use")
    parser.add_argument("--train_episodes", type=int, default=5000, help="Number of episodes to train for")
    parser.add_argument("--render_test", action='store_true', help="Render test")
    args = parser.parse_args()

    # If no model was passed, train a policy from scratch.
    # Otherwise load the policy from the file and go directly to testing.
    if args.test is None:
        try:
            train(args.env, train_episodes=args.train_episodes)
        # Handle Ctrl+C - save model and go to tests
        except KeyboardInterrupt:
            print("Interrupted!")
    else:
        state_dict = torch.load(args.test)
        print("Testing...")
        test(args.env, 100, state_dict, args.render_test)

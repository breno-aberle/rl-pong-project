from wimblepong import Wimblepong
import random


class SimpleAi(object):
    def __init__(self, env, player_id=1):
        if type(env) is not Wimblepong:
            raise TypeError("I'm not a very smart AI. All I can play is Wimblepong.")
        self.env = env
        # Set the player id that determines on which side the ai is going to play
        self.player_id = player_id  
        # Ball prediction error, introduce noise such that SimpleAI reflects not
        # only in straight lines
        self.bpe = 4                
        self.name = "SimpleAI"
        self.timestep = 0
        self.counter = 0
        self.target = 35
    def get_name(self):
        """
        Interface function to retrieve the agents name
        """
        return self.name

    def get_action(self, ob=None):
        """
        Interface function that returns the action that the agent took based
        on the observation ob
        """
        # Get the player id from the environmen
        player = self.env.player1 if self.player_id == 1 else self.env.player2
        # Get own position in the game arena
        my_y = player.y
        # Get the ball position in the game arena
        ball_y = self.env.ball.y + (random.random()*self.bpe-self.bpe/2)
        if my_y<138 and my_y>132:
            self.timestep=0
        if self.counter%20000:
            self.target=self.target-2
        if self.target < 2:
            self.target=0


        # Compute the difference in position and try to minimize it
        y_diff = my_y - ball_y
        randomnumber=0

        randomnumber= int(self.target*random.random())
        self.counter += 1
        if abs(y_diff) < 2+randomnumber:
            action = 0  # Stay
        else:
            if y_diff > 0:
                action = self.env.MOVE_UP  # Up
            else:
                action = self.env.MOVE_DOWN  # Down
        return action

    def reset(self):
        # Nothing to done for now...
        return



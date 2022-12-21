import sys

import numpy as np
import utils

class Agent:    
    def __init__(self, actions, Ne=40, C=40, gamma=0.7):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path,self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None

    def max_q(self, state):
        max_val = -sys.maxsize
        for action in self.actions:
            # print(action)
            if max_val < self.Q[state][action]:
                max_val = self.Q[state][action]
        return max_val

    def learning_rate(self, state):
        return self.C / (self.C + self.N[state][self.a])

    def update_q(self, next_state, current_state, reward):
        lrate = self.learning_rate(current_state)
        max_val = self.max_q(next_state)
        self.Q[current_state][self.a] +=  lrate * (reward + self.gamma * max_val - self.Q[current_state][self.a])

    def choose_next_action(self, state):
        q_val = -sys.maxsize
        next_action = 0
        for action in self.actions:
            n_val = self.N[state][action]
            if self.Ne <= n_val:
                c_q_val = self.Q[state][action]
                if c_q_val > q_val:
                    q_val = c_q_val
            else:
                if q_val <= 1:
                    q_val = 1
            next_action = action
        self.N[state][next_action] += 1
        self.a = next_action
        # print(next_action)
        return next_action

    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''

        next_state = self.generate_state(environment)
        if self._train and self.a is not None and self.s is not None:
            reward = -0.1
            if dead:
                reward = -1
            elif points > self.points:
                reward = 1
            current_state = self.s
            self.update_q(next_state, current_state, reward)
        if dead:
            self.reset()
            return 0
        else:
            self.points = points
            self.s = next_state
        return self.choose_next_action(next_state)

    def generate_state(self, environment):
        # TODO: Implement this helper function that generates a state given an environment
        snake_head_x, snake_head_y, snake_body, food_x, food_y = environment
        adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right, adjoining_wall_x\
            , adjoining_wall_y = 0, 0, 0, 0, 0, 0
        print(snake_head_y)
        if snake_head_y == utils.DISPLAY_HEIGHT - 2:
            adjoining_wall_y = 2
        elif snake_head_y == 1:
            adjoining_wall_y = 1

        if snake_head_x == utils.DISPLAY_WIDTH - 2:
            adjoining_wall_x = 2
        elif snake_head_x == 1:
            adjoining_wall_x = 1

        food_dir_x, food_dir_y = 1, 1
        if snake_head_x == food_x:
            food_dir_x = 0
        elif snake_head_x < food_x:
            food_dir_x = 2
        if snake_head_y == food_y:
            food_dir_y = 0
        elif snake_head_y < food_y:
            food_dir_y = 2

        for pos_x, pos_y in snake_body:
            if pos_x == snake_head_x and 0 < pos_y < snake_head_y:
                adjoining_body_top = 1
            if pos_x == snake_head_x and snake_head_y < pos_y < utils.DISPLAY_HEIGHT - 1:
                adjoining_body_bottom = 1
            if pos_y == snake_head_y and 0 < pos_x < snake_head_x:
                adjoining_body_left = 1
            if pos_y == snake_head_y and utils.DISPLAY_WIDTH - 1 > pos_x > snake_head_x:
                adjoining_body_right = 1

        return food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom\
            , adjoining_body_left, adjoining_body_right

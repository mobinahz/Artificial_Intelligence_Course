from cube import Cube
from constants import *
from utility import *
import math

import random
import numpy as np

GRID_SIZE = 20
ACTIONS_SIZE = 4  # Left, Right, Up, Down

class Snake:
    body = []
    turns = {}
    
    def __init__(self, color, pos, file_name=None):
        # pos is given as coordinates on the grid ex (1,5)
        self.color = color
        self.head = Cube(pos, color=color)
        self.body.append(self.head)
        self.dirnx = 0
        self.dirny = 1
        self.last_direction = (self.dirnx, self.dirny) 
        self.last_action = 3
        self.state = [0] * 4
        self.move_history = []  # List to track last few moves
        
        # Q-learning parameters
        try:
            self.q_table = np.load(file_name)
        except:
            self.q_table = np.zeros((4, 4, 4, 4, ACTIONS_SIZE))
        self.lr = 0.1  # Learning rate
        self.discount_factor = 0.9  # Discount factor
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.1  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate for epsilon
        self.episode_count = 0  #Episode counter
    
    def get_optimal_policy(self, state):
        # Get Q-values for all actions from the Q-table
        q_values = self.q_table[state[0], state[1], state[2], state[3], :]
        optimal_action = np.argmax(q_values)        
        return optimal_action


    def make_action(self, state):
        chance = random.random()
        if chance < self.epsilon:
            action = random.randint(0, 3)
        else:
            action = self.get_optimal_policy(state)
        # action = np.clip(action, 0, ACTIONS_SIZE - 1)     
        return action
    
    def decay_epsilon(self, reset=False):
        if reset:
            self.epsilon = 1  # Reset epsilon to 1 when the snake eats a snack
        else:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                if self.epsilon < self.epsilon_min:
                    self.epsilon = self.epsilon_min



    def update_q_table(self, state, action, next_state, reward):
        # if self.q_table is None:
        #     return
        future_q = np.max(self.q_table[next_state[0], next_state[1], next_state[2], next_state[3], :])
        current = self.q_table[state[0], state[1], state[2], state[3], action]
        target = reward + self.discount_factor * future_q
        new_q = ((1 - self.lr) * current) + (self.lr * target)
        self.q_table[state[0], state[1], state[2], state[3], action] = new_q
        
        # self.decay_epsilon()


    def move(self, snack, other_snake):
        state = self.create_state(snack, other_snake)
        self.state = state
        action = self.make_action(state)
        self.last_action = action
        self.episode_count += 1
        if (self.dirnx == 1 and action == 2) or (self.dirnx == -1 and action == 3) or\
            (self.dirny == 1 and action == 0) or (self.dirny == -1 and action == 1):
                action = (action + 2) % 4

        if action == 0:  # Left
            self.dirnx = -1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 1:  # Right
            self.dirnx = 1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 2:  # Up
            self.dirny = -1
            self.dirnx = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 3:  # Down
            self.dirny = 1
            self.dirnx = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
         
        self.last_direction = (self.dirnx, self.dirny)  # Update last direction
        self.move_history.append(self.last_direction)  # Add to move history
        
        if len(self.move_history) > 3:  # Keep only the last 3 moves
            self.move_history.pop(0)

        for i, c in enumerate(self.body):
            p = c.pos[:]
            if p in self.turns:
                turn = self.turns[p]
                c.move(turn[0], turn[1])
                if i == len(self.body) - 1:
                    self.turns.pop(p)
            else:
                c.move(c.dirnx, c.dirny)

        next_state = self.create_state(snack, other_snake)    
        if ((self.episode_count % 350) == 0):
            self.decay_epsilon()    
        return state, next_state, action


    def check_out_of_board(self):
        headPos = self.head.pos
        if headPos[0] >= ROWS - 1 or headPos[0] < 1 or headPos[1] >= ROWS - 1 or headPos[1] < 1:
            self.reset((random.randint(3, 18), random.randint(3, 18)))
            return True
        return False

    
    def calc_reward(self, snack, other_snake):
        reward = 0
        win_self, win_other = False, False
        snake_x, snake_y = self.head.pos
        snack_x, snack_y = snack.pos
        
        if self.check_out_of_board():
            # TODO: Punish the snake for getting out of the board
            reward -= 120
            win_other = True
            reset(self, other_snake)
        
        if self.head.pos == snack.pos:
            self.addCube()
            snack = Cube(randomSnack(ROWS, self), color=(0, 255, 0))
            # self.decay_epsilon(reset=True)
            # TODO: Reward the snake for eating
            reward += 200
         
        if (self.state[0] == self.last_action):
            reward += 150
            
        if (self.state[0] == 0 and self.last_action == 1) or (self.state[0] == 1 and self.last_action == 0) or\
            (self.state[0] == 2 and self.last_action == 3) or (self.state[0] == 3 and self.last_action == 2):
                reward -= 150
           
        # if (abs(snake_x - snack_x) < 2) | (abs(snake_y - snack_y) < 2):
        #     reward += 90
        # elif (abs(snake_x - snack_x) < 3) | (abs(snake_y - snack_y) < 3):
        #     reward += 80
        # elif (abs(snake_x - snack_x) < 4) | (abs(snake_y - snack_y) < 4):
        #     reward += 70
        # elif (abs(snake_x - snack_x) < 5) | (abs(snake_y - snack_y) < 5):
        #     reward += 60
        
        # elif (abs(snake_x - snack_x) < 10) | (abs(snake_y - snack_y) < 10):
        #     reward += 50
          
        if self.head.pos in list(map(lambda z: z.pos, self.body[1:])):
            # TODO: Punish the snake for hitting itself
            reward -= 100
            win_other = True
            reset(self, other_snake)
            
            
        if self.head.pos in list(map(lambda z: z.pos, other_snake.body)):
            
            if self.head.pos != other_snake.head.pos:
                # TODO: Punish the snake for hitting the other snake
                reward -= 100
                win_other = True
            else:
                if len(self.body) > len(other_snake.body):
                    # TODO: Reward the snake for hitting the head of the other snake and being longer
                    reward += 100
                    win_self = True
                elif len(self.body) == len(other_snake.body):
                    # TODO: No winner
                    reward += 0
                else:
                    # TODO: Punish the snake for hitting the head of the other snake and being shorter
                    reward -= 100
                    win_other = True
                    
            reset(self, other_snake)
            # Check for noise move (oscillating direction)
        if len(self.move_history) >= 3:
            if self.move_history[-1] == (-self.move_history[-2][0], -self.move_history[-2][1]) and \
               self.move_history[-3] == (-self.move_history[-2][0], -self.move_history[-2][1]):
                reward = -50  # Penalty for noise move
            
        return snack, reward, win_self, win_other


    def reset(self, pos):
        self.head = Cube(pos, color=self.color)
        self.body = []
        self.body.append(self.head)
        self.turns = {}
        self.dirnx = 0
        self.dirny = 1

    def addCube(self):
        tail = self.body[-1]
        dx, dy = tail.dirnx, tail.dirny

        if dx == 1 and dy == 0:
            self.body.append(Cube((tail.pos[0] - 1, tail.pos[1]), color=self.color))
        elif dx == -1 and dy == 0:
            self.body.append(Cube((tail.pos[0] + 1, tail.pos[1]), color=self.color))
        elif dx == 0 and dy == 1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] - 1), color=self.color))
        elif dx == 0 and dy == -1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] + 1), color=self.color))

        self.body[-1].dirnx = dx
        self.body[-1].dirny = dy

    def draw(self, surface):
        for i, c in enumerate(self.body):
            if i == 0:
                c.draw(surface, True)
            else:
                c.draw(surface)

    def save_q_table(self, file_name):
        np.save(file_name, self.q_table)
        
    def create_state(self, snack, other_snake):
        snake_head_x, snake_head_y = self.head.pos
        snack_x, snack_y = snack.pos
        other_snake_head_x, other_snake_head_y = other_snake.head.pos

        state = [0] * 4  # Initialize state with three elements

        # Encode relative position of the snack
        if snack_y < snake_head_y:
            state[0] = 2  # Snack is North
        elif snack_y > snake_head_y:
            state[0] = 3  # Snack is South
        elif snack_x > snake_head_x:
            state[0] = 1  # Snack is East
        elif snack_x < snake_head_x:
            state[0] = 0  # Snack is West

        # Encode relative position of the other snake's head
        if other_snake_head_y < snake_head_y:
            state[1] = 2  # Other snake is North
        elif other_snake_head_y > snake_head_y:
            state[1] = 3  # Other snake is South
        elif other_snake_head_x > snake_head_x:
            state[1] = 1  # Other snake is East
        elif other_snake_head_x < snake_head_x:
            state[1] = 0  # Other snake is West

        # Encode immediate collision danger
        # danger_north = self.check_collision((snake_head_x, snake_head_y - 1))
        # danger_south = self.check_collision((snake_head_x, snake_head_y + 1))
        # danger_east = self.check_collision((snake_head_x + 1, snake_head_y))
        # danger_west = self.check_collision((snake_head_x - 1, snake_head_y))
        
        # if danger_north:
        #     state[2] = 2  # Danger North
        # elif danger_south:
        #     state[2] = 3  # Danger South
        # elif danger_east:
        #     state[2] = 1  # Danger East
        # elif danger_west:
        #     state[2] = 0  # Danger West
            
        # Add last direction to the state
        if self.last_direction == (0, -1):
            state.append(0)  # Last move was North
        elif self.last_direction == (0, 1):
            state.append(1)  # Last move was South
        elif self.last_direction == (1, 0):
            state.append(2)  # Last move was East
        elif self.last_direction == (-1, 0):
            state.append(3)  # Last move was West 
            
        snack_x_distance = abs(snake_head_x - snack_x)
        snack_y_distance = abs(snake_head_y - snack_y)
        snack_distance = math.sqrt(snack_x_distance ** 2 + snack_y_distance ** 2)
            
        if (snack_distance < 2):
            state[3] = 0  
        elif (snack_distance < 3):
            state[3] = 1
        elif (snack_distance < 4):
            state[3] = 2
        else:
            state[3] = 3
 
        
        return state

        
    def get_state(self, snack, other_snake):
        snake_head_x, snake_head_y = self.head.pos
        snack_x, snack_y = snack.pos
        other_snake_head_x, other_snake_head_y = other_snake.head.pos

        state = 0b0000000000000000

        snack_x_distance = abs(snake_head_x - snack_x)
        snack_y_distance = abs(snake_head_y - snack_y)
        snack_distance = math.sqrt(snack_x_distance ** 2 + snack_y_distance ** 2)
            
        if (snack_x_distance < 2):
            state |= 1 << 0  
        elif (snack_distance < 3):
            state |= 1 << 1  
        elif (snack_distance < 4):
            state |= 1 << 2  
        else:
            state |= 1 << 3  

        # Encode relative position of the snack
        if snack_y < snake_head_y:
            state |= 1 << 4  # Snack is North
        elif snack_y > snake_head_y:
            state |= 1 << 5  # Snack is South
        elif snack_x > snake_head_x:
            state |= 1 << 6  # Snack is East
        elif snack_x < snake_head_x:
            state |= 1 << 7  # Snack is West

        # Encode relative position of the other snake's head
        if other_snake_head_y < snake_head_y:
            state |= 1 << 8  # Other snake is North
        elif other_snake_head_y > snake_head_y:
            state |= 1 << 9  # Other snake is South
        elif other_snake_head_x > snake_head_x:
            state |= 1 << 10  # Other snake is East
        elif other_snake_head_x < snake_head_x:
            state |= 1 << 11  # Other snake is West

        # Encode immediate collision danger
        if (self.check_collision((snake_head_x, snake_head_y - 2))):
            state |= 1 << 12  # Danger North
        elif (self.check_collision((snake_head_x, snake_head_y + 2))):
            state |= 1 << 13  # Danger South
        elif (self.check_collision((snake_head_x + 2, snake_head_y))):
            state |= 1 << 14  # Danger East
        elif (self.check_collision((snake_head_x - 2, snake_head_y))):
            state |= 1 << 15  # Danger West

        return state
    
    
    def check_collision(self, pos):
        if (pos[0]) >= GRID_SIZE or (pos[0]) < 0 or (pos[1]) >= GRID_SIZE or (pos[1]) < 0:
            return True
        return False
    
    def check_body_collision(self, pos):
        return pos in list(map(lambda z: z.pos, self.body))
        

  
  



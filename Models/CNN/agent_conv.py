import torch
import random
import copy
import numpy as np
from collections import deque
from game_for_agent_conv import SnakeGameAI, Direction, Point
from model_conv import Linear_QNet, QTrainer
import matplotlib.pyplot as plt
from IPython import display
import math

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
SIZE=3
# NE KREÁLJ MINDIG ÚJ MONDELLT.

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        
        
        self.model = Linear_QNet(SIZE,3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        # Sima
        head = game.snake[0]
        game_map= np.zeros((int(game.h/20)+1,int(game.w/20)+1), np.int16)
        
        game_map[int(head.x/20)][int(head.y/20)] = 3
        for i in range(13):
            game_map[i][0] = 2
            game_map[i][12] = 2            
            game_map[0][i] = 2
            game_map[12][i] = 2
            
        for i in game.snake[1:]:
            game_map[int(i.x/20)][int(i.y/20)] = 1
            
        game_map[int(game.food.x/20)][int(game.food.y/20)] = 4
        
        
        game_map_list = game_map.tolist()
        
        state = [[game_map_list]]
        return state

    def remember(self, state, action, reward, next_state, done):

        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached
        
  
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory
        
        #states, actions, rewards, next_states, dones = zip(*mini_sample)
        #self.trainer.train_step(states, actions, rewards, next_states, dones)
        print(len(mini_sample))
        for state, action, reward, next_state, done in mini_sample:
            self.trainer.train_step(state, action, reward, next_state, done)

        
    def train_short_memory(self, state, action, reward, next_state, done):

        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 300 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def plot(scores, mean_scores, save):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)
    if save==True:
        plt.savefig('./model/uj_CNN_model_kep.png')

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    last_few_states=[]
    elso_500=True
    save=False
    while elso_500==True:
        # get old state
        state_old = agent.get_state(game)
#        print(torch.tensor(state_old).shape)

        old_distance_x= abs(int(game.snake[0].x/20) - int(game.food.x/20))
        old_distance_y= abs(int(game.snake[0].y/20) - int(game.food.y/20))
#        print("state_old_x: ",old_distance_x)
#        print("state_old_y: ",old_distance_y)
        
        
        
        #
        if len(last_few_states)==1:
            if len(last_few_states[0]) == 3:
                last_few_states[0].append(state_old[0][0])
#        print(torch.tensor(last_few_states).shape)
#        print(len(last_few_states))
        
        #
        def convert(x):
            g=[]
            if len(torch.tensor(x).shape)==5:
                for i in x:
                    g.append(i[0][0])
            return [g]
        if len(last_few_states)==0:
            while len(last_few_states) < (SIZE):
                last_few_states.append(state_old)
            last_few_states=convert(last_few_states)
#        print(torch.tensor(last_few_states).shape)
        
        
        #
        if len(last_few_states)==1:
            if len(last_few_states[0]) == (SIZE+1):
                last_few_states[0].pop(0)
#        print(torch.tensor(last_few_states).shape)
#        print('last_few:',"\n",last_few_states) 
         
        
        # get move
        final_move = agent.get_action(last_few_states)
#        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game) 
        
        new_distance_x= abs(int(game.snake[0].x/20) - int(game.food.x/20))
        new_distance_y= abs(int(game.snake[0].y/20) - int(game.food.y/20))
#        print("state_new_x: ",new_distance_x)
#        print("state_new_y: ",new_distance_y)
        
        if reward==0:
            if old_distance_x > new_distance_x or old_distance_y > new_distance_y:
                reward=1
            if old_distance_x < new_distance_x or old_distance_y < new_distance_y:
                reward=-1
            
        
        #
        last_new_states = copy.deepcopy(last_few_states)
        last_new_states[0].append(state_new[0][0])
        last_new_states[0].pop(0)
#        print('last_new:',"\n",last_new_states)
             

        # train short memory
        agent.train_short_memory(last_few_states, final_move, reward, last_new_states, done)
#        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(last_few_states, final_move, reward, last_new_states, done)
#        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            #print(reward)
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
#            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
            
            print('Game', agent.n_games, 'Score', score, 'Record:', record)#, 'Reward:', reward)
            last_few_states=[]
            last_new_states=[]
            
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            if agent.n_games == 500:
                save=True
                plot(plot_scores, plot_mean_scores, save)
                elso_500=False
            else:
                plot(plot_scores, plot_mean_scores, save)


if __name__ == '__main__':
    train()
# importing libraries
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# architecture
class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        #constructor
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        # connector between output and 1st hidden layer
        # connect all input layers to all hiden layers
        # 1st argument = number of input layers
        # 2nd argument = how many hidden neurons we want to have
        # you can always create more hidden layers or use more or less hidden neurons
        # @TODO try and test it with different values
        self.fc1 = nn.Linear(input_size, 200)
        self.fc2 = nn.Linear(200,150)
        # connector between hidden layer and output layer
        self.fc3 = nn.Linear(150, nb_action)
        
    def forward(self, state):
        # activates first set of neurons (input to 1st hidden layer)
        x = F.relu(self.fc1(state))
        y = F.relu(self.fc2(x))
        # takes the Q values from the 1st hidden layer and stores them
        q_values = self.fc3(y)
        return q_values

# Experience Replay class implementation
class ReplayMemory(object):
    
    def __init__(self, capacity):
        # constructor 
        # we will implement capacity times memoryes
        self.capacity = capacity
        self.memory = []
        
    def push(self, event):
        #adding events to the memory function
        self.memory.append(event)
        #check if the memory is less or equal then the capacity
        if len(self.memory) > self.capacity:
            #delete the first memory entered
            del self.memory[0]
            
    def sample(self, batch_size):
        # we will take random sample sizes and use them
        # *random.sample(...) - takes a random sample from the memory, having a batch size
        # zip(...) - reshapes the list
        # example :
        # f = ( (1,2,3), (4,5,6) ) ; zip(f) = ( (1,4), (2,5), (3,6) )
        sample = zip(*random.sample(self.memory, batch_size))

        # cannot just simply return the sample
        # we need to 'translate' the variables to a pytorch variable
        # Variable function will return a list of pytorch variables
        # and we will concat them just to allign them
        # each torch variable will contain a tensor and a gradient

        return map(lambda x: Variable(torch.cat(x, 0)), sample)
    
# Deep Q Learning Implementation
class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        # constructor
        # the gamma value of the Q learning formula
        self.gamma = gamma
        # evolution of the last 1000 rewards
        self.reward_window = []
        # creating the neural network
        self.model = Network(input_size, nb_action)
        # creating the memory
        self.memory = ReplayMemory(100000)
        # creating the optimmizer
        # using the Adam algorithm
        # 1st parameter - the model's parameters
        # 2nd parameter - the learning rate
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.002)
        # defining last_state
        # it has to be in a batch
        # we will create a Tensor vector, having input_size + 1 dimensions
        # the input_size + 1 dimension is the "fake" dimension
        # *.unsqueeze(...) is the function that creates the 1st vector
        # it has as parameter the position that we want the vector to be in
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        # defining last_action
        # initialize with 0, first action from the vector
        self.last_action = 0
        # defining last_reward
        # initialize with 0, to not 'scare' the AI at start xD
        self.last_reward = 0
        
    def select_action(self, state):
        # function that selects the right move
        # we will use SoftMax
        # we want the torch tensor state into a torch variable
        # we don't want the gradiants, so we add volatile = True to not include them
        # we will include the gradiants associated to this state of all the computations !!
        # save memory + efficiency
        with torch.no_grad():
            probs = F.softmax(self.model(Variable(state)) * 90, dim=1) 
        # Temperature  making the AI more sure of itself (motivational speaking ?)
        # If the Temperature is 0, the AI is disabled
        # softmax([1,2,3]) = [0.04, 0.11, 0.85] => softmax([1,2,3] * 3(temperature paremeter)) = [0, 0.02, 0.98]
        #   values               q values            values        temperature_variable             q values
        # get a random draw based on q values (probabilities)
        action = probs.multinomial(1)
        # multinomial returns with the fake dimension/batch and the values that interests us, is in the data[0,0]
        return action.data[0, 0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        # *.gather(1, batch_action) -  we only want the action that was chosen
        # batch_state - has fake dimension 
        # batch_action - does not have it
        # we add *.unsqueeze(1) to have the same input
        # than we kill it, with *.squeeze(1) because we are in the output vector
        # and we only want a simple vector
        # we get the prediction
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        # we get the next_output
        # we detach all options of all states that we have
        # we take the maximum with respect to the action
        # we have to specify that it is with respect with the action : specify index max(1)
        # we are taking the index[0] (represents the states) . so we get the maximum of q values of the next states from index 1
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        # we get the target
        target = self.gamma * next_outputs + batch_reward
        # we will compute the loss
        # temporal difference loss
        # torch.nn.functional.smooth_l1_loss - best lost function for Q learning ?
        td_loss = F.smooth_l1_loss(outputs,target)
        # we must re-initialize the optimizer at each iteration of the loop
        self.optimizer.zero_grad()
        # backward propagation
        # retain_variables = True to free some memory
        td_loss.backward(retain_graph = True)
        # update weights
        self.optimizer.step()
        
    def update(self, reward, new_signal):
        # convert the new_signal vector to a Tensor vector and make them all float
        # don't forget to add the fake dimension on the 1st position
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        # update the memory with this new action
        # all elements must be a Tensor type
        # last_state was converted to Tensor type on line 92
        # new_state was converted to Tensor type on line 147
        # last_action can only have the values 0, 1 or 2, so we convert it to a LongTensor and we make sure that it is an int
        # last_rewards is a float number, so we just simply convert it to a Tensor type
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        # update/defining next action
        # it is like, we got to a point T in time, and now we have to tell the AI, what it should do, or more precise, let it decide
        action = self.select_action(new_state)
        # start learning when we have reached more than 100 events to make it viable and efficient
        # 1st memory is from the DQN class
        # 2nd memory is from the ReplayMemory, the vector where we store all our actions
        if len(self.memory.memory) > 100:
            # it will learn from the 100 memories of the transition
            # same parameters as in the learn function, we obviously need them (not the same, their local ones and we send them as parameters)
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        # update the last_action
        self.last_action = action
        # update the last_state
        self.last_state = new_state
        # update the last_reward
        self.last_reward = reward
        # update the log of rewards
        self.reward_window.append(reward)
        # check to not have a reward window that too much data, 1000 logs is fine
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        # compute the mean of all reward window
        # we added +1 to avoid the case where len is 0 
        return sum(self.reward_window) / (len(self.reward_window) + 1)
    
    def save(self):
        # save the brain for when you close the application
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    }, 'last_brain.pth')
    
    def load(self):
        # load the model from the saved file
        # check if the file exists
        if os.path.isfile('last_brain.pth'):
            print("=> loading the brain...")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            print("done!")
        else:
            print("no last brain found :( ...")
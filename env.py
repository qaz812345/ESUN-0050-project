import copy
import math
import numpy as np

import gym
from gym import spaces, logger
from gym.utils import seeding
from gym.envs.registration import register

# Code reference: https://github.com/selimamrouni/Deep-Portfolio-Management-Reinforcement-Learning/blob/master/environment.py

class TradeEnv():

    """
    This class is the trading environment (render) of our project.

    The trading agent calls the class by giving an action at the time t.
    Then the render gives back the new portfolio at the next step (time t+1).

    #parameters:

    - windonw_length: this is the number of time slots looked in the past to build the input tensor
    - portfolio_value: this is the initial value of the portfolio
    - trading_cost: this is the cost (in % of the traded stocks) the agent will pay to execute the action
    - interest_rate: this is the rate of interest (in % of the money the agent has) the agent will:
        -get at each step if he has a positive amount of money
        -pay if he has a negative amount of money
    -train_size: % of data taken for the training of the agent - please note the training data are taken with respect
    of the time span (train -> | time T | -> test)
    """

    def __init__(self, data , window_length=50, n_batch=None,
                 portfolio_value= 10000, trading_cost=[0.1425/100, 0.4425/100], interest_rate=0.02/250, best=False):

        self.data = data

        # set parameters
        if n_batch is None:
            n_batch = 1
        self.n_batch = n_batch
        self.portfolio_value = [portfolio_value] * n_batch
        self.window_length = window_length
        self.trading_cost = trading_cost
        self.interest_rate = interest_rate

        # number of stocks and features
        self.nb_stocks = self.data.shape[1]
        self.nb_features = self.data.shape[0]
        self.end_train = self.data.shape[2]

        # init state and index
        self.index = None
        self.state = None
        self.done = False # not used

        # for best mode
        self.best = best
        self.actions = list()
        for i in range(self.nb_stocks+1):
            # Ecch action is form as an one hot vector with shape (11,)
            action = np.array([0]*(i) + [1] + [0]*(self.nb_stocks-i))
            self.actions.append(action)

        #init seed
        self.seed()

    def return_pf(self):
        """
        return the value of the portfolio
        """
        return self.portfolio_value

    def readTensor(self,X,t):
        ## this is not the tensor of equation 18
        ## need to batch normalize if you want this one
        return X[ : , :, t-self.window_length:t ]

    def readUpdate(self, t):
        """
        return the return of each stock for the day t
        """
        return np.array([1 + self.interest_rate] + self.data[-1, :, t].tolist())

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, w_init, p_init, t=0):

        """
        This function restarts the environment with given initial weights and given value of portfolio

        """
        # batch mode
        if self.n_batch > 1:
            data_list = list()
            w_list = list()
            p_list = list()
            for i in range(self.n_batch):
                data_list.append(self.readTensor(self.data, t[i]))
                w_list.append(w_init)
                p_list.append(p_init)

            self.state = (np.array(data_list), np.array(w_list), np.array(p_list))
            self.index = t.copy()
        # inference mode
        else:
            self.state = (self.readTensor(self.data, t) , w_init , p_init)
            self.index = t

        self.done = False

        return self.state, self.done

    def step(self, action):
        """
        This function is the main part of the render.
        At each step t, the trading agent gives as input the action he wants to do.
        So, he gives the new value of the weights of the portfolio.

        The function computes the new value of the portfolio at the step (t+1),
        it returns also the reward associated with the action the agent took.
        The reward is defined as the evolution of the the value of the portfolio in %.

        """
        index = self.index
        state = self.state
        done = self.done
        # batch mode
        if self.n_batch > 1:
            data_list = list()
            w_list = list()
            p_list = list()
            r_list = list()
            for i in range(self.n_batch):

                # get Xt from data
                # beginning of the day (previous state)
                w_previous = state[1][i]
                #print('w_previous shape: {}'.format(w_previous.shape))
                pf_previous = state[2][i]
                #print('pf_previous shape: {}'.format(pf_previous.shape))

                # the update vector is the vector of the opening price of the day divided by 
                # the opening price of the previous day
                update_vector = self.readUpdate(index[i])
                #print('update_vector shape: {}'.format(update_vector.shape))

                # allocation choice
                w_alloc = action[i]
                pf_previous = state[2][i]
                pf_alloc = pf_previous

                # compute transaction cost
                w_change = (w_alloc[1:]-w_previous[1:])
                w_buy = w_change[w_change > 0]
                w_sell = w_change[w_change < 0]
                buy_cost = pf_alloc * np.linalg.norm(w_buy,ord = 1)* self.trading_cost[0]
                sell_cost = pf_alloc * np.linalg.norm(w_sell,ord = 1)* self.trading_cost[1]
                
                # convert weight vector into value vector
                v_alloc = pf_alloc*w_alloc

                # pay transaction costs
                pf_trans = pf_alloc - buy_cost - sell_cost
                
                v_trans = pf_trans * w_alloc

                # market prices evolution
                # compute new value vector
                v_evol = v_trans * update_vector

                # compute new portfolio value
                pf_evol = np.sum(v_evol)

                # compute weight vector
                w_evol = v_evol/pf_evol

                # compute instanteanous reward
                reward = (pf_evol-pf_previous)/pf_previous
                r_list.append(update_vector)

                # update index
                index[i] = index[i] + 1

                data_list.append(self.readTensor(self.data, index[i]))
                w_list.append(w_evol)
                p_list.append(pf_evol)

                if index[i] >= self.end_train:
                    done = True

            #compute state
            state = (np.array(data_list), np.array(w_list), np.array(p_list))
            reward = np.array(r_list)

        # inference mode
        else:
            # beginning of the day (previous state)
            w_previous = state[1]
            pf_previous = state[2]
            #print('w_previous: ', w_previous)
            #print('pf_previous = pf_alloc: ', pf_previous)

            # the update vector is the vector of the opening price of the day divided by 
            # the opening price of the previous day
            update_vector = self.readUpdate(index)
            #print('update_vector: ', np.round(update_vector, 3))

            # best mode
            if self.best:
                max = np.max(update_vector[1:])
                if max >= 1.0:
                    max_stock = np.argmax(update_vector[1:])
                    action = self.actions[max_stock+1]
                else:
                    # money
                    action = self.actions[0]
                #print('Best action: ', action)

            w_alloc = action
            pf_alloc = pf_previous
            w_change = (w_alloc[1:]-w_previous[1:])
            w_buy = w_change[w_change > 0]
            w_sell = w_change[w_change < 0]
            #print('w_alloc: ', w_alloc)

            # compute transaction cost
            buy_cost = pf_alloc * np.linalg.norm(w_buy,ord = 1)* self.trading_cost[0]
            sell_cost = pf_alloc * np.linalg.norm(w_sell,ord = 1)* self.trading_cost[1]
            #print('cost: ', cost)

            # convert weight vector into value vector
            v_alloc = pf_alloc*w_alloc
            #print('v_alloc: ', v_alloc)

            # pay transaction costs
            pf_trans = pf_alloc - buy_cost - sell_cost
            #print('pf_trans: ', pf_trans)

            cost = buy_cost + sell_cost
            if cost <= v_alloc[0]:
                v_trans = v_alloc - np.array([cost] + [0]*self.nb_stocks)
            else:
                v_trans = pf_trans * w_alloc
        
            # market prices evolution
            # compute new value vector
            v_evol = v_trans*update_vector
            #print('v_evol: ', v_evol)

            # compute new portfolio value
            pf_evol = np.sum(v_evol)
            #print('pf_evol: ', pf_evol)

            # compute weight vector
            w_evol = v_evol/pf_evol
            #print('w_evol: ', w_evol)

            # compute instanteanous reward
            #reward = (pf_evol-pf_previous)/pf_previous
            reward = update_vector

            # update index
            index = index+1

            # compute state
            state = (self.readTensor(self.data, index), w_evol, pf_evol)

            if index >= self.end_train:
                done = True

        self.state = state
        self.index = index
        self.done = done

        return state, reward, done

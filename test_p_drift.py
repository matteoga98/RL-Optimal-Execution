from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math as m
import statistics
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm 
transition = namedtuple("transition", ("state", "action", "next_state", "reward"))

state = namedtuple("state", "inventory, time,qdr_var,price")
action = namedtuple("action", "amount_sold")

in_features=5

class DQN(nn.Module):
    def __init__(self, in_size, hidden_layers_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_size, hidden_layers_size)
        self.fc2 = nn.Linear(hidden_layers_size, hidden_layers_size)
        self.fc3 = nn.Linear(hidden_layers_size, hidden_layers_size)
        self.fc4 = nn.Linear(hidden_layers_size, hidden_layers_size)
        self.fc5 = nn.Linear(hidden_layers_size, 1)

    def forward(self, x):

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))

        return self.fc5(x)


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def add(self, state, action, next_state, reward):
        self.memory.append(transition(state, action, next_state, reward))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# in_sampled_features is deque of namedtuple('transition',('state', 'action', 'next_state', 'reward'))
# batch='transition',('state =(state_1, state_2 .... ) ', 'action=(action_1, action_2 ....)', 'next_state', 'reward'))


class Agent:
    def __init__(self, inventory, batch_size=15):

        self.main_net = DQN(in_size=in_features, hidden_layers_size=30)
        self.target_net = DQN(in_size=in_features, hidden_layers_size=30)
        
        for p in self.target_net.parameters():
            p.requires_grad=False

        self.initial_capital = inventory

        self._update_target_net()

        self.learning_rate = 0.001
        self.optimizer = optim.Adam(
            params=self.main_net.parameters(), lr=self.learning_rate
        )
        self.AC_risk_aversion=0.000001
        self.gamma = 0.99

        self.epsilon = 1
        self.delta = 0.99
        self.a_penalty = 10

        self.number_trading_decisions = 5
        self.lots_size=100

        self.starting_trading_time = 15 * 60
        self.finishing_trading_time = 75 * 60
        
        self.time_subdivisions = m.floor(
            (self.finishing_trading_time - self.starting_trading_time) / self.number_trading_decisions)
        self.time_bonus=m.floor(self.time_subdivisions/2)
        
        self.last_time=self.finishing_trading_time+self.time_bonus

        self.batch_size = batch_size
        self.memory = ReplayMemory(50000)

        self.timestep = 0
        self.update_target_steps = 15
        
        self.drift=-0.00000001
       

#get tuple of qdr_var between every trading decisions for a data set
    def pre_train(self, data):

        qdr_var=[]
        M = self.time_subdivisions
        T_0 = self.starting_trading_time 
        
        price=[]
        
        for trading_decision in range(self.number_trading_decisions):
            price_temp=[]
            for t in range(T_0 + M * trading_decision , T_0+ M*(trading_decision+1)):
            
    	        price_temp.append(data[t]-data[0])
    	        
            price.append(statistics.mean(price_temp))

        for trading_decision in range(self.number_trading_decisions):
            qdr_var_temp=0
            for t in range(T_0 + M * (trading_decision) , T_0+ M*(trading_decision +1)):

                qdr_var_temp += (data[t +1] - data[t])**2
                
            qdr_var.append(qdr_var_temp)

        return qdr_var, price
        
    def assign_qdr_values(self,qdr_var_min, qdr_var_max):

        self.qdr_var_min = qdr_var_min
        self.qdr_var_max = qdr_var_max
        
    def assign_price_values(self, price_min, price_max):
    
        self.price_min = price_min
        self.price_max = price_max
        
    #to use this method, one has to initialize min and max with the method assign_qdr_values and assign_price_values in the pre_training part.
    
    def qdr_var_normalize(self,qdr_var):
        
        middle_point=(self.qdr_var_max+self.qdr_var_min)/2
        half_length=(self.qdr_var_max-self.qdr_var_min)/2

        qdr_var = (qdr_var - middle_point)/half_length

        return qdr_var
    
    def price_normalize(self, price):

        middle_point=(self.price_max+self.price_min)/2
        half_length=(self.price_max-self.price_min)/2

        price = (price - middle_point)/half_length
        
        return price
        
    def time_transform(self, t):
        tc = (self.number_trading_decisions-1) / 2
        return (t - tc) / tc

    def inventory_action_transform(self, q, x):

        q_0 = self.initial_capital +1

        q = q / q_0 - 1
        x = x / q_0
        r = m.sqrt(q**2 + x**2)
        theta = m.atan((-x / q))
        z = -x / q

        if theta <= m.pi / 4:
            r_tilde = r * m.sqrt((pow(z, 2) + 1) * (2 * (m.cos(m.pi / 4 - theta)) ** 2))
        else:
            r_tilde = r * m.sqrt(
                (pow(z, -2) + 1) * (2 * (m.cos(theta - m.pi / 4)) ** 2)
            )
        return 2 * (-r_tilde * m.cos(theta)) + 1, 2 * (r_tilde * m.sin(theta)) - 1
    

        
    def normalization(self,q,t,qdr_var,price,x):
    
        q,x = self.inventory_action_transform (q,x)
        t = self.time_transform(t)
        qdr_var=self.qdr_var_normalize(qdr_var)
        price=self.price_normalize(price)
        
        return q,t,qdr_var,price,x

    def PeL_QL(self, selling_strategy, data):
    
        PeL = 0
        a = self.a_penalty 
        M = self.time_subdivisions
        T_0=self.starting_trading_time

        for i in range(self.number_trading_decisions):

            x = selling_strategy[i].action.amount_sold*self.lots_size
            xs = x / M

            for t in range(M):
                if T_0+ i * M + t < len(data):
                    PeL += xs * data[T_0 + i * M + t] - a * (xs**2)
                    
        return PeL

    def PeL_TWAP(self, data):
    
        PeL = 0
        M = self.time_subdivisions
        a = self.a_penalty 
        T_0 = self.starting_trading_time

        x = self.initial_capital / self.number_trading_decisions *  self.lots_size
        xs = x / M

        for i in range(self.number_trading_decisions):

            for t in range(M):
                if T_0+ i * M + t < len(data):
                    PeL+= xs * data[T_0 + i * M + t] - a * (xs**2)
        return PeL
        
    def PeL_AC_with_drift(self,data):
        
        PeL = 0
        M = self.time_subdivisions
        penalty = self.a_penalty
        T_f = self.number_trading_decisions*M
        T_0 = self.starting_trading_time
        q_0=self.initial_capital

        mu=self.drift
        gamma=self.AC_risk_aversion
        V=100
        eta=V*self.a_penalty                                                      
        sigma=0.2
                
        b=V*(pow(sigma,2))*gamma/(2*eta)
        a=-(V*mu)/(2*eta)

        q_now=q_0
        for i in range(self.number_trading_decisions):
            next_t=(i+1)*M
            q_future =  ((q_0+(a/b))*(m.sinh(m.sqrt(b)*(T_f-next_t)))+(a/b)*m.sinh(m.sqrt(b)*next_t))/m.sinh(m.sqrt(b)*T_f)-(a/b)
            xs=self.lots_size*(q_now-q_future)/M 
                
            for t in range(M):
                if T_0 + i * M + t < len(data):

                    PeL += xs* data[T_0 + i * M + t] - penalty* (xs**2)
                    
            q_now=q_future 
        return PeL   
                   
    def PeL_AC_limit_with_drift(self,data):
        
        PeL = 0
        M = self.time_subdivisions
        penalty = self.a_penalty
        T_f = self.number_trading_decisions*M
        T_0 = self.starting_trading_time
        q_0=self.initial_capital

        mu=self.drift
        gamma=self.AC_risk_aversion
        V=100
        eta=V*self.a_penalty                                                      
        sigma=0.2
     
        q_now=q_0
        for k in range(self.number_trading_decisions):
            next_t=(k+1)*M
            q_future=((T_f-next_t)*(4*eta*q_0+mu*V*next_t*T_f)/(4*eta*T_f))
            xs=self.lots_size*(q_now-q_future)/M
            for t in range(M):
                if T_0 + k * M + t < len(data):
                    PeL += xs* data[T_0 + k * M + t] - penalty* (xs**2)
            q_now=q_future

        return PeL                      
        

    #calculate qdr_var between state.time and state.time +1
    
    def calculate_qdr_var(self, current_state, data):
    	
    	M = self.time_subdivisions
    	T_0 = self.starting_trading_time 
    	qdr_var=0
    	
    	for t in range(T_0 + M * (current_state.time) , T_0+ M*(current_state.time+1)):
    		qdr_var += (data[t +1] - data[t])**2 
 
    	return qdr_var
    
    def calculate_price(self, current_state, data):
        
        M = self.time_subdivisions
        T_0 = self.starting_trading_time 
        price=[]

        for t in range(T_0 + M * (current_state.time) , T_0+ M*(current_state.time+1)):
            price.append(data[t]-data[0])

        return statistics.mean(price)
        
    def _update_target_net(self):
        self.target_net.load_state_dict(self.main_net.state_dict())
        
    def choose_best_action(self,state):
        
        with torch.no_grad():
            
            t=self.time_transform(state.time)
            features=[]
            for x in range(state.inventory+1):
                q,t,qdr_var,price,x=self.normalization(state.inventory,state.time,state.qdr_var,state.price,x)

                features.append(torch.tensor(
                        [
                            q,
                            t,
                            qdr_var,
                            price,
                            x
                        ],
                        dtype=torch.float
                    )
                )
            qs_value=self.main_net.forward(torch.stack(features))
            
            return action(torch.argmax(qs_value).item())

    	
    def choose_action(self, state):
        #print(self.epsilon)
        q = state.inventory
        t = state.time
            
        if q==0:
        
            return action(0)

        if self.epsilon > random.uniform(0,1):
            rand_act=np.random.binomial(q, 1 / (self.number_trading_decisions - t))
            
            return action(rand_act)
            
        x=self.choose_best_action(state)
        
        return x
        
    def calculate_reward(self, state, action, data_set):
        # data_set = np.array
        reward = 0
        M = self.time_subdivisions
        t = state.time
        q = state.inventory
        x = action.amount_sold
        a = self.a_penalty
        xs = x / M  # amount sold each second
        T_0 = self.starting_trading_time
        
        '''
        for i in range(M):
            if T_0+M*state.time+i< len(data_set):
                reward+=xs*data_set[T_0+M*state.time+i] - a*(xs**2)
        return reward
        '''
        inventory_left=q
        for i in range(T_0 + M*t, T_0 + M*(t+1)):
        
            if i + 1 < len(data_set):
                reward += inventory_left * (data_set[i + 1]- data_set[i]) - a * (xs**2)
                inventory_left-=xs
                     
        return reward
    	
    def evaluate_Q(self,current_state,current_action,type_input,net='main_net'):
 
        if type_input == 'scalar':
            #print(current_state.inventory, current_state.time, current_action.amount_sold)
            q,t,qdr_var,price,x=self.normalization(current_state.inventory,
                                            current_state.time,
                                            current_state.qdr_var,
                                            current_state.price,
                                            current_action.amount_sold
                                            )

            in_features=torch.tensor(
                    [
                        q,
                        t,
                        qdr_var,
                        price,
                        x
                    ],
                    dtype=torch.float
                )
                
        elif type_input == 'tensor':
            in_features=[]
            for (current_state, current_action) in zip(current_state,current_action):
                q,t,qdr_var,price,x= self.normalization(current_state.inventory, 
                                                current_state.time, 
                                                current_state.qdr_var,
                                                current_state.price,
                                                current_action.amount_sold
                                                )          
            
                in_features.append(torch.tensor(
                    [
            			q,
            			t,
            			qdr_var,
            			price,
            			x,
            		],
            		dtype=torch.float
            	    )
            	)      	
            
            in_features=torch.stack(in_features)
            in_features.type(torch.float64)
            
        if net == 'main_net':
            return self.main_net(in_features).type(torch.float64)
        elif net == 'target_net':
            return self.target_net(in_features).type(torch.float64)
        

    def step(self, current_state, data):

        self.timestep += 1

        x = self.choose_action(current_state)
        
        reward = self.calculate_reward(current_state, x, data)

        
        next_state = state(
            current_state.inventory - (x.amount_sold), 
            current_state.time + 1 ,
            self.calculate_qdr_var(current_state,data),
            self.calculate_price(current_state,data)
        )
        


        self.memory.add(current_state, x, next_state, reward)

        if len(self.memory) < self.batch_size:
            return 1, 0, reward, next_state, self.epsilon
        else:
            transitions = self.memory.sample(self.batch_size)
        
        return *self.train(transitions,data), reward, next_state, self.epsilon
        
            
    def train(self, in_sampled_features, data):

        batch = transition(*zip(*in_sampled_features))
        batch_current_state = batch.state
        batch_action=batch.action
        batch_next_state = batch.next_state
        batch_reward = torch.t(torch.tensor(batch.reward).unsqueeze(0))

        
        # calculate Q(S,A) -> tensor (batch_size,1)

        current=self.evaluate_Q(
                        batch_current_state,
                        batch_action,
                        'tensor',
                        'main_net'
                     )
                     
        #calculate target r+gamma*Q(nextS,bestA)     
        target=[]

        for (current_state, current_action, next_state, reward) in zip(
                                                    
                                                    batch_current_state,
                                                    batch_action, 
                                                    batch_next_state, 
                                                    batch_reward
                                                    ):
                
            #it T=4, modify Q function to force sell everything
            if next_state.time == self.number_trading_decisions -1 :
            
                q=current_state.inventory
                x=current_action.amount_sold
                future_next_price=data[self.last_time]
                next_state_price=data[self.finishing_trading_time]
                a=self.a_penalty
                
                best_future_action=self.choose_best_action(next_state)
                correction_term=(q-x)*(future_next_price-next_state_price)-a*((q-x)**2)
                
                target_value=reward+self.gamma*correction_term 
                target.append(target_value)
                
            elif next_state.time == self.number_trading_decisions :
                
                target_value=reward
                target.append(target_value)

            else:
            
                best_future_action=self.choose_best_action(next_state)                    
                target_value=reward+self.gamma*self.evaluate_Q(
                                                            next_state,
                                                            best_future_action,
                                                            'scalar',
                                                            'target_net')
                target.append(target_value)
                
        #calculate gradient norm   
        total_norm = 0
        for p in self.main_net.parameters():
            
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            
            grad_norm= total_norm ** 0.5  
          
        target=torch.stack(target)
        loss=F.mse_loss(target, current)
        
        #optimize 
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_net.parameters(), 1)
        self.optimizer.step()

        if not self.timestep % self.update_target_steps:
            self._update_target_net()
            self.epsilon = self.epsilon * self.delta
            

        return loss.cpu().item(), grad_norm

    def test_RL(self, current_state, data_set):
        
        if current_state.time==4:
            x= action(current_state.inventory)
        else:
            x = self.choose_action(current_state)
        
        reward = self.calculate_reward(current_state, x, data_set)

        qdr_var = self.calculate_qdr_var(current_state, data_set)
        price=self.calculate_price(current_state,data_set)

        next_state = state(
            (current_state.inventory - x.amount_sold),
            current_state.time + 1,
            qdr_var,
            price,
        )


        return (current_state, x, next_state, reward)
    
    def test_TWAP(self, current_state, data_set):
        q_0=self.initial_capital
        T=self.number_trading_decisions
        x=action(q_0/T)
        
        reward = self.calculate_reward(current_state, x, data_set)

        qdr_var = self.calculate_qdr_var(current_state, data_set)
        price=self.calculate_price(current_state,data_set)

        next_state = state(
            (current_state.inventory - x.amount_sold),
            current_state.time + 1,
            qdr_var,
            price,
        )
        return (current_state, x, next_state, reward)
        
    def test_AC(self, current_state, data_set):
        q_0=self.initial_capital
        M=self.time_subdivisions
        T_f = self.number_trading_decisions*M

        mu=self.drift
        gamma=self.AC_risk_aversion
        V=100
        eta=V*self.a_penalty
        sigma=0.2
                
        b=V*(pow(sigma,2))*gamma/(2*eta)
        a=-(V*mu)/(2*eta)
        t=(current_state.time)*M
        next_t=(current_state.time +1)*M
        
        q_future= ((q_0+(a/b))*(m.sinh(m.sqrt(b)*(T_f-next_t)))+(a/b)*m.sinh(m.sqrt(b)*(next_t)))/m.sinh(m.sqrt(b)*T_f)-(a/b)
        q_present =  ((q_0+(a/b))*(m.sinh(m.sqrt(b)*(T_f-t)))+(a/b)*m.sinh(m.sqrt(b)*t))/m.sinh(m.sqrt(b)*T_f)-(a/b)
        x = action(q_present-q_future)

 
        reward = self.calculate_reward(current_state, x, data_set)

        qdr_var = self.calculate_qdr_var(current_state, data_set)
        price=self.calculate_price(current_state,data_set)
        
        next_state = state(
            (current_state.inventory - x.amount_sold),
            current_state.time + 1,
            qdr_var,
            price,
        )
        return (current_state, x, next_state, reward)
        
    def test_AC_limit(self,current_state,data_set):
        q_0=self.initial_capital
        M=self.time_subdivisions
        T_f = self.number_trading_decisions*M

        mu=self.drift
        gamma=self.AC_risk_aversion
        V=100
        eta=V*self.a_penalty
        sigma=0.2
                
        b=V*(pow(sigma,2))*gamma/(2*eta)
        a=-(V*mu)/(2*eta)
        
        t=(current_state.time)*M
        next_t=(current_state.time +1)*M
        q_present=((T_f-t)*(4*eta*q_0+mu*V*t*T_f)/(4*eta*T_f))
        q_future=((T_f-next_t)*(4*eta*q_0+mu*V*next_t*T_f)/(4*eta*T_f))
        
        x = action(q_present-q_future)
        
        reward = self.calculate_reward(current_state, x, data_set)

        qdr_var = self.calculate_qdr_var(current_state, data_set)
        price=self.calculate_price(current_state,data_set)
        
        next_state = state(
            (current_state.inventory - x.amount_sold),
            current_state.time + 1,
            qdr_var,
            price,
        )
        return (current_state, x, next_state, reward)
    

##########################################################################



def performance(statistics):
    conta = 0
    for element in statistics:
        if element > 0:
            conta += 1
    return conta / len(statistics) * 100


def mean(statistics):
    mean = 0
    for element in statistics:
        mean += element
    return mean / len(statistics)
    
#calculate qdr_var before starting trading
def pre_calculate_var(agent, data):

    M = agent.time_subdivisions
    T_0 = agent.starting_trading_time 
    qdr_var_temp=0
    
    for t in range(T_0):

        qdr_var_temp +=(data[t +1] - data[t])**2

    return qdr_var_temp
    
def pre_calculate_price(agent,data):

    M = agent.time_subdivisions
    T_0 = agent.starting_trading_time 
    price=[]

    for t in range(T_0):

        price.append(data[t]-data[0])

    return statistics.mean(price)
    

#inizialize max and min for qdr_var and return the list of all qdr_var used
def pre_train(agent,dict_pre_training):
    
    qdr_var=[]
    price=[]
    
    for stock in tqdm(dict_pre_training.keys()):
        for i in tqdm(range(dict_pre_training[stock])):
        
            data = np.load("./data_drift_down_low/" + stock + "/p_" + str(i) + ".npy")
            qdr_var_episode,price_episode=agent.pre_train(data)
            qdr_var.extend(qdr_var_episode)
            price.extend(price_episode)
    
    
    qdr_var_mean=statistics.mean(qdr_var)
    qdr_var_stdev=statistics.stdev(qdr_var)
    
    price_mean=statistics.mean(price)
    price_stdev=statistics.stdev(price)
    
    qdr_var_max=min(max(qdr_var),qdr_var_mean + 2*qdr_var_stdev)
    qdr_var_min=max(min(qdr_var),qdr_var_mean - 2*qdr_var_stdev)
    price_min=max(min(price),price_mean - 2 *price_stdev)
    price_max=min(max(price),price_mean + 2* price_stdev)
    
    agent.assign_qdr_values(qdr_var_min, qdr_var_max)
    agent.assign_price_values(price_min,price_max)
    
    return qdr_var,price
    
def train(agent):

    inventory = agent.initial_capital

    stock_dict = {"train": 6000}
    stock_dict_pre_training={"train": 6000}
        
    reward_history=[]
    epsilon_history=[]
    grad_norm_history=[]
    loss_history=[]
    
    qdr_var_history, price_history = pre_train(agent,stock_dict_pre_training)
 
    for t in tqdm(range(2)):
        for stock in tqdm(stock_dict.keys()):
            for i in tqdm(range(stock_dict[stock])):

                data = np.load("./data_drift_down_low/" + stock + "/p_" + str(i) + ".npy")

                # starting with (q=inventory, T=0)

                current_state = state(inventory, 0, pre_calculate_var(agent,data), pre_calculate_price(agent,data))
                
                reward_episode=0
                for t in range(agent.number_trading_decisions):

                    (loss, grad_norm, reward, next_state,epsilon) = agent.step(current_state, data)
                    
                    current_state = next_state
                    reward_episode+= reward 
                
                loss_history.append(m.log(loss))    
                epsilon_history.append(epsilon)    
                reward_history.append(reward_episode)
                grad_norm_history.append(grad_norm)

    return reward_history,epsilon_history, grad_norm_history, loss_history, qdr_var_history, price_history
    
    
def test(agent):

    inventory = agent.initial_capital

    stock_dict = {"test": 1000}
    # for every stock, list with performance (x) and mean (x) with x_i= P&L_QL_i - P&L_TWAP_i / P&L_TWAP_i i being one trade set
    PeL_TWAP=[]
    PeL_AC=[]
    PeL_AC_limit=[]
    PeL_QL=[]
    performance_list_TWAP = []
    mean_list_TWAP = []
    performance_list_AC = []
    mean_list_AC = []
    performance_list_AC_limit = []
    mean_list_AC_limit = []

    for stock in tqdm(stock_dict.keys()):
    
        transaction_cost_balance_TWAP= []
        transaction_cost_balance_AC= []
        transaction_cost_balance_AC_limit= []
        
        learned_strategy_RL= {'0': 0, '1' : 0 , '2': 0, '3':0, '4':0}
        learned_strategy_TWAP = {'0': 0, '1' : 0 , '2': 0, '3':0, '4':0}
        learned_strategy_AC = {'0': 0, '1' : 0 , '2': 0, '3':0, '4':0}
        learned_strategy_AC_limit = {'0': 0, '1' : 0 , '2': 0, '3':0, '4':0}
        
        for i in tqdm(range(stock_dict[stock])):

            data = np.load("./data_drift_up_down/" + stock + "/p_" + str(i) + ".npy")

            selling_strategy_RL = deque()
            selling_strategy_TWAP = deque()
            selling_strategy_AC = deque()
            selling_strategy_AC_limit = deque()
            
#simulate RL strategy for the episode

            current_state = state(inventory, 0, pre_calculate_var(agent,data), pre_calculate_price(agent,data))
            for t in range(agent.number_trading_decisions):

                (current_state, action, next_state, reward) = agent.test_RL(
                    current_state, data
                )

                selling_strategy_RL.append(
                    transition(current_state, action, next_state, reward)
                )

                learned_strategy_RL[str(t)] = learned_strategy_RL[str(t)] * i / (i+1) + action.amount_sold /(i+1)
                
                current_state=next_state
#simulate TWAP strategy for the episode 
            current_state = state(inventory, 0, pre_calculate_var(agent,data), pre_calculate_price(agent,data))
            for t in range(agent.number_trading_decisions):

                (current_state, action, next_state, reward) = agent.test_TWAP(
                    current_state, data
                )

                selling_strategy_TWAP.append(
                    transition(current_state, action, next_state, reward)
                )

                learned_strategy_TWAP[str(t)] = learned_strategy_TWAP[str(t)] * i / (i+1) + action.amount_sold /(i+1)
                
                current_state=next_state
#simulate AC strategy for the episode   
            current_state = state(inventory, 0, pre_calculate_var(agent,data), pre_calculate_price(agent,data))
            
            for t in range(agent.number_trading_decisions):

                (current_state, action, next_state, reward) = agent.test_AC(
                    current_state, data
                )

                selling_strategy_AC.append(
                    transition(current_state, action, next_state, reward)
                )

                learned_strategy_AC[str(t)] = learned_strategy_AC[str(t)] * i / (i+1) + action.amount_sold /(i+1)
                
                current_state=next_state
            
            current_state = state(inventory, 0, pre_calculate_var(agent,data), pre_calculate_price(agent,data))
            
            for t in range(agent.number_trading_decisions):

                (current_state, action, next_state, reward) = agent.test_AC_limit(
                    current_state, data
                )

                selling_strategy_AC_limit.append(
                    transition(current_state, action, next_state, reward)
                )

                learned_strategy_AC_limit[str(t)] = learned_strategy_AC_limit[str(t)] * i / (i+1) + action.amount_sold /(i+1)
                
                current_state=next_state
                
            PeL_TWAP.append(agent.PeL_TWAP(data))
            PeL_AC.append(agent.PeL_AC_with_drift(data))
            PeL_AC_limit.append(agent.PeL_AC_limit_with_drift(data))
            PeL_QL.append(agent.PeL_QL(selling_strategy_RL,data))
            
            transaction_cost_balance_TWAP.append(
                (agent.PeL_QL(selling_strategy_RL, data) - agent.PeL_TWAP(data))
                / (agent.PeL_TWAP(data))
            )

            transaction_cost_balance_AC.append(
                (agent.PeL_QL(selling_strategy_RL, data) - agent.PeL_AC_with_drift(data))
                / (agent.PeL_AC_with_drift(data))
            )
            
            transaction_cost_balance_AC_limit.append(
                (agent.PeL_QL(selling_strategy_RL, data) - agent.PeL_AC_limit_with_drift(data))
                / (agent.PeL_AC_limit_with_drift(data))
            )
#test AC strategy 
        '''
        plt.bar(*zip(*learned_strategy_RL.items()))
        plt.show()  
        plt.bar(*zip(*learned_strategy_TWAP.items()))
        plt.show()  
        plt.bar(*zip(*learned_strategy_AC.items()))
        plt.title('AC strategy: risk aversion = ' + str(agent.AC_risk_aversion))
        plt.show()  
        plt.bar(*zip(*learned_strategy_AC_limit.items()))
        plt.title('AC strategy limit gamma ->0')
        plt.show()  
        '''
        mean_list_TWAP.append(100*statistics.mean(transaction_cost_balance_TWAP))
        performance_list_TWAP.append(performance(transaction_cost_balance_TWAP))

        mean_list_AC.append(100*statistics.mean(transaction_cost_balance_AC))
        performance_list_AC.append(performance(transaction_cost_balance_AC))

        mean_list_AC_limit.append(100*statistics.mean(transaction_cost_balance_AC_limit))
        performance_list_AC_limit.append(performance(transaction_cost_balance_AC_limit))
            
        PeL_TWAP=statistics.mean(PeL_TWAP)
        PeL_AC=statistics.mean(PeL_AC)
        PeL_AC_limit=statistics.mean(PeL_AC_limit)
        PeL_QL=statistics.mean(PeL_QL)
        '''
        plt.hist(transaction_cost_balance_TWAP,100)
        plt.title('PeL TWAP ' + stock )
        mean_cost_TWAP = statistics.mean(transaction_cost_balance_TWAP)
        plt.axvline(x = mean_cost_TWAP,color='r')
        plt.show()
        
        plt.hist(transaction_cost_balance_AC,100)
        plt.title('PeL AC ' + stock + '. Risk aversion = ' + str(agent.AC_risk_aversion))
        mean_cost_AC = statistics.mean(transaction_cost_balance_AC)
        plt.axvline(x = mean_cost_AC, color='r')
        plt.show()
        
        plt.hist(transaction_cost_balance_AC_limit,100)
        plt.title('PeL AC limit')
        mean_cost_AC_limit = statistics.mean(transaction_cost_balance_AC_limit)
        plt.axvline(x = mean_cost_AC_limit, color='r')
        plt.show()

    #print performances
    t=0
    print('How much in percentage we outperform TWAP')
    for stock in stock_dict.keys():
        print ('Stock: ' + stock +': ' + str(mean_list_TWAP[t])+'%')
        t+=1
    t=0
    print('\n')
    print('How many times in percentage we outperform TWAP')
    for stock in stock_dict.keys():
        print ('Stock: ' + stock +': '+ str(performance_list_TWAP[t])+'%')
        t+=1 

    print('\n')
    
    t=0
    print('How much in percentage we outperform AC')
    print('risk aversion gamma: ', agent.AC_risk_aversion)
    for stock in stock_dict.keys():
        print ('Stock: ' + stock +': ' + str(mean_list_AC[t])+'%')
        t+=1
    t=0
    print('\n')
    print('How many times in percentage we outperform AC')
    print('risk aversion gamma: ', agent.AC_risk_aversion)
    for stock in stock_dict.keys():
        print ('Stock: ' + stock +': '+ str(performance_list_AC[t])+'%')
        t+=1 
    
    t=0
    print('How much in percentage we outperform AC_limit')

    for stock in stock_dict.keys():
        print ('Stock: ' + stock +': ' + str(mean_list_AC_limit[t])+'%')
        t+=1
    t=0
    print('\n')
    print('How many times in percentage we outperform AC_limit')
    for stock in stock_dict.keys():
        print ('Stock: ' + stock +': '+ str(performance_list_AC_limit[t])+'%')
        t+=1 
   ''' 
    
        
        
        
    return mean_list_TWAP, mean_list_AC, mean_list_AC_limit, PeL_TWAP, PeL_AC, PeL_AC_limit,PeL_QL, learned_strategy_RL
    


    
def get_heatmap(agent,qdr_var,price):

    def choose_best_action(q,t,qdr_var,price):   
        define_state=state(q,t,qdr_var,price)
        return agent.choose_best_action(define_state)
    
    array=np.zeros((agent.initial_capital+1,agent.number_trading_decisions))
    
    for q in range(agent.initial_capital+1):
        for t in range(agent.number_trading_decisions):
            action=choose_best_action(q,t,qdr_var,price)
            x=action.amount_sold
            array[q][t]=x
    
    return array


def main(agent):

    #train the agent
    reward_history,epsilon_history, grad_norm_history, loss_history, qdr_var_history, price_history = train(agent)
    mean_list_TWAP,mean_list_AC,mean_list_AC_limit,PeL_TWAP, PeL_AC, PeL_AC_limit,PeL_QL,learned_strategy=test(agent)
    #make plots
    #fig, (ax1, ax2) = plt.subplots(1, 2)
    
    low_var= 0
    medium_var= 50
    high_var=100
    
    low_price=-20
    medium_price=0
    high_price=20
    '''
    sns.heatmap(get_heatmap(agent,low_var,low_price))
    plt.title('low var, low price')
    plt.show()

    sns.heatmap(get_heatmap(agent,low_var,medium_price))
    plt.title('low var, medium price')
    plt.show()
    
    sns.heatmap(get_heatmap(agent,low_var,high_price))
    plt.title('low var, high price')
    plt.show()
    
    sns.heatmap(get_heatmap(agent,medium_var,low_price))
    plt.title('medium var, low price')
    plt.show()
    
    sns.heatmap(get_heatmap(agent,medium_var,medium_price))
    plt.title('medium var, medium price')
    plt.show()
    
    sns.heatmap(get_heatmap(agent,medium_var, high_price))
    plt.title('medium var, high price')
    plt.show()
    
    sns.heatmap(get_heatmap(agent,high_var,low_price))
    plt.title('high var, low price')
    plt.show()
    
    sns.heatmap(get_heatmap(agent,high_price,medium_price))
    plt.title('high var, medium price')
    plt.show()
    
    sns.heatmap(get_heatmap(agent,high_var,high_price))
    plt.title('high var, high price')
    plt.show()
    
    plt.hist(qdr_var_history, 100)
    plt.title('qdr_var')
    plt.show()    
    
    plt.hist(price_history,100)
    plt.title('price')
    plt.show()
    
    price_normalized=[]
    for x in price_history:
        price_normalized.append(agent.price_normalize(x))
        
    var_normalized=[]
    for x in qdr_var_history:
        var_normalized.append(agent.qdr_var_normalize(x))
    
    plt.hist(price_normalized)
    plt.title('price normalized')
    plt.show()
    
    plt.hist(var_normalized)
    plt.title('var_normalized')
    plt.show()
    '''
    plt.plot(reward_history)
    series = pd.Series(np.array(reward_history)).rolling(window=100)
    plt.title('reward')
    plt.plot(series.mean())
    plt.show()
    '''
    plt.plot(epsilon_history)
    plt.title('epsilon')
    plt.show()

    plt.plot(grad_norm_history)
    plt.title('grad norm')
    plt.show()

    plt.plot(loss_history)
    plt.title('loss')
    plt.show()
    '''

    return mean_list_TWAP,mean_list_AC,mean_list_AC_limit, PeL_TWAP, PeL_AC, PeL_AC_limit,PeL_QL, learned_strategy
    
def grafici(agent):
    T_f=3600
    time=np.linspace(0,T_f,50)
    values_AC=[]
    values_AC_limit=[]
    q_0=agent.initial_capital
    q_now_AC=agent.initial_capital
    q_now_AC_limit=agent.initial_capital
    M=agent.time_subdivisions
    
    mu=agent.drift
    gamma=agent.AC_risk_aversion
    V=100
    eta=V*agent.a_penalty
    sigma=0.2
            
    b=V*(pow(sigma,2))*gamma/(2*eta)
    a=-(V*mu)/(2*eta)
    
    for t in time:
        values_AC.append(((q_0+(a/b))*(m.sinh(m.sqrt(b)*(T_f-t)))+(a/b)*m.sinh(m.sqrt(b)*t))/m.sinh(m.sqrt(b)*T_f)-(a/b))
        values_AC_limit.append((T_f-t)*(4*eta*q_0+mu*V*t*T_f)/(4*eta*T_f))
    plt.plot(time,values_AC)
    plt.plot(time,values_AC_limit)
    plt.title('AC strategies')
    plt.legend(['AC with gamma = ' + str(agent.AC_risk_aversion), 'AC limit'])
    plt.show()
    
    x_AC=[]
    x_AC_limit=[]
    for t in range(agent.number_trading_decisions):
        next_t=(t+1)*M

        q_future_AC=((q_0+(a/b))*(m.sinh(m.sqrt(b)*(T_f-next_t)))+(a/b)*m.sinh(m.sqrt(b)*next_t))/m.sinh(m.sqrt(b)*T_f)-(a/b)
        q_future_AC_limit=((T_f-next_t)*(4*eta*q_0+mu*V*next_t*T_f)/(4*eta*T_f))

        x_AC.append(q_now_AC-q_future_AC)
        x_AC_limit.append(q_now_AC_limit-q_future_AC_limit)
        q_now_AC=q_future_AC
        q_now_AC_limit=q_future_AC_limit
    print(x_AC,x_AC_limit)
    plt.bar([0,1,2,3,4],x_AC)
    plt.title('AC strategy with gamma = ' +str(agent.AC_risk_aversion))
    plt.show()
    plt.bar([0,1,2,3,4],x_AC_limit)
    plt.title('AC limit strategy for gamma ->0')
    plt.show()
        
if __name__ == "__main__":
    learned_strategy_average={'0': 0, '1' : 0 , '2': 0, '3':0, '4':0}
    mean_list_TWAP=[]
    mean_list_AC=[]
    mean_list_AC_limit=[]
    iterations=4
    for i in range(iterations):
        agent = Agent(inventory=20, batch_size=15)
        x,y,z, PeL_TWAP, PeL_AC, PeL_AC_limit, PeL_QL, learned_strategy=main(agent)
        print(x,y,z,PeL_TWAP, PeL_AC, PeL_AC_limit,PeL_QL)
        mean_list_TWAP.append(*x)
        mean_list_AC.append(*y)
        mean_list_AC_limit.append(*z)
        for t in range(5):
            learned_strategy_average[str(t)] = learned_strategy_average[str(t)] * i / (i+1) + learned_strategy[str(t)] /(i+1)

    grafici(agent)    
    plt.bar(*zip(*learned_strategy_average.items()))
    plt.show()  

    iterations=2*iterations
    mean_list_TWAP_stdev=statistics.stdev(mean_list_TWAP)
    mean_list_AC_stdev=statistics.stdev(mean_list_AC)
    mean_list_AC_limit_stdev=statistics.stdev(mean_list_AC_limit)
    
    error_mean_list_TWAP=mean_list_TWAP_stdev/m.sqrt(iterations)
    error_mean_list_AC=mean_list_AC_stdev/m.sqrt(iterations)
    error_mean_list_AC_limit=mean_list_AC_limit_stdev/m.sqrt(iterations)
    
    print('average PeL vs TWAP =', statistics.mean(mean_list_TWAP))
    print('Standard deviation= ', mean_list_TWAP_stdev)
    print('Error= ', error_mean_list_TWAP)
    print('\n')
    print('average PeL vs AC, (gamma= ' +str(agent.AC_risk_aversion) +') =', statistics.mean(mean_list_AC))
    print('Standard deviation= ', mean_list_AC_stdev)
    print('Error= ', error_mean_list_AC)
    print('\n')
    print('average PeL vs ac_limit =', statistics.mean(mean_list_AC_limit))
    print('Standard deviation= ', mean_list_AC_limit_stdev)
    print('Error= ', error_mean_list_AC_limit)
    print('\n')
    plt.hist(mean_list_TWAP, iterations)
    plt.title('Histogram PeL RL vs TWAP')
    plt.axvline(x=statistics.mean(mean_list_TWAP), color='r')
    plt.show()   
    
    plt.hist(mean_list_AC, iterations)
    plt.title('Histogram PeL RL vs AC, gamma= '+ str( agent.AC_risk_aversion))
    plt.axvline(x=statistics.mean(mean_list_AC), color='r')
    plt.show()   

    plt.hist(mean_list_AC_limit, iterations)
    plt.title('Histogram PeL RL vs AC limit')
    plt.axvline(x=statistics.mean(mean_list_AC_limit), color='r')
    plt.show()   
    




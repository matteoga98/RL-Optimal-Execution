import numpy as np
import matplotlib.pyplot as plt
from numpy import save
from tqdm import tqdm
import random 

class Brownian():
	
	def __init__(self,x0=0):
		
		self.x0=float(x0)

		
	def gen_random_walk(self,n_step=100):
		
		w=np.ones(n_step)*self.x0
		
		for i in range(1,n_step):
			yi=np.random.choice([1,-1])
			
			w[i]=w[i-1]+(yi) + self.drift
		
		return w
	def gen_normal(self,n_step=100,var=1,drift=0):

		w = np.ones(n_step)*self.x0

		for i in range(1,n_step):

			yi = np.random.normal(0,var)
			w[i] = w[i-1]+ yi + drift

		return w

#creates 1000 brownian in the folder ./data/dataX

def save_file (name_file,var,drift):	

	w=Brownian(150).gen_normal(7200,var,drift)
	data=np.array(w)
	save('./data_drift_up_down/test/' + name_file ,data)


if __name__=='__main__':

    for i in tqdm(range(1000)):
        var=0.04
        #drift=-0.00000001
        drift=random.choice([-0.00000001,0.00000001])
        save_file('p_' + str(i),var,drift)


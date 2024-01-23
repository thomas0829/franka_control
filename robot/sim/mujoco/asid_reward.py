import copy
import numpy as np

class ASIDRewardWrapper():
	
	def __init__(self,envs,delta=0.1, normalization=20., articulation=False):
		self.envs = envs
		self.delta = delta
		self.normalization = normalization
		self.max_reward = 1000
		self.articulation = articulation

	def get_reward(self,full_state,action,params=None,verbose=False):
		# compute gradient
		grad = self.estimate_loss_grad(full_state,action,params=params,verbose=verbose)
		# compute reward
		reward = np.trace(grad.T @ grad)
		# clip reward
		return np.clip(reward,0.,self.max_reward)

	def estimate_loss_grad(self,full_state,action,params=None,verbose=False):
		
		# get parameters
		if params is None:
			current_param = self.envs.get_parameters()
		else:
			current_param = params
		#current_param = [np.random.uniform(-1+self.delta,1-self.delta,size=1)]
		if self.articulation:
			current_param = [0] # set articulated, otherwise grad is 0
		params = copy.deepcopy(current_param)
		num_params = len(current_param)
		
		# compute gradient w/ finite-differences
		grad = None
		for i in range(num_params):
			params_temp = copy.deepcopy(params)
			
			# define +delta
			params_temp[i] += self.delta
			# apply theta_delta
			self.envs.set_parameters(params_temp)
			x = copy.deepcopy(full_state)
			# set theta state
			self.envs.set_full_state(x)
			# step environment
			obs1, _, _, _ = self.envs.step(action)
			if verbose:
				print(obs1,full_state)

			params_temp = copy.deepcopy(params)
			# define -delta
			params_temp[i] -= self.delta
			# apply theta_delta
			self.envs.set_parameters(params_temp)
			# set theta state
			self.envs.set_full_state(copy.deepcopy(full_state))
			# step environment
			obs2, _, _, _ = self.envs.step(action)
			
			# compute gradient
			g = (obs1 - obs2) / (2 * self.delta)
			if verbose:
				print(f'grad obs1 {obs1} obs2 {obs2} diff {obs1-obs2} grad {g}')
			
			if grad is None:
				d = len(g)
				grad = np.zeros((d,num_params))
			grad[:,i] = g

			# reset theta
			self.envs.set_parameters(current_param)
		
		# normalize gradient
		return grad / self.normalization
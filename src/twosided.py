"""
Two sided marketplaces: 
Rochet Tirole's canonical model,
* without fixed costs/benefits. 
* logistic specification for participation.
* tanh() externality instead of linear (saturation, not critical mass).
"""

import torch

"""

## Start with a simple version of firm's problem (profit maximization given demand, monopoly version)
#Elementary inverse demand: p(q) = a - bq,  with revenue aq-bq^2 and marginal revenue a-2bq
a = torch.tensor([10.0],requires_grad=False) #intercept, or choke price
b = torch.tensor([1.],requires_grad=False) #slope of the (inverse) demand curve
c = torch.tensor([2.], requires_grad=False) #constant marginal cost, no fixed cost

# simple numerical optimization over quantity
lr=0.25 #learning rate
q=torch.rand(1, requires_grad=True) #quantity, the choice variable
for i in range(10):
    profit = (a-c)*q - b*(q**2) #alternatively, loss = -profit
    profit.backward() 
    print(i, q.item(), profit.item())
    with torch.no_grad():
        q += lr*q.grad # -= for minimization problem i.e. for loss (change direction of adjustment)
        q.grad.zero_()
        
print('\nThe optimal quantity is {:.2f}\n'.format(q.item()))

# simple numerical optimization over price
lr=0.25
p = torch.tensor([5.],requires_grad=True)
for i in range(10):
	q = (a-p)/b
	loss = -(a-c)*q + b*(q**2) #cost - revenue
	loss.backward()
	print(i, p.item(), loss.item())
	with torch.no_grad():
		p -= lr*p.grad # -= for minimization problem
		p.grad.zero_()
	 
print('\nThe optimal price is {:.2f}\n'.format(p.item()))


# How about the consumer side? utility maximization
p = torch.rand(1,requires_grad=False) #price taking consumer with p<1 (given u(x)=ln(1+x))
lr=0.25 #learning rate
q=torch.rand(1, requires_grad=True) #quantity, the choice variable
for i in range(50):
    utility = torch.log(1.+ q) - p*q #utility - expenditure
    utility.backward() 
    print(i, q.item(), utility.item())
    with torch.no_grad():
        q += lr*q.grad # -= for minimization problem i.e. for loss (change direction of adjustment)
        q.grad.zero_()
        
print('\nThe user\'s optimal quantity for price {:.3f} is {:.2f}\n'.format(p.item(), q.item()))


## Equilibrium involves interacting consumer decisions with firm's decisions

a = torch.tensor([10.0],requires_grad=False) #intercept, or choke price
b = torch.tensor([1.],requires_grad=False) #slope of the (inverse) demand curve
c = torch.tensor([2.], requires_grad=False) #constant marginal cost, no fixed cost

#define demand function: qty demanded
#calling this function elsewhere won't work with autograd to calculate dq/dp. 
def q_d(p):
	lr=0.25 #learning rate
	q=torch.tensor([1.],requires_grad=True)
	for i in range(20):
		utility = a*q - 0.5*b*(q**2) -p*q #for quadratic util u(q) = aq - 0.5*b*q**2
		utility.backward() 
		with torch.no_grad():
			q += lr*q.grad # -= for minimization problem i.e. for loss (change direction of adjustment)
			q.grad.zero_()
			#p.grad.zero_()		
	return q#torch.clamp(q, min=0)

#version of demand where autograd will work (for dq/dp)
def q_d2(p):
	return (a-p)/b #first order condition 

lr=0.25
p = torch.tensor([1.],requires_grad=True)
for i in range(20):
	q = q_d(p) #q_d2(p) #autograd does not work with q_d. Needs explicit math relationships as in q_d2
	#approximate the derivative dq/dp manually
	q_eps = q_d(p+.0001)
	p_grad_eps = (q_eps.item()-q.item())/.0001
	
	#print(q)
	loss = -(a-c)*q + b*(q**2) #cost - revenue
	loss.backward()
	#print('\nThe grads are p.grad q.grad:', q.grad.item())
	print(i, p.item(), loss.item(), q.grad.item())
	with torch.no_grad():
		p -= lr*q.grad*p_grad_eps # -= for minimization problem
		q.grad.zero_()
	 
print('\nThe optimal price is {:.2f} and quantity {:.2f}\n'.format(p.item(), q.item()))

"""







# Moving on to Rochet and Tirole:
## version with no fixed costs or benefits

##potential number of buyers and sellers (overall, not all will join platform).
# normalized to measure 1
N_B, N_S = torch.tensor([1.,1.]) 

#initial fraction of actual users (not in RT)
#""sigmoid Requires theta > 0.5 ""
theta_B, theta_S = torch.tensor([.25,.25])  

#Actual participant measures (not in RT)
n_B = torch.tensor(theta_B * N_B, requires_grad=True)
n_S = torch.tensor(theta_S * N_S, requires_grad=True)


## Costs of providing the service
# Fixed costs:
#C_B, C_S = torch.tensor([.0,.0]) #platform's fixed costs per user (type dependent)

#Cost per transaction or interaction.
#different from RT (they have a common cost 'c' for each interaction).
c_B,c_S = torch.tensor([.0,.0]) #cost of facilitating each transaction between a buyer and a seller

#Member benefits also have fixed and variable components 
#B_B, B_S = torch.tensor([.0,.0]) #these membership benefits may be negative (e.g. app developers initial cost) 
b_B, b_S = torch.tensor([7.5,7.5]) #these may be negative for sellers, marginal cost of providing the service

#platform charges, prices or fees: fixed 'A' (e.g. membership) and variable 'a' (per transaction)
#A_B = torch.tensor([.0], requires_grad=False) #fixed
#A_S = torch.tensor([.0], requires_grad=False) #fixed
# using 'p' instead of 'a'. RT use p for per-transaction p = a + (A-C)/N 
p_B = torch.tensor([.1])#, requires_grad=True) #per transaction
p_S = torch.tensor([.1])#, requires_grad=True) #per transaction

lr = 0.01 #learning rate
#update prices to optimize profit
#print('i','nb','ns','pb','ps','profit')

for i in range(20):
	#Net utility for sellers in RT
	"""
	#network effect, net utility (per trans) gets scaled by number of participants on other side of platform
	U_S = (b_S-a_S)*n_B + (B_S - A_S) #from RT but using n_B not N_B
	n_S = prob(U_S>0) #from RT (writing this as n_S/N_s = prob does not change things much).
	n_S/N_S = prob(U_S>0) #but we do it anyway (consistent with using \theta fraction )
	#using logisting formulation n_S = (exp(u))/(1+ exp(u))	
	n_S/N_S = exp((b_S-p_S)*n_B)/(1 + exp((b_S-p_S)*n_B)) 
	More mods: replace externality effect n_B by tanh() version
	"""
	#rewriting above as inverse demand
	net_ext_B2S = (1+ torch.tanh(n_B - .75*N_B)) #diminishing returns after 75 percent of potential users
	"""sigmoid Requires theta > 0.5 """
	#p_S = b_S - (1/net_ext_B2S) * torch.log(n_S/(N_S - n_S)) #sigmoid 
	
	p_S = b_S - (1/net_ext_B2S) * torch.log((1 + n_S/N_S)/(1 - n_S/N_S) ) #2*sigmoid -1
	
	#similarly for buyers (inverse demand on the buyers side)
	net_ext_S2B = (1+ torch.tanh(n_S - .75*N_S))
	#p_B = b_B - (1/net_ext_S2B) * torch.log(n_B/(N_B - n_B)) #sigmoid
	p_B = b_B - (1/net_ext_S2B) * torch.log((1 + n_B/N_B)/(1 - n_B/N_B) ) #2*sigmoid -1
	"""
	notes: obviously n<1. But we also need n>N/2, else log(n/N-n) < 0 => p > b!
	The 1/2 comes from the logistic formulation. Other distributions will have different bounds.
	Which is why using n/N = prob(u>0) doesn't change things much. But we do it anyway.	
	tanh() is one option, 
	
	other is (2*sigmoid -1) which leads to	
	p_S = b_S - (1/net_ext_B2S) * torch.log((1 + n_S/N_S)/(1 - n_S/N_S) )
	p_B = b_B - (1/net_ext_S2B) * torch.log((1 + n_B/N_B)/(1 - n_B/N_B) )
	"""
	
	
	#print(b_B.item(),b_S.item(), p_B.item(), p_S.item())
	
	#platform profit: number of interactions (nB*NB*nS*NS) times margin from each
	profit = n_B * n_S* (p_S - c_S + p_B - c_B) #neglecting N_B * N_S has no impact
	print(i, n_B.item(), n_S.item(),p_B.item(), p_S.item(), profit.item())

	profit.backward()#retain_graph=True)
	with torch.no_grad():
		n_B += lr*n_B.grad
		n_B.grad.zero_()
		n_S += lr*n_S.grad
		n_S.grad.zero_()




#simple plot for participation probability (logistic function)
import matplotlib.pyplot as plt
import numpy as np


#The above as a function
def adopt(
			N_B = torch.tensor([1.]), #exog
			N_S = torch.tensor([1.]), #Exog
			theta_B=torch.tensor([.25]), theta_S = torch.tensor([.25]), #Exog not interesting
			c_B = torch.tensor([.0]),c_S = torch.tensor([.0]), #Exog
			b_B= torch.tensor([7.5]), b_S = torch.tensor([7.5]), #Exog
			#p_B= torch.tensor([.1]), p_S = torch.tensor([.1]) #initialize
			):
	n_B = torch.tensor(theta_B * N_B, requires_grad=True)
	n_S = torch.tensor(theta_S * N_S, requires_grad=True)
	
	lr = 0.01 #learning rate
	for i in range(10):		
		#rewriting above as inverse demand
		net_ext_B2S = (1+ torch.tanh(n_B - .75*N_B)) #diminishing returns after 75 percent of potential users
		#p_S = b_S - (1/net_ext_B2S) * torch.log(n_S/(N_S - n_S))
		p_S = b_S - (1/net_ext_B2S) * torch.log((1 + n_S/N_S)/(1 - n_S/N_S) )
		#similarly for buyers (inverse demand on the buyers side)
		net_ext_S2B = (1+ torch.tanh(n_S - .75*N_S))
		#p_B = b_B - (1/net_ext_S2B) * torch.log(n_B/(N_B - n_B))
		p_B = b_B - (1/net_ext_S2B) * torch.log((1 + n_B/N_B)/(1 - n_B/N_B) )
		#platform profit: number of interactions (nB*NB*nS*NS) times margin from each
		profit = n_B * n_S* (p_S - c_S + p_B - c_B) #neglecting N_B * N_S has no impact
		#print(i, n_B.item(), n_S.item(),p_B.item(), p_S.item(), profit.item())
		profit.backward()
		with torch.no_grad():
			n_B += lr*n_B.grad
			n_B.grad.zero_()
			n_S += lr*n_S.grad
			n_S.grad.zero_()

	return n_B, n_S, p_B, p_S


print('\n\n')
n_B, n_S, p_B, p_S = adopt(N_S= .5)
print('nb','ns','pb','ps')
print(n_B.item(), n_S.item(), p_B.item(), p_S.item())



"""
x = np.arange(0,10,.2)
y1 = np.exp(x)/(1+np.exp(x))
y2 = 2* np.exp(x)/(1+np.exp(x))
z1 = np.tanh(x)
z2 = (1 + np.tanh(x-1))*.5 
f, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
ax1.plot(x, y1)
ax1.plot(x, y2-1)
ax1.set_title('exp(x)/(1+exp(x))')

#ax2.set_title('exp(x-1)/(1+exp(x-1))')
ax2.plot(x,z1)
ax2.plot(x,y1)
ax2.set_title('tanh(x), (1+tanh(x-1))/2 ')
#ax3.set_title('(1+tanh(x-1))/2')
plt.show()
"""

"""
Determine optimal adoption ( i.e. marketplace participation) rates
"""
import torch
import matplotlib.pyplot as plt
import numpy as np


def adopt(
			N_B = torch.tensor([1.]), #exog
			N_S = torch.tensor([1.]), #Exog
			theta_B=torch.tensor([.25]), theta_S = torch.tensor([.25]), #Exog not interesting
			c_B = torch.tensor([.0]),c_S = torch.tensor([.0]), #Exog
			b_B= torch.tensor([7.5]), b_S = torch.tensor([7.5]), #Exog
			cv_S = .75, cv_B = .75
			):
	"""
	returns platform  adoption rates and prices for buyers and sellers
	"""
	
	n_B = torch.tensor(theta_B * N_B, requires_grad=True)
	n_S = torch.tensor(theta_S * N_S, requires_grad=True)
	
	tol = .005
	delta = 1
	lr = 0.001 #learning rate
	#print('\n\n')
	#for i in range(25):		
	while delta > tol:
		#rewriting above as inverse demand
		net_ext_B2S = (1+ torch.tanh(n_B - cv_S*N_B)) #diminishing returns after 75 percent of potential users
		#p_S = b_S - (1/net_ext_B2S) * torch.log(n_S/(N_S - n_S))
		p_S = b_S - (1/net_ext_B2S) * torch.log((1 + n_S/N_S)/(1 - n_S/N_S) )
		#similarly for buyers (inverse demand on the buyers side)
		net_ext_S2B = (1+ torch.tanh(n_S - cv_B*N_S))
		#p_B = b_B - (1/net_ext_S2B) * torch.log(n_B/(N_B - n_B))
		p_B = b_B - (1/net_ext_S2B) * torch.log((1 + n_B/N_B)/(1 - n_B/N_B) )
		#platform profit: number of interactions (nB*NB*nS*NS) times margin from each
		profit = n_B * n_S* (p_S - c_S + p_B - c_B) #neglecting N_B * N_S has no impact
		#print(n_B.item(), n_S.item() )
		profit.backward()
		with torch.no_grad():
			delta = max(abs(lr*n_B.grad.item()/n_B.item()),abs(lr*n_S.grad.item()//n_S.item()))
			n_B += lr*n_B.grad
			n_B.grad.zero_()
			n_S += lr*n_S.grad
			n_S.grad.zero_()

	return n_B, n_S, p_B, p_S

"""
print('\n\n')
n_B, n_S, p_B, p_S = adopt(N_S= .5)
print('nb','ns','pb','ps')
print(n_B.item(), n_S.item(), p_B.item(), p_S.item())

quit()
"""

#plot outcomes for various N_S
N_B=torch.tensor([1.])
N_S = torch.arange(.5,1.5,.1)
len = N_S.size()[0]
n_B, n_S, p_B, p_S = torch.empty([len]), torch.empty([len]),torch.empty([len]),torch.empty([len])
for i in range(len):
	n_B[i], n_S[i],p_B[i],p_S[i] = adopt(N_B = N_B, N_S = N_S[i])
	
f,((ax1,ax3), (ax2,ax4)) = plt.subplots(2,2,sharey='row', sharex='col')
ax1.plot(N_S.numpy(), n_B.detach().numpy()/N_B.numpy(), 'orange', label='n_B/N_B') #detach() grad-required variable to
#ax1.plot(N_S.numpy(), n_S.detach().numpy(), 'pink', label='n_S')
ax1.plot(N_S.numpy(), n_S.detach().numpy()/N_S.numpy(), 'g', label='n_S/N_S')
ax1.legend()
#ax1.set_xlabel('N_S')
ax2.plot(N_S.numpy(), p_B.detach().numpy(), 'orange', label='p_B')
ax2.plot(N_S.numpy(), p_S.detach().numpy(), 'g', label='p_S')
ax2.legend()
ax2.set_xlabel('N_S')
#ax2.set_title('p_B (r) and p_S (b) with N_S')
#plt.show()

#del(n_B, n_S, p_B, p_S, len)


#plot outcomes for various b_S
cv_S = torch.tensor([.2,.8]) #torch.arange(.2,.8,.1)
len = cv_S.size()[0]
n_B, n_S, p_B, p_S = torch.empty([len]), torch.empty([len]),torch.empty([len]),torch.empty([len])
for i in range(len):
	n_B[i], n_S[i],p_B[i],p_S[i] = adopt(cv_S = cv_S[i])

#print(n_B)
#print('\n',n_S)	
#f,(ax1,ax2) = plt.subplots(1,2,sharey=False, sharex=False)
ax3.plot(cv_S.numpy(), n_B.detach().numpy(), 'orange', label='n_B/N_B') #detach() grad-required variable to
ax3.plot(cv_S.numpy(), n_S.detach().numpy(), 'g', label='n_S/N_B')
ax3.legend()
#ax3.set_xlabel('cv_S')
ax4.plot(cv_S.numpy(), p_B.detach().numpy(), 'orange', label='p_B')
ax4.plot(cv_S.numpy(), p_S.detach().numpy(), 'g', label='p_S')
ax4.legend()
ax4.set_xlabel('cv_S')
#ax2.set_title('p_B (r) and p_S (b) with N_S')
plt.show()


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
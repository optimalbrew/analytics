""" 
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License

Mods: print function. xrange() to range(), added split() to data to convert to words
' '.join(), instead of ''.join(), in the printing of generative process,
modified hyper paras to be smaller.
"""

import numpy as np

# data I/O
"""
original did not use .split()

Clearly lot more unique words than characters. This greatly increases the input size of 
word-based language models compared to character based input. However, since sentences are
composed of a few words, the length of RNN (unrolling) i.e history dependence will be less
with word-based models.
"""
data = open('input.txt', 'r').read().split() # should be simple plain text file 

chars = list(set(data)) #set extracts each unique character/word
data_size, vocab_size = len(data), len(chars)
print('data has {} characters, {} unique.' .format(data_size, vocab_size))
print('Sorted', sorted(chars))


#index dict (1 to 1 mapping) from char to position and back
#input and output (targets) are lists of integers 
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters (not learning here)
hidden_size = 50 # def 100. size of hidden layer of neurons
seq_length = 10 # def 25. number of steps to unroll the RNN for (too long can lead to overfitting)
"""
this is a sequence learning model that will generate sequences as well. 
e.g. input sequence is 25 chars, output is the sequence of 25 chars shifted by 1
a word-based model would use a lower number.
"""
learning_rate = 1e-1

# model parameters (these will be learned). Start with random initialization
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
#no bias term needed in the input layer? (No data! no training!) Unless manually add unknown, or rare or some other tag.
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias
"""note 1: the dimensions of "Wxh" and "Why" are opposite (obviously to each other, but also to what 
one might ordinarily think. e.g. Wxh should map from vocab_size*hidden). Could be done the 
other way, perhaps this is more convenient. This allows writing the dot prod as WX rather than XW.
note 2: no bias in input. not needed here as the list of char is assumed exhaustive?
"""


def lossFun(inputs, targets, hprev):
  """
  inputs, targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  
  does a forward pass and a backward pass to compute gradients
 
  returns:
  * loss, 
  * gradients on model parameters, 
  * and last hidden state: why is this needed? for checking the derivatives later?
  """
  xs, hs, ys, ps = {}, {}, {}, {} #empty dicts
  hs[-1] = np.copy(hprev) #assign val hprev to key "-1", we need h[t-1] below for t=0.
  loss = 0
  # forward pass: note the recursion via h[t-1] for all levels of "unrolling" the network
  #xs are data. hs[] and ys[] are computed. 
  for t in range(len(inputs)): #len(inputs) and the recursion of state below is tied to seq_length in the implementation
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1 #dummy var/ or one-hot encoding
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss) 

  # backward pass: compute gradients going backwards
  #np.zeros_like() returns zeros of same dim and type as its arg
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  #note: reversal in the range
  for t in reversed(range(len(inputs))):
    dy = np.copy(ps[t])
    #p_k = e^(f_k)/\sum(e^{f_j}) (softmax), and loss is -log softmax
    #Loss = -log(p(f_y)), partial(-log(p_k))/partial f_k = p_k - I(y=k), hence '-=' below..
    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]



def sample(h, seed_ix, n):
  """
  We can visualize the network's progress (in learning the sequence model)  
  by generating sequences from time to time. That's what this sampling achieves.

  h is memory state, seed_ix is seed letter for first time step
  sample a sequence of integers from the model  
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1 #starting seed (character or word)
  ixes = [] #sequence to be generated
  for t in range(n):
  	#do a forward pass
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    #obtain softmax probs for prediction
    p = np.exp(y) / np.sum(np.exp(y))
    #randomly draw a prediction based on above logistic prob 
    ix = np.random.choice(range(vocab_size), p=p.ravel()) #np.ravel() flattens to 1D
    	#https://docs.scipy.org/doc/numpy/reference/generated/numpy.ravel.html
    x = np.zeros((vocab_size, 1))
    x[ix] = 1 #save for the next iteration
    ixes.append(ix) #generated sequence
  return ixes


## and now the main implementation: recall seq_length is hyperparameter. Used to control
#input (and target) sequence length.

#iteration conter and data-sequence pointer
n, p = 0, 0

#memory variables
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
while True:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0: 
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  #target is just one spot ahead in the chain (including spaces, punctuation)
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  # sample from the model now and then
  if n % 100 == 0:
    sample_ix = sample(hprev, inputs[0], 200)
    #txt = ''.join(ix_to_char[ix] for ix in sample_ix) #char version, no spacing, ''.join()
    txt = ' '.join(ix_to_char[ix] for ix in sample_ix), #word version, ' '.join()
    print('----\n {} \n----' .format(txt))

  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: print('iter {}, loss: {}' .format(n, smooth_loss)) # print progress
  
  # perform parameter update with Adagrad
  """note the use of zip() to aggregate lists"""
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam #grad squared
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += seq_length # move data pointer
  n += 1 # iteration counter 


"""

# Separately, AK provides gradient checking code
# gradient checking
from random import uniform

def gradCheck(inputs, target, hprev):
  global Wxh, Whh, Why, bh, by
  num_checks, delta = 10, 1e-5
  _, dWxh, dWhh, dWhy, dbh, dby, _ = lossFun(inputs, targets, hprev)
  for param,dparam,name in zip([Wxh, Whh, Why, bh, by], [dWxh, dWhh, dWhy, dbh, dby], ['Wxh', 'Whh', 'Why', 'bh', 'by']):
    s0 = dparam.shape
    s1 = param.shape
    if s0 != s1:
    	print('Error dims dont match: {} and {}.' .format(s0, s1))
    
    print(name)
    for i in range(num_checks):
      ri = int(uniform(0,param.size))
      # evaluate cost at [x + delta] and [x - delta]
      old_val = param.flat[ri]
      param.flat[ri] = old_val + delta
      cg0, _, _, _, _, _, _ = lossFun(inputs, targets, hprev)
      param.flat[ri] = old_val - delta
      cg1, _, _, _, _, _, _ = lossFun(inputs, targets, hprev)
      param.flat[ri] = old_val # reset old value for this parameter
      # fetch both numerical and analytic gradient
      grad_analytic = dparam.flat[ri]
      grad_numerical = (cg0 - cg1) / ( 2 * delta )
      rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
      print('{}, {} => {} ' .format(grad_numerical, grad_analytic, rel_error))
      # rel_error should be on order of 1e-7 or less
      
"""
import sys
sys.path.append("/home/haggis/Desktop/StocPyDev")
import stocPyDev as stocPy
from venture.shortcuts import *
import ppUtils as pu
import scipy.stats as ss
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import cPickle

obs = (0.9, 0.8, 0.7, 0, -0.025, 5, 2, 0.1, 0, 0.13, 0.45, 6, 0.2, 0.3, -1, -1)
sProbs = (1.0/3, 1.0/3, 1.0/3)
 
tProbs = {
  0 : (0.1, 0.5, 0.4),
  1 : (0.2, 0.2, 0.6),
  2 : (0.15, 0.15, 0.7)
}
eMeans = (-1,1,0)
eProbs = {
  0 : lambda x: ss.norm.pdf(x, eMeans[0], 1),
  1 : lambda x: ss.norm.pdf(x, eMeans[1], 1),
  2 : lambda x: ss.norm.pdf(x, eMeans[2], 1)
}

def fwd_bkw(x, a_0, a, e):
  L = len(x) + 1
  states = range(len(a_0))

  fwd = []
  f_prev = {}
  # forward part of the algorithm
  for i, x_i in enumerate((None,) + x):
    f_curr = {}
    for st in states:
      if i == 0:
        # base case for the forward part
        prev_f_sum = a_0[st]
      else:
        prev_f_sum = sum(f_prev[k]*a[k][st] for k in states)
 
      if i == 0:
        f_curr[st] = prev_f_sum
      else:
        f_curr[st] = e[st](x_i) * prev_f_sum
 
    fwd.append(f_curr)
    f_prev = f_curr
 
  p_fwd = sum(f_curr[k] for k in states)
 
  bkw = []
  b_prev = {}
  # backward part of the algorithm
  for i, x_i_plus in enumerate(reversed(x+(None,))):
    b_curr = {}
    for st in states:
      if i == 0:
        # base case for backward part
        b_curr[st] = 1
      else:
        b_curr[st] = sum(a[st][l]*e[l](x_i_plus)*b_prev[l] for l in states)
 
    bkw.insert(0,b_curr)
    b_prev = b_curr
 
  p_bkw = sum(a_0[l] * e[l](x[0]) * b_curr[l] for l in states)
 
  # merging the two parts
  posterior = []
  for i in range(L):
    posterior.append({st: fwd[i][st]*bkw[i][st]/p_fwd for st in states})
 
  assert abs(p_fwd - p_bkw) < 0.0001
  return posterior

def hmmPost():
  rez = fwd_bkw(obs, sProbs, tProbs, eProbs)
  print '\n'.join(map(str, rez))
  vals = [[],[],[]]
  for col in rez:
    vals[0].append(col[0])
    vals[1].append(col[1])
    vals[2].append(col[2])
  plotHeatMap(vals)

def plotHeatMap(vals):
  plt.imshow(list(reversed(vals)), interpolation='None')
  plt.xticks(range(17))
  plt.yticks(range(3), ["2","1","0"])
  plt.show()

def genHeatMap(sampsDic=None, fn=None):
  if fn:
    with open(fn, 'r') as f:
      sampsDic = cPickle.load(f)
  vals = [[],[],[]]
  for name, samps in sorted(sampsDic.items(), key=lambda (k,v): int(k[1:])):
    norm = float(len(samps))
    vals[0].append(samps.count(0) / norm)
    vals[1].append(samps.count(1) / norm)
    vals[2].append(samps.count(2) / norm)
  print map(lambda v:np.mean(v[1:]), vals)
  plotHeatMap(vals)

def hmmVenture(v, xs, a_0, a, em, samples, burn = 0, step = 1):
  v.assume("getTransProb", "(lambda (s i) (if (= s 0) (if (= i 0) %f (if (= i 1) %f %f))"%a[0] + " (if (= s 1) (if (= i 0) %f (if (= i 1) %f %f))"%a[1] + " (if (= i 0) %f (if (= i 1) %f %f)))))"%a[2])
  v.assume("transition", "(lambda (prevState) (categorical (getTransProb prevState 0) (getTransProb prevState 1) (getTransProb prevState 2)))")
  v.assume("getState", "(mem (lambda (index) (if (<= index 0) (categorical %f %f %f) (transition (getState (- index 1))))))"%a_0)
  v.assume("getObsMean", "(lambda (s) (if (= s 0) %f (if (= s 1) %f %f)))"%em)

  [v.observe("(normal (getObsMean (getState " + str(ind) + ")) 1)", str(obs)) for ind,obs in enumerate(xs, start=1)]

  #v.assume("s16", "(getState 16)")
  [v.assume("s" + str(i), "(getState " + str(i) +")") for i in range(17)]
  #return map(lambda (x,y): y, pu.posterior_samples(v, "s16", samples, burn, step))
  return pu.posterior_mult_samples(v, ["s" + str(i) for i in range(17)], samples, burn, step)

stateDic = {}
def getState(index):
  global stateDic
  try:
    return stateDic[index]
  except:
    if index <= 0:
      dist = sProbs
    else:
      dist = tProbs[getState(index-1)]

    state = np.where(np.random.multinomial(1, dist) == 1)[0][0]
    stateDic[index] = state
    return state

def hmm(init=False):
  pass

if __name__ == "__main__":
  #v = make_church_prime_ripl()
  #with open("hmm/hmmVenture50k", 'w') as f:
  #  cPickle.dump(hmmVenture(v, obs, sProbs, tProbs, eMeans, 50000), f)

  #samps = hmmVenture(v, obs, sProbs, tProbs, eMeans, 10000)
  #norm = float(len(samps))
  #print samps.count(0) / norm, samps.count(1) / norm, samps.count(2) / norm
  #hmmPost()
  genHeatMap(fn = "hmm/hmmVenture50k")
  #hmmPost()

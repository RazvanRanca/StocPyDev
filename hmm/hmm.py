import sys
sys.path.append("/home/haggis/Desktop/StocPyDev")
import stocPyDev as stocPy
from venture.shortcuts import *
import scipy.stats as ss
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import cPickle
import ppUtils as pu
from collections import Counter
import time
import copy

#obs = (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
obs = (0.9, 0.8, 0.7, 0.0, -0.025, 5.0, 2.0, 0.1, 0.0, 0.13, 0.45, 6.0, 0.2, 0.3, -1.0, -1.0)

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
  #return map(lambda (x,y): y, stocPy.posterior_samples(v, "s16", samples, burn, step))
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

def categoricalExp(ps, name): #scipy.stats has no categorical. Should implement this in StocPy eventually.
  assert(abs(sum(ps) - 1) < 0.0001)
  c = stocPy.unifCont(0,1, name=name) 
  s = 0
  for i in range(len(ps)):
    s += ps[i]
    if s > c:
      return i
  assert(False) #shouldn't ever get here

stateHist = [] 
singleStateHist = {}
def hmmExp():
  states = []
  states.append(categoricalExp(sProbs, name="s0"))
  for i in range(1,17):
    states.append(categoricalExp(tProbs[states[i-1]], name="s"+str(i)))
    stocPy.normal(eMeans[states[i]], 1, cond=obs[i-1], name="c"+str(i))
  global stateHist
  global singleStateHist
  curTime = time.time() - startTime
  singleStateHist[curTime] = states[sind]
  #stateHist[curTime] = tuple(states)
  #for i in range(len(states)):
  #  try:
  #    stateHist[i][curTime] = states[i]
  #  except:
  #    stateHist.append({curTime:states[i]})

def categorical(ps): #scipy.stats has no categorical. Should implement this in StocPy eventually.
  assert(abs(sum(ps) - 1) < 0.0001)
  c = stocPy.stocPrim("uniform", (0,1), part=pind) 
  s = 0
  for i in range(len(ps)):
    s += ps[i]
    if s > c:
      return i
  assert(False) #shouldn't ever get here

def hmm():
  states = []
  states.append(stocPy.stocPrim("Categorical", (sProbs,), obs=True, part=pind))
  for i in range(1,17):
    #print states
    states.append(stocPy.stocPrim("Categorical", (tProbs[states[i-1]],), obs=True, part=pind))
    stocPy.normal(eMeans[states[i]], 1, cond=obs[i-1])

def hmmSpec():
  states = []
  states.append(categorical(sProbs))
  for i in range(1,17):
    states.append(categorical(tProbs[states[i-1]]))
    stocPy.normal(eMeans[states[i]], 1, cond=obs[i-1])
  #global singleStateHist
  curTime = time.time() - startTime
  #singleStateHist[curTime] = states[sind]
  global stateHist
  for i in range(len(states)):
    try:
      stateHist[i][curTime] = states[i]
    except:
      stateHist.append({curTime:states[i]})

def hmm1():
  states = []
  states.append(categorical(sProbs))
  for i in range(1,17):
    states.append(categorical(tProbs[states[i-1]]))
  for i in range(1,17):
    stocPy.normal(eMeans[states[i]], 1, cond=obs[i-1])
  global singleStateHist
  curTime = time.time() - startTime
  singleStateHist[curTime] = states[sind]
  #global stateHist
  #curTime = time.time() - startTime
  #stateHist[curTime] = tuple(states)
  #for i in range(len(states)):
  #  try:
  #    stateHist[i][curTime] = states[i]
  #  except:
  #    stateHist.append({curTime:states[i]})

def genRuns(model, noRuns, runTime, fn, alg, autoNames=False, discAll=False):
  global startTime
  global stateHist
  global singleStateHist
  runs = []
  for i in range(noRuns):
    print str(model), "Run", i
    #startTime = time.time()
    samples = stocPy.getTimedSamples(model, runTime, alg=alg, autoNames=autoNames, discAll=discAll, orderNames=True)
    #print map(lambda x: dict([(k,v/float(len(stateHist))) for (k,v) in sorted(dict(Counter(x)).items())]), zip(*stateHist))
    runs.append(samples)#copy.deepcopy(stateHist))#singleStateHist))#map(lambda x: dict([(k,v/float(len(stateHist))) for (k,v) in sorted(dict(Counter(x)).items())]), zip(*stateHist)))
    #stateHist = []
    #singleStateHist = {}
  cd = stocPy.getCurDir(__file__)
  #print runs
  with open(cd + "/" + fn, 'w') as f:
    cPickle.dump(runs, f)

def showRuns(fn, proc=False):
  with open(fn, 'r') as f:
    runs = cPickle.load(f)
  for run in runs:
    if proc:
      run = map(lambda dic: dict([(k,v/float(len(dic))) for (k,v) in dict(Counter(dic.values())).items()]), run)
    print run
    vals = [[],[],[]]
    for col in run:
      vals[0].append(col.get(0,0))
      vals[1].append(col.get(1,0))
      vals[2].append(col.get(2,0))
    plotHeatMap(vals)

def procRuns(posts, fns, names, proc=False, aggSums = True):
  runs = []
  for fn in fns:
    with open(fn, 'r') as f:
      samples = cPickle.load(f)
      if proc:
        for run in samples:
          dic = []      
          for curTime, states in sorted(run.items()):
            for i in range(len(states)):
              try:
                dic[i][curTime] = states[i]
              except:
                dic.append({curTime:states[i]})
          runs.append(copy.deepcopy(dic))
      else:
        runs.append(samples)
  print map(len, runs)
  print map(len, runs[0])
  #print map(len, runs[0][0])
  #print '\n'.join([str([dict([(k,str(v/float(len(runs[m][r])))[:5]) for (k,v) in dict(Counter(runs[m][r].values())).items()]) for r in range(len(runs[0]))]) for m in range(len(runs))])
  print posts
  stocPy.calcKLSumms(posts, runs, names=names, burnIn = 0, aggSums=aggSums, title="Performance on HMM model")# + str(sind), xlabel="Seconds")

if __name__ == "__main__":
  #hmmPost()
  global sind
  global pind
  sind = 5 
  pind = 5 
  #genRuns(hmm, noRuns=10, runTime=60, fn="hmmRunPart5_10_60", alg="met", discAll = True, autoNames=True)
  #showRuns("hmmRunMet_2_test_"+str(sind), proc=True)
  #assert(False)
  #genRuns(hmm, noRuns=10, runTime=600, fn="hmmRunSliceTD_10_600", alg="sliceTD", discAll=True, autoNames=True)
  pind = 1
  #genRuns(hmm, noRuns=10, runTime=600, fn="hmmRunPart1_10_600", alg="met", autoNames=True)
  posts = fwd_bkw(obs, sProbs, tProbs, eProbs)#[sind]
  #procRuns(posts, ["hmmRunMet_2_test_"+str(sind), "hmmRunSliceTD_2_test_"+str(sind)], ["Met", "SliceTD"], proc=False, aggSums=True)
  procRuns(posts, ["hmmRunMet_10_600", "hmmRunSliceTD_10_600"], ["Met", "SliceTD"], proc=False, aggSums=True)
  #procRuns(posts, ["hmmRunPart5_10_60"], ["Part5"], proc=False, aggSums=True)
  #stocPy.getTimedSamples(hmm, 30, alg="met")
  #print '\n'.join(map(lambda x: str([(k,str(v/float(len(stateHist)))[:5]) for (k,v) in sorted(dict(Counter(x)).items())]), zip(*stateHist)))
  #v = make_church_prime_ripl()
  #with open("hmmVenture10k", 'w') as f:
  #  cPickle.dump(hmmVenture(v, obs, sProbs, tProbs, eMeans, 10000), f)

  #samps = hmmVenture(v, obs, sProbs, tProbs, eMeans, 10000)
  #norm = float(len(samps))
  #print samps.count(0) / norm, samps.count(1) / norm, samps.count(2) / norm
  #hmmPost()
  #genHeatMap(fn = "hmmVenture10k")
  #hmmPost()

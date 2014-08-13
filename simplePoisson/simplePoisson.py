import sys
sys.path.append("/home/haggis/Desktop/StocPyDev")
import stocPyDev as stocPy
import scipy.stats as ss
import math
import numpy as np
from matplotlib import pyplot as plt
import cPickle

noLetters1 = [7,2,3,1,4,0,0]
noLetters2 = [2,2,2,2,2,2,2]# # number of letters received each day for a week
noLetters3 = [8,8,8,8,8,8,8]
noLetters4 = [2,8]

def lettersPerDayPartU20(): # generative model
  expectedLetters = stocPy.stocPrim("uniform",(0, 20), obs=True, part=stocPy.stocPrim("randint",(0,21))) # prior on the number of letters
  for datum in noLetters:
    stocPy.poisson(expectedLetters, cond=datum) # conditioning on the data we have

def lettersPerDayPart10(): # generative model
  expectedLetters = stocPy.stocPrim("uniform",(0, 20), obs=True, part=10) # prior on the number of letters
  for datum in noLetters:
    stocPy.poisson(expectedLetters, cond=datum) # conditioning on the data we have

def lettersPerDay(): # generative model
  expectedLetters = stocPy.unifCont(0, 20, obs=True) # prior on the number of letters
  for datum in noLetters:
    stocPy.poisson(expectedLetters, cond=datum) # conditioning on the data we have

def anlPost(data, ps, prior = ss.uniform(0,20).pdf):
  probs = {}
  for p in ps:
    val = prior(p)
    if val > 0:
      ep = math.exp(-float(p))
      for d in data:
	val *= float(p)**float(d) * ep / math.factorial(float(d))
    probs[p] = val
  probs = stocPy.norm(probs)
  return probs

def genRuns(model, noRuns, time, fn):
  runs = []
  for i in range(noRuns):
    print str(model), "Run", i
    samples = stocPy.getTimedSamples(model, time)
    assert(len(samples.keys()) == 1)
    samples = samples[samples.keys()[0]]
    runs.append(samples)
  cd = stocPy.getCurDir(__file__)
  with open(cd + "/" + fn, 'w') as f:
    cPickle.dump(runs, f)

def genPost(ps, fn=None, show=False):
  post = anlPost(noLetters, ps) 
  xs, ys = zip(*sorted(post.items()))
  if fn:
    cd = stocPy.getCurDir(__file__)
    with open(cd + "/" + fn, 'w') as f:
      cPickle.dump((xs,ys), f)
  if show:
    plt.plot(xs, ys)
    plt.show()

if __name__ == "__main__":
  no = "4"
  iters = 5
  time = 20
  term = "_" + str(iters) + "_" + str(time)
  global noLetters
  noLetters = getattr(sys.modules[__name__], "noLetters" + no)
  genPost(np.arange(0,20,0.01), show=True, fn = "post" + no)
  genRuns(lettersPerDay, iters, time, "runs" + no + term )
  genRuns(lettersPerDayPart10, iters, time, "runs" + no + "Part10" + term)
  genRuns(lettersPerDayPartU20, iters, time, "runs" + no + "PartU20" + term)
  cd = stocPy.getCurDir(__file__)
  stocPy.calcKSTests(cd + "/post" + no , [cd + "/runs" + no + term, cd + "/runs" + no + "Part10" + term, cd + "/runs" +  no + "PartU20" + term], names = ["Depth0", "Depth10", "Depth_U20"], burnIn=100, aggFreq=np.logspace(1,math.log(1000000,10),10), modelName="simplePoisson" + no)
  #stocPy.calcKSSumms(cd + "/post2" , [cd + "/runs2_10_1m", cd + "/runs2Part10_10_1m", cd + "/runs2PartU20_10_1m"], names = ["Depth0", "Depth10", "Depth_U20"], burnIn=100, aggFreq=np.logspace(1,math.log(1000000,10),10), modelName="simplePoisson2")

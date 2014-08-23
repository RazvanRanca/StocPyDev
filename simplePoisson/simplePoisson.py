import sys
sys.path.append("/home/haggis/Desktop/StocPyDev")
import stocPyDev as stocPy
import scipy.stats as ss
import math
import numpy as np
from matplotlib import pyplot as plt
import cPickle

noLetters1 = [7,2,3,1,4,0,0]
noLetters2 = [2,2,2,2,2,2,2]# number of letters received each day for a week
noLetters3 = [8,8,8,8,8,8,8]
noLetters4 = [2,8]
noLetters5 = [2,0]

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

def lettersPerDayExp(): # generative model
  expectedLetters = stocPy.unifCont(0, 20, obs=True, name="p") # prior on the number of letters
  for i in range(len(noLetters)):
    stocPy.poisson(expectedLetters, cond=noLetters[i], name="c" + str(i)) # conditioning on the data we have

def wrapLetter():
  for i in range(5):
    lettersPerDay()

def lettersPerDay0(): # generative model
  expectedLetters = stocPy.unifCont(0, 20, obs=True) # prior on the number of letters
  stocPy.poisson(expectedLetters, cond=0) # conditioning on the data we have

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

def genRuns(model, noRuns, time, fn, alg="met", autoNames=True):
  runs = []
  for i in range(noRuns):
    print str(model), "Run", i
    samples = stocPy.getTimedSamples(model, time, alg=alg, autoNames=autoNames)
    assert(len(samples.keys()) == 1)
    samples = samples[samples.keys()[0]]
    runs.append(samples)
  cd = stocPy.getCurDir(__file__)
  with open(cd + "/" + fn, 'w') as f:
    cPickle.dump(runs, f)

def genPost(ps, data, fn=None, show=False, prior=ss.uniform(0,20).pdf, xlim=None, title=None):
  post = anlPost(data, ps, prior) 
  xs, ys = zip(*sorted(post.items()))
  if fn:
    cd = stocPy.getCurDir(__file__)
    with open(cd + "/" + fn, 'w') as f:
      cPickle.dump((xs,ys), f)
  if show:
    plt.plot(xs, ys)
    if xlim:
      plt.xlim(xlim)
    if title:
      plt.title(title)
    plt.show()

if __name__ == "__main__":
  #for i in [0,1,2,4,8,16,32,64,128,256]:
  #  genPost(np.arange(0,10,0.01), prior=ss.uniform(0,10).pdf, data=[4]*i, show=True, xlim=(-1,11), title="data = " + str(i) + " 4s")
  #assert(False)
  no = "1"
  iters = 5
  time = 30
  term = "_" + str(iters) + "_" + str(time)
  global noLetters
  noLetters = getattr(sys.modules[__name__], "noLetters" + no)

  #met = stocPy.extractDict(stocPy.getTimedSamples(lettersPerDay, 10, alg="met"))
  #metExp = stocPy.extractDict(stocPy.getTimedSamples(lettersPerDayExp, 10, alg="met", autoNames=False))
  #slic = stocPy.extractDict(stocPy.getTimedSamples(lettersPerDay, 10, alg="sliceTD"))
  #slicExp = stocPy.extractDict(stocPy.getTimedSamples(lettersPerDayExp, 10, alg="sliceTD", autoNames=False))
  #print len(met)/float(len(metExp)), len(slic)/float(len(slicExp))
  #assert(False)
  #genPost(np.arange(0,20,0.01), noLetters, show=True, fn = "post" + no)
  #genRuns(lettersPerDayExp, iters, time, "runs" + no + "Exp" + term, autoNames=False)
  #genRuns(lettersPerDay, iters, time, "runs" + no + term)
  genRuns(lettersPerDayPart10, iters, time, "runs" + no + "Part10" + term, alg="met")
  genRuns(lettersPerDayPart10, iters, time, "runs" + no + "SlicePart10" + term, alg="sliceTD")
  #genRuns(lettersPerDayPartU20, iters, time, "runs" + no + "PartU20" + term)
  cd = stocPy.getCurDir(__file__)
  #stocPy.calcKSTests(cd + "/post" + no , [cd + "/runs" + no + term, cd + "/runs" + no + "Part10" +term, cd + "/runs" +  no + "PartU20" + term], names = ["Depth0", "Depth10", "Depth_U20"], burnIn=100, aggFreq=np.logspace(1,math.log(1000000,10),10), modelName="simplePoisson" + no)
  stocPy.calcKSSumms(cd + "/post" + no , [cd + "/runs" + no + term, cd + "/runs" + no + "Part10" + term, cd + "/runs" +  no + "SlicePart10" + term], names = ["Depth0", "Depth10", "SliceDepth10"], burnIn=100, aggFreq=np.logspace(1,math.log(1000000,10),10), modelName="simplePoisson" + no)
  #stocPy.calcKSSumms(cd + "/post" + no , [cd + "/runs" + no + term, cd + "/runs" + no + "Exp" + term, cd + "/runs" +  no + "PartU20" + term], names = ["AutoNames", "ExpNames", "Auto-U20"], burnIn=100, aggFreq=np.logspace(1,math.log(1000000,10),10), modelName="simplePoisson" + no)

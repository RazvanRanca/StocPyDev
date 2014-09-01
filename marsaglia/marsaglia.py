import sys
sys.path.append("/home/haggis/Desktop/StocPyDev/")
import stocPyDev as stocPy
import math
from matplotlib import pyplot as plt
import cPickle
import numpy as np
import scipy.stats as ss


def marsaglia(mean, var):
  x = stocPy.stocPrim("uniform", (-1, 2), part=pind) #ss params are start and length of interval
  y = stocPy.stocPrim("uniform", (-1, 2), part=pind)
  s = x*x + y*y
  #print sampleInd, depth, x, y, s
  if s < 1:
    return mean + (math.sqrt(var) * (x * math.sqrt(-2 * (math.log(s) / s))))
  else:
    return marsaglia(mean, var)

obsMean = []
def marsagliaMean():
  global sampleInd
  mean = marsaglia(1, 5)
  stocPy.normal(mean, math.sqrt(2), cond=9)
  stocPy.normal(mean, math.sqrt(2), cond=8)
  obsMean.append(mean)

def normalMean():
  mean = stocPy.normal(1, math.sqrt(5), obs=True)
  stocPy.normal(mean, math.sqrt(2), cond=9)
  stocPy.normal(mean, math.sqrt(2), cond=8)

def getPost(start, end, inc, show = True, fn=None, rfn = None):
  xs = []
  ys = []
  if rfn:
    with open(rfn,'r') as f:
      xs, ys = cPickle.load(f)
  else:
    for m in np.arange(start, end+inc, inc):
      xs.append(m)
      ys.append(ss.norm.pdf(9, m, math.sqrt(2)) * ss.norm.pdf(8, m, math.sqrt(2)) * ss.norm.pdf(m, 1, math.sqrt(5)))
    ys = stocPy.norm(ys)
  
  if show:
    plt.plot(xs,ys, linewidth=3)
    plt.ylabel("Probability", size=20)
    plt.xlabel("x", size=20)
    plt.title("True Posterior for MarsagliaMean model", size=30)
    plt.show()
  if fn:
    with open(fn,'w') as f:
      cPickle.dump((xs,ys),f)
  return dict(zip(xs, ys))

def genRuns(model, noRuns, time, fn, alg="met", autoNames=True):
  global obsMean
  runs = []
  for i in range(noRuns):
    print str(model), "Run", i
    samples, traceAcc = stocPy.getTimedSamples(model, time, alg=alg, autoNames=autoNames, outTraceAcc=True)
    runs.append(stocPy.procUserSamples(obsMean, traceAcc))
    obsMean = []
  cd = stocPy.getCurDir(__file__)
  print map(lambda run: (min(run.values()), max(run.values())), runs)
  with open(cd + "/" + fn, 'w') as f:
    cPickle.dump(runs, f)

if __name__ == "__main__":
  global pind 
  pind = None 
  #print stocPy.getSamples(marsagliaMean, 1000, alg="sliceTD")
  #print obsMean
  #assert(False)
  #getPost(-20, 35, 0.001, show=False, fn="marsagliaPost")
  #stocPy.plotCumPost(getPost(3, 11, 0.01, show=False))
  #getPost(3, 11, 0.01, show=True)
  #stocPy.plotSamples(stocPy.getSamples(normalMean, 1000, alg="sliceTD"))
  cd = stocPy.getCurDir(__file__) + "/"
  global pind 
  pind = None
  noRuns = 5
  time = 6
  term = "_" + str(noRuns) + "_" + str(time)
  genRuns(marsagliaMean, noRuns=noRuns, time=time, fn="MetRuns" + term, alg="met")
  genRuns(marsagliaMean, noRuns=noRuns, time=time, fn="SliceRuns" + term, alg="sliceTD")
  pind = 5
  genRuns(marsagliaMean, noRuns=noRuns, time=time, fn="Part" + str(pind) + "Runs" + term, alg="met")
  genRuns(marsagliaMean, noRuns=noRuns, time=time, fn="SlicePart" + str(pind) + "Runs" + term, alg="sliceTD")
  #samples, traceAcc = stocPy.getTimedSamples(marsagliaMean, 10, alg="met", outTraceAcc=True)
  #print '\n'.join(map(str, samples.items()))
  #print obsMean, traceAcc
  #print procSamples(obsMean, traceAcc)
  #stocPy.plotSamples(stocPy.procUserSamples(obsMean, traceAcc))
  #stocPy.plotSamples(stocPy.getSamples(marsagliaMean, 10, alg="met"))
  #stocPy.calcKSTests(cd + "marsagliaPost" , [cd + "MetRuns" + term, cd + "SliceRuns" + term, cd + "Part" + str(pind) + "Runs" + term, cd + "SlicePart" + str(pind) + "Runs" + term], names = ["Met", "Slice", "Part" + str(pind), "SlicePart" + str(pind)], burnIn=0, aggFreq=np.logspace(1,math.log(1000000,10),10), modelName="Marsaglia")
  stocPy.calcKSSumms(cd + "marsagliaPost" , [cd + "MetRuns" + term, cd + "SliceRuns" + term, cd + "Part" + str(pind) + "Runs" + term, cd + "SlicePart" + str(pind) + "Runs" + term], names = ["Met", "Slice", "Part" + str(pind), "SlicePart" + str(pind)], burnIn=0, aggFreq=np.logspace(1,math.log(1000000,10),10), modelName="Marsaglia")

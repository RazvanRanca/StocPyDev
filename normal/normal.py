import sys
sys.path.append("/home/haggis/Desktop/StocPyDev/")
import stocPyDev as stocPy
import math
import scipy.stats as ss
import numpy as np
from matplotlib import pyplot as plt
from venture.shortcuts import *
import cPickle

obs = 5

ms = []
vs = []

def normal1Dec():
  m1 = stocPy.normal(0, math.sqrt(0.5), obs=True)
  m2 = stocPy.normal(0, math.sqrt(0.25), obs=True)
  m3 = stocPy.normal(0, math.sqrt(0.125), obs=True)
  m4 = stocPy.normal(0, math.sqrt(0.0625), obs=True)
  m5 = stocPy.normal(0, math.sqrt(0.0625), obs=True)
  m = m1+m2+m3+m4+m5
  cond = None
  if init:
    cond = obs
  stocPy.normal(m, 1, cond)

def normal2Dec1():
  m1 = stocPy.normal(0, math.sqrt(0.5), obs=True)
  m2 = stocPy.normal(0, math.sqrt(0.25), obs=True)
  m3 = stocPy.normal(0, math.sqrt(0.125), obs=True)
  m4 = stocPy.normal(0, math.sqrt(0.0625), obs=True)
  m5 = stocPy.normal(0, math.sqrt(0.0625), obs=True)
  m = m1+m2+m3+m4+m5
  v = stocPy.invGamma(3, 1)
  cond = None
  if init:
    cond = obs
  stocPy.normal(m, math.sqrt(v), cond)

def normal2Dec2():
  m1 = stocPy.normal(0, math.sqrt(0.5), obs=True)
  m2 = stocPy.normal(0, math.sqrt(0.25), obs=True)
  m3 = stocPy.normal(0, math.sqrt(0.125), obs=True)
  m4 = stocPy.normal(0, math.sqrt(0.0625), obs=True)
  m5 = stocPy.normal(0, math.sqrt(0.0625), obs=True)
  m = m1+m2+m3+m4+m5

  v1 = stocPy.normal(0, math.sqrt(0.5))
  v2 = stocPy.normal(0, math.sqrt(0.25))
  v3 = stocPy.normal(0, math.sqrt(0.125))
  v4 = stocPy.normal(0, math.sqrt(0.0625))
  v5 = stocPy.normal(0, math.sqrt(0.0625))
  vn = ss.norm.cdf(v1+v2+v3+v4+v5, loc=0, scale=1)
  v = ss.invgamma.ppf(vn, 3, loc=0, scale=1) 
  cond = None
  if init:
    cond = obs
  stocPy.normal(m, math.sqrt(v), cond)

def normal4Dec2():
  global ms
  global vs
  m1 = stocPy.normal(0, math.sqrt(0.5), obs=True)
  m2 = stocPy.normal(0, math.sqrt(0.25), obs=True)
  m3 = stocPy.normal(0, math.sqrt(0.125), obs=True)
  m4 = stocPy.normal(0, math.sqrt(0.0625), obs=True)
  m5 = stocPy.normal(0, math.sqrt(0.0625), obs=True)
  m = m1+m2+m3+m4+m5
  if m > 0:
    v = 1.0/3
  else:
    v1 = stocPy.normal(0, math.sqrt(0.5))
    v2 = stocPy.normal(0, math.sqrt(0.25))
    v3 = stocPy.normal(0, math.sqrt(0.125))
    v4 = stocPy.normal(0, math.sqrt(0.0625))
    v5 = stocPy.normal(0, math.sqrt(0.0625))
    vn = ss.norm.cdf(v1+v2+v3+v4+v5, loc=0, scale=1)
    v = ss.invgamma.ppf(vn, 3, loc=0, scale=1) 
    vs.append(v)
  cond = None
  if init:
    cond = obs
  stocPy.normal(m, math.sqrt(v), cond)

  ms.append(m)

def normal4Dec1():
  global ms
  global vs
  m1 = stocPy.normal(0, math.sqrt(0.5), obs=True)
  m2 = stocPy.normal(0, math.sqrt(0.25), obs=True)
  m3 = stocPy.normal(0, math.sqrt(0.125), obs=True)
  m4 = stocPy.normal(0, math.sqrt(0.0625), obs=True)
  m5 = stocPy.normal(0, math.sqrt(0.0625), obs=True)
  m = m1+m2+m3+m4+m5s
  if m > 0:
    v = 1.0/3
  else:
    v = stocPy.invGamma(3, 1)
    vs.append(v)
  cond = None
  if init:
    cond = obs
  stocPy.normal(m, math.sqrt(v), cond)

  ms.append(m)

def normal6Dec5():
  n = 5
  var = 1.0
  ms = []
  for i in range(n):
    ms.append(stocPy.normal(0, math.sqrt(var/(2**(i+1))), obs=True))
  ms.append(stocPy.normal(0, math.sqrt(var/(2**n)), obs=True))
  m = sum(ms)
  for datum in normalData:
    stocPy.normal(m, 1, datum)

def normal6Dec5():
  n = 5
  var = 10.0
  ms = []
  for i in range(n):
    ms.append(stocPy.normal(0, math.sqrt(var/(2**(i+1))), obs=True))
  ms.append(stocPy.normal(0, math.sqrt(var/(2**n)), obs=True))
  m = sum(ms)
  for datum in normalData:
    stocPy.normal(m, 1, datum)

def normal6Dec10():
  n = 10
  var = 10.0
  ms = []
  for i in range(n):
    ms.append(stocPy.normal(0, math.sqrt(var/(2**(i+1))), obs=True))
  ms.append(stocPy.normal(0, math.sqrt(var/(2**n)), obs=True))
  m = sum(ms)
  for datum in normalData:
    stocPy.normal(m, 1, datum)

def normal6Dec15():
  n = 15
  var = 10.0
  ms = []
  for i in range(n):
    ms.append(stocPy.normal(0, math.sqrt(var/(2**(i+1))), obs=True))
  ms.append(stocPy.normal(0, math.sqrt(var/(2**n)), obs=True))
  m = sum(ms)
  for datum in normalData:
    stocPy.normal(m, 1, datum)

def normal7Dec5():
  n = 5
  var = 1.0
  ms = []
  for i in range(n):
    ms.append(stocPy.normal(0, math.sqrt(var/(2**(i+1))), obs=True))
  ms.append(stocPy.normal(0, math.sqrt(var/(2**n)), obs=True))
  m = 100 * ss.norm.cdf(sum(ms), loc = 0, scale = 1)
  for datum in normalData:
    stocPy.normal(m, 1, datum)

def normal7Dec10():
  n = 10
  var = 1.0
  ms = []
  for i in range(n):
    ms.append(stocPy.normal(0, math.sqrt(var/(2**(i+1))), obs=True))
  ms.append(stocPy.normal(0, math.sqrt(var/(2**n)), obs=True))
  m = 100 * ss.norm.cdf(sum(ms), loc = 0, scale = 1)
  for datum in normalData:
    stocPy.normal(m, 1, datum)

def normal7Dec15():
  n = 15
  var = 1.0
  ms = []
  for i in range(n):
    ms.append(stocPy.normal(0, math.sqrt(var/(2**(i+1))), obs=True))
  ms.append(stocPy.normal(0, math.sqrt(var/(2**n)), obs=True))
  m = 100 * ss.norm.cdf(sum(ms), loc = 0, scale = 1)
  for datum in normalData:
    stocPy.normal(m, 1, datum)

def normal8Dec5():
  n = 5
  var = 1.0
  ms = []
  for i in range(n):
    ms.append(stocPy.normal(0, math.sqrt(var/(2**(i+1))), obs=True))
  ms.append(stocPy.normal(0, math.sqrt(var/(2**n)), obs=True))
  m = 10000 * ss.norm.cdf(sum(ms), loc = 0, scale = 1)
  for datum in normalData:
    stocPy.normal(m, 1, datum)

def normal8Dec10():
  n = 10
  var = 1.0
  ms = []
  for i in range(n):
    ms.append(stocPy.normal(0, math.sqrt(var/(2**(i+1))), obs=True))
  ms.append(stocPy.normal(0, math.sqrt(var/(2**n)), obs=True))
  m = 10000 * ss.norm.cdf(sum(ms), loc = 0, scale = 1)
  for datum in normalData:
    stocPy.normal(m, 1, datum)

def normal8Dec15():
  n = 15
  var = 1.0
  ms = []
  for i in range(n):
    ms.append(stocPy.normal(0, math.sqrt(var/(2**(i+1))), obs=True))
  ms.append(stocPy.normal(0, math.sqrt(var/(2**n)), obs=True))
  m = 10000 * ss.norm.cdf(sum(ms), loc = 0, scale = 1)
  for datum in normalData:
    stocPy.normal(m, 1, datum)

def normal8DecU20():
  n = stocPy.stocPrim("randint", (0, 21))
  var = 1.0
  ms = []
  for i in range(n):
    ms.append(stocPy.normal(0, math.sqrt(var/(2**(i+1))), obs=True))
  ms.append(stocPy.normal(0, math.sqrt(var/(2**n)), obs=True))
  m = 10000 * ss.norm.cdf(sum(ms), loc = 0, scale = 1)
  for datum in normalData:
    stocPy.normal(m, 1, datum)

def normal9Dec5():
  n = 5
  var = 1.0
  ms = []
  for i in range(n):
    ms.append(stocPy.normal(0, math.sqrt(var/(2**(i+1))), obs=True))
  ms.append(stocPy.normal(0, math.sqrt(var/(2**n)), obs=True))
  m = sum(ms)

def normal9Dec10():
  n = 10
  var = 1.0
  ms = []
  for i in range(n):
    ms.append(stocPy.normal(0, math.sqrt(var/(2**(i+1))), obs=True))
  ms.append(stocPy.normal(0, math.sqrt(var/(2**n)), obs=True))
  m = sum(ms)

def normal9Dec15():
  n = 15
  var = 1.0
  ms = []
  for i in range(n):
    ms.append(stocPy.normal(0, math.sqrt(var/(2**(i+1))), obs=True))
  ms.append(stocPy.normal(0, math.sqrt(var/(2**n)), obs=True))
  m = sum(ms)

def normal1():
  m = stocPy.normal(0, 1, obs=True)
  stocPy.normal(m, 1, obs)

def normal2():
  m = stocPy.normal(0, 1, obs=True)
  v = stocPy.invGamma(3, 1)
  cond = None
  if init:
    cond = obs
  stocPy.normal(m, math.sqrt(v), cond)

def normal3():
  global ms
  global vs
  m = stocPy.normal(0, 1, obs=True)
  if m > 0:
    v = 1
  else:
    v = stocPy.invGamma(3, 1)
    vs.append(v)
  cond = None
  if init:
    cond = obs
  stocPy.normal(m, math.sqrt(v), cond)

  ms.append(m)

def normal4():
  global ms
  global vs
  m = stocPy.normal(0, 1, obs=True)
  if m > 0:
    v = 1.0/3
  else:
    v = stocPy.invGamma(3, 1)
    vs.append(v)
  cond = None
  if init:
    cond = obs
  stocPy.normal(m, math.sqrt(v), cond)

  ms.append(m)

def normal5():
  m = stocPy.normal(0, 1, obs=True)
  for datum in normalData:
    stocPy.normal(m, 1, datum)

def normal6():
  m = stocPy.normal(0, 10, obs=True)
  for datum in normalData:
    stocPy.normal(m, 1, datum)

def normal7():
  m = stocPy.unifCont(0, 100, obs=True)
  for datum in normalData:
    stocPy.normal(m, 1, datum)

def normal7Part():
  m = stocPy.stocPrim("uniform", (0, 100), obs=True, part=10)
  for datum in normalData:
    stocPy.normal(m, 1, datum)

def normal8Old():
  m = stocPy.unifCont(0, 10000, obs=True)
  for datum in normalData:
    stocPy.normal(m, 1, datum)

def normal8():
  m = stocPy.stocPrim("uniform", (0, 10000), obs=True)
  for datum in normalData:
    stocPy.normal(m, 1, datum)

def normal8Part():
  m = stocPy.stocPrim("uniform", (0, 10000), obs=True, part=stocPy.stocPrim("randint", (0, 21)))
  for datum in normalData:
    stocPy.normal(m, 1, datum)

def normal8Part2():
  m = stocPy.stocPrim("uniform", (0, 10000), obs=True, part=10)
  for datum in normalData:
    stocPy.normal(m, 1, datum)

def normal9(): #prior = post
  m = stocPy.normal(0, 1, obs=True)

def normalVenture(model, v, sample, burn = 0, lag = 1):
  if model == 1:
    v.assume("m", "(normal 0 1)")
    v.observe("(normal m 1)", str(obs))
  elif model == 2:
    v.assume("m", "(normal 0 1)")
    v.assume("v", "(inv_gamma 3 1)")
    v.observe("(normal m (power v 0.5))", str(obs))
  elif model == 3:
    v.assume("m", "(normal 0 1)")
    v.assume("v", "(if (> m 0) 1 (inv_gamma 3 1))")
    v.observe("(normal m (power v 0.5))", str(obs))

  samples = stocPy.posterior_samples(v, "m" ,sample, burn, lag)

  vals = map(lambda x:x[1], samples)
  plt.hist(vals,100)
  print "Sample mean: ", np.mean(vals), " Sample Stdev: ", np.std(vals)
  
  hist = {}
  for v in vals:
    try:
      hist[v] += 1
    except:
      hist[v] = 1

  print sorted(hist.items())
  plt.show()
  #stocPy.save_samples(samples, os.getcwd(), "cont600")

def getLikAprox(model, m, data = None):
  if model == 1:
    return ss.norm.pdf(obs, m, 1) * ss.norm.pdf(m, 0, 1)
  elif model == 2:
    pdm = 0
    for v in np.arange(0.1,100,0.1):
      pdm += ss.norm.pdf(obs, m, math.sqrt(v)) * ss.invgamma.pdf(v, 3, 0, 1)
    return pdm * ss.norm.pdf(m, 0, 1)
  elif model == 3:
    pdm = 0
    if m < 0:
      norm = 0
      for v in np.arange(0.1,100,0.1):
        pv = ss.invgamma.pdf(v, 3, 0, 1)
        pdm += ss.norm.pdf(obs, m, math.sqrt(v)) * pv
        norm += pv
      pdm /= norm
    else:
      pdm = ss.norm.pdf(obs, m, 1)
    return pdm * ss.norm.pdf(m, 0, 1)
  elif model == 4:
    pdm = 0
    if m < 0:
      norm = 0
      for v in np.arange(0.1,100,0.1):
        pv = ss.invgamma.pdf(v, 3, 0, 1)
        pdm += ss.norm.pdf(obs, m, math.sqrt(v)) * pv
        norm += pv
      pdm /= norm
    else:
      pdm = ss.norm.pdf(obs, m, math.sqrt(1.0/3))
    return pdm * ss.norm.pdf(m, 0, 1)
  elif model == 5: # return LL instead
    pm = ss.norm.logpdf(m, 0, 1)
    pdm = 0
    for datum in data:
      pdm += ss.norm.logpdf(datum, m, 1)
    return pdm + pm
  elif model == 6: # return LL instead
    pm = ss.norm.logpdf(m, 0, 10)
    pdm = 0
    for datum in data:
      pdm += ss.norm.logpdf(datum, m, 1)
    return pdm + pm
  elif model == 7: # return LL instead
    pm = ss.uniform.logpdf(m, 0, 100)
    pdm = 0
    for datum in data:
      pdm += ss.norm.logpdf(datum, m, 1)
    return pdm + pm
  elif model == 8: # return LL instead
    pm = ss.uniform.logpdf(m, 0, 10000)
    pdm = 0
    for datum in data:
      pdm += ss.norm.logpdf(datum, m, 1)
    return pdm + pm
  elif model == 9: # this is just a gaussian(0,1)
    return ss.norm.pdf(m, 0, 1)
  else:
    raise Exception("Unknown model type: " + str(model))

def getLikExact(model, m):
  if model == 1:
    return ss.norm.pdf(m, obs/2.0, math.sqrt(0.5))
  elif model == 2:
    pass
  elif model == 3:
    pass
  else:
    raise Exception("Unknown model type: " + str(model))

def getPost(model, start, end, inc, aprox=True, show = True, fn=None, rfn = None, data=None):
  xs = []
  ys = []
  if rfn:
    with open(rfn,'r') as f:
      xs, ys = cPickle.load(f)
  else:
    for m in np.arange(start, end+inc, inc):
      xs.append(m)
      if aprox:
        ys.append(getLikAprox(model, m, data=data))
      else:
        ys.append(getLikExact(model, m))

    if aprox and (model < 5 or model == 9):
      ys = stocPy.norm(ys)
    else:
      print ys
      corr = 1500
      print [math.e**(y + corr) for y in ys]
      ys = stocPy.norm([math.e**(y + corr) for y in ys])
  print sum(ys)
  if show:
    plt.plot(xs,ys, linewidth=3)
    plt.ylabel("Probability", size=20)
    plt.xlabel("x", size=20)
    #plt.title("True Posterior for NormalMean5 model", size=30)
    plt.title("True Posterior for NormalMean" + str(model) + " model", size=30)
    plt.show()
  if fn:
    with open(fn,'w') as f:
      cPickle.dump((xs,ys),f)

def genRuns(model, alg, noRuns = 100, length = 20000, thresh=0.1, fn=None, agg=None, name=None, time=None):

  if time:
    length = time
    stocPyFunc = stocPy.getTimedSamples
  else:
    stocPyFunc = stocPy.getSamplesByLL

  if model == normal1:
    name = 'normal1-6-0'
    no = 1
  elif model == normal2:
    name = 'normal2-13-0'
    no = 2
  elif model == normal3:
    name = 'normal3-22-0'
    no = 3
  elif model == normal4:
    name = 'normal4-49-0'
    no = 4
  elif model == normal1Dec:
    name = None
    no = 1
  elif model == normal2Dec1 or model == normal2Dec2:
    name = None
    no = 2
  elif model == normal4Dec1 or model == normal4Dec2:
    name = None
    no = 4
  else:
    assert(fn)

  runs = []
  for i in range(noRuns):
    print "Run", i
    if name:
      runs.append(stocPyFunc(model, length, alg=alg, thresh=thresh)[name])
    else:
      if agg:
        samples = stocPy.aggDecomp(stocPyFunc(model, length, alg=alg, thresh=thresh), func=agg)
        runs.append(samples)
      else:
        samples = stocPyFunc(model, length, alg=alg, thresh=thresh)
        assert(len(samples.keys()) == 1)
        runs.append(samples[samples.keys()[0]])


  if not fn:
    fn = "normal/normal"+str(no)+"PerLL" + alg.capitalize()
    if alg == "sliceMet":
      fn += str(thresh)
    if name == None:
      fn += "Dec"

  with open(fn,'w') as f:
    cPickle.dump(runs, f)

def procRuns(no, alg):
  with open("normal/normal"+str(no)+"PerLL" + alg.capitalize(), 'r') as f:
    stocPy.plotSamples(cPickle.load(f))

def genData(no, m=0, v=1, fn=None):
  data = ss.norm.rvs(loc=m, scale=math.sqrt(v), size=no)
  if fn:
    with open(fn, 'w') as f:
      cPickle.dump(data, f)
  return data

def loadData(fn):
  with open(fn, 'r') as f:
    return cPickle.load(f)

def runKsTest(mi):
  mi = str(mi)
  expDir = stocPy.getCurDir(__file__) + "/"
  paths = ["normal" + mi + "Met3600", "normal" + mi + "Dec5Met600", "normal" + mi + "Dec10Met600", "normal" + mi + "Dec15Met600"]
  titles = ["Metropolis 60m", "Metropolis Dec5 10m", "Metropolis Dec10 10m", "Metropolis Dec15 10m"]
  paths = [expDir + path for path in paths]

  #stocPy.calcKSTest(expDir + "normal" + mi + "Post", paths, names = titles)
  stocPy.calcKSTests(expDir + "normal" + mi + "Post", paths, aggFreq=np.logspace(1,math.log(50000,10),10), burnIn=1000, single=True, alpha=1, names=titles, ylim=[0.0225, 1.5], title="Single Run performance on NormalMean" + mi +"-prior = Unif(0,10000)")

def runKsRuns(mi, term = "MetRuns"):
  mi = str(mi)

  expDir = stocPy.getCurDir(__file__) + "/"

  paths = ["normal" + mi + term, "normal" + mi + "Dec5" + term, "normal" + mi + "Dec10" + term, "normal" + mi + "Dec15" + term]
  titles = ["Metropolis", "Metropolis Dec5", "Metropolis Dec10", "Metropolis Dec15"]
  paths = [expDir + path for path in paths]

  #stocPy.calcKSTest(expDir + "normal" + mi + "Post", paths, names = titles)
  #stocPy.calcKSTests(expDir + "normal" + mi + "Post", paths, aggFreq=np.logspace(1,math.log(2000,10),10), burnIn=100, postXlim = [0,10000], names=titles)
  stocPy.calcKSSumms(expDir + "normal" + mi + "Post", paths, aggFreq=np.logspace(1,math.log(100000,10),10), burnIn=1000, names=titles, modelName = "NormalMean" + mi)

def movementFromMode((priorMean, priorStd), (postMean, postStd), curSamp, iters):
  move = []
  for prop in ss.norm.rvs(loc=priorMean, scale=priorStd, size=iters):
    move.append((prop, min(1, ss.norm.pdf(prop, loc=postMean, scale=postStd) / ss.norm.pdf(curSamp, loc=postMean, scale=postStd))))
  return move

def testMovement(ds, dFunc, aFunc, iters, title, priorStd = 1.0): # test if movement prob (transformed by dFunc) follows aFunc
  probs = map(lambda p: dFunc(movementFromMode((0, priorStd), (0, p), 0, iters), p), [priorStd/d for d in ds])
  rProbs = []
  rDs = np.logspace(math.log(ds[0],10),math.log(ds[-1],10), 500)
  for i in rDs:
    rProbs.append(aFunc(i))
  plt.plot(rDs,rProbs, 'r')
  plt.plot(ds, probs, 'bD')
  plt.xscale("log")
  plt.xlabel("priorStd / postStd")
  plt.title(title)
  plt.show()

def testMovementProb(ds, iters, title=""): # test if movement prob follows 1/sqrt(d^2 + 1)
  testMovement(ds, lambda xs, p: np.mean(map(lambda x : x[1], xs)), lambda d: 1/math.sqrt(d**2 + 1), iters, title=title)

def testMovementExpectation(ds, iters, priorStd = 1.0, title=""): # test if E(movement) follows (sqrt(2/pi) * d) / (d^2 + 1)
  testMovement(ds, lambda xs, p: np.mean(map(lambda x : (abs(x[0])/p)*x[1], xs)), lambda d: (math.sqrt(2/math.pi) * d) / (d**2 + 1), iters, title=title, priorStd = priorStd)

def testMovementDiff(rs, priorStd, postStd, iters, title=""): # test if perf. diff given partition diff r is (r^2 + 1)/2r
  rDiffs = []
  diffs = []
  d = float(priorStd) / postStd
  for r in rs:
    p2 = d/r
    m1 = np.mean([(abs(x[0])/postStd)*x[1] for x in movementFromMode((0, priorStd/d), (0, postStd), 0, iters)])
    m2 = np.mean([(abs(x[0])/postStd)*x[1] for x in movementFromMode((0, priorStd/p2), (0, postStd), 0, iters)])
    print r, p2, m1, m2, m1/m2
    diffs.append(m1/m2)    

  nRs = np.logspace(math.log(rs[0],10),math.log(rs[-1],10), 500)
  for r in nRs:
    rDiffs.append((r**2 + 1) / (2*r))

  diffs.reverse() # otherwise it would be part2 / optimalPart
  plt.plot(nRs, rDiffs, 'r') 
  plt.plot(rs, diffs, 'bD')
  plt.xscale("log")
  plt.xlabel("optimalPartition / partition2")
  plt.ylabel("optimalMovement / movement2")
  plt.title(title)
  #plt.ylim([0,100])
  plt.show()

def testMovementDiffExp(priorStd, postStd, iters, title=""): # test if perf. diff given exp partition diff c is (2^(2c) + 1) / (2^(c+1))
  nRs = np.logspace(-3,3, 500)
  rDiffs = []  
  for r in nRs:
    rDiffs.append((r**2 + 1) / (2*r))

  eDiffs = []
  eCs = range(-9,10)
  eRs = []
  for c in eCs:
    eDiffs.append((2**(2.0*c) + 1.0) / (2**(c+1.0)))
    eRs.append(2.0**c)

  diffs = []
  d = float(priorStd) / postStd
  for c in eCs:
    p2 = d/(2.0**c)
    m1 = np.mean([(abs(x[0])/postStd)*x[1] for x in movementFromMode((0, priorStd/d), (0, postStd), 0, iters)])
    m2 = np.mean([(abs(x[0])/postStd)*x[1] for x in movementFromMode((0, priorStd/p2), (0, postStd), 0, iters)])
    print r, p2, m1, m2, m1/m2
    diffs.append(m1/m2)  

  diffs.reverse()
  print zip(eRs, eDiffs)
  plt.plot(nRs, rDiffs, 'r') 
  plt.plot(eRs, eDiffs, 'bD')
  plt.plot(eRs, diffs, 'g^')
  plt.xscale("log")
  plt.xlabel("optimalPartition / partition2")
  plt.ylabel("optimalMovement / movement2")
  plt.title(title)
  #plt.ylim([0,100])
  plt.show()

def testMovementExpConv(priorStd, postStd, iters, title="", norm=False): # test convergence of movement perf as # partitions around optimum increase
  parts = [0]
  mov = [0]
  for p in range(11):
    parts.append(parts[-1] + 1)
    mov.append(mov[-1] + (2**(p+1.0)/(2**(2.0*p) + 1)))
    if p > 0:
      parts.append(parts[-1] + 1)
      mov.append(mov[-1] + (2**(p+1.0)/(2**(2.0*p) + 1)))

  limit = 2*math.pi / math.log(4)
  if norm:
    mov = map(lambda x: x/limit, mov)
    limit = 1
  plt.plot([0, parts[-1]+1], [limit, limit], 'r', linewidth=3)
  plt.plot(parts, mov, 'bD')
  plt.xlim([0, parts[-1]+1])
  plt.xlabel("Partitions")
  if norm:
    plt.ylabel("Prop Maximal Movement")
    plt.title("Normalized convergence of movement as number of paritions increases")
  else:
    plt.ylabel("Total Movement / Optimal Parition Movement")
    plt.title("Convergence of movement as number of paritions increases")
  print zip(parts, mov)
  plt.show()

def verifyMovementExpDepth(priorStd, postStd, maxDepth, iters, title=""): # test performance of depth given d follows formula
  diffs = []
  rDiffs = []
  mds = []
  for depth in range(maxDepth + 1):
    diffs.append(0)
    rDiffs.append(0)
    mds.append(depth)
    for d in range(1, depth + 1):
      em = np.mean([(abs(x[0])/postStd)*x[1] for x in movementFromMode((0, priorStd/(2.0**d)), (0, postStd), 0, iters)])
      print d, "/", depth, "-", em
      diffs[-1] += em
      dp = (priorStd/postStd) / (2.0**d)
      rDiffs[-1] += (math.sqrt(2.0/math.pi) * dp) / (dp**2.0 + 1) 
    dp = (priorStd/postStd) / 2.0**depth
    rDiffs[-1] += (math.sqrt(2.0/math.pi) * dp) / (dp**2.0 + 1)
    rDiffs[-1] /= (depth + 1.0)
    diffs[-1] += np.mean([(abs(x[0])/postStd)*x[1] for x in movementFromMode((0, priorStd/(2.0**depth)), (0, postStd), 0, iters)])
    diffs[-1] /= (depth + 1.0)

  plt.plot(mds, diffs, 'bD')
  plt.plot(mds, rDiffs, 'rx', markersize=12)
  plt.xlabel("Depth")
  plt.ylabel("Expected Movement / postStd")
  plt.title("Performance as number of partitions increases for priorStd/postStd=10")
  plt.show()

def getMixExpMovement(d, depths, weights = None):
  if not weights:
    weights = [1.0 for i in range(len(depths))]
  weights = stocPy.norm(weights)
  
  mix = 0
  for i in range(len(depths)):
    mix += weights[i] * getExpMovement(d, depths[i])

  return mix

def getExpMovement(d, depth):
  em = 0
  for p in range(1, depth + 1):
    dp = d / (2.0**p)
    em += (math.sqrt(2.0/math.pi) * dp) / (dp**2.0 + 1) 
  dp = d / 2.0**depth
  em += (math.sqrt(2.0/math.pi) * dp) / (dp**2.0 + 1)
  em /= (depth + 1.0)
  return em

def testMovementExpDepth(priorStd, postStd, maxDepth, title=""): # test performance of depth given d
  rDiffs = []
  mds = []
  for depth in range(maxDepth + 1):
    mds.append(depth)
    rDiffs.append(getExpMovement(priorStd/postStd, depth))

  print zip(mds, rDiffs)
  plt.plot(mds, rDiffs, 'bD')
  plt.xlabel("Depth")
  plt.ylabel("Expected Movement / postStd")
  plt.title("Performance as number of partitions increases for priorStd/postStd=10")
  plt.show()

"""
def testOptimalDepthNoTerm(priorStd, postStd): # get Optimal depth for d, verify against testMovementExpDepth
  depth = 1
  d = priorStd / postStd
  sumVal = 0
  while True:
    val = 2.0**depth / (d**2.0 + 2**(2.0 * depth))
    print depth, val, sumVal
    if sumVal == 0 or val > (sumVal/(depth-1)):
      sumVal += val
      depth +=1
    else:
      break
  print depth -1, math.log(d,2)
  propStop = max(1, math.ceil(math.log(d, 2))) + 1
  # if depth != propStop and depth != propStop-1:
  #   print d, depth, propStop 
"""
def getOptimalDepth(d, dump=False): # get Optimal depth for d, verify against testMovementExpDepth
  depth = 1
  sumVal = 0
  valPrev = 0
  while True:
    valCur = 2.0**depth / (d**2.0 + 2**(2.0 * depth))
    if sumVal == 0:
      sumVal += valCur
      depth +=1
      valPrev = valCur
    else:
      #left = 2.0/(depth+1) * (valCur - sumVal/(depth-1))
      #right = 1.0/depth * (valPrev - sumVal/(depth-1))
      cond = 2.0 * depth * valCur - (depth+1) * valPrev - sumVal
      #assert ((left > right) == (cond > 0)) 
      #print depth, valPrev, valCur, sumVal, left, right
      if sumVal == 0 or cond > 0:
        sumVal += valCur
        depth +=1
        valPrev = valCur
      else:
        break
  diff = abs(depth-1 - math.log(d,2))
  #print d, depth -1, math.log(d,2), diff
  if dump:
    return (diff, d, depth-1, math.log(d,2))
  else:
    return depth-1
  #propStop = max(1, math.ceil(math.log(d, 2))) + 1
  # if depth != propStop and depth != propStop-1:
  #   print d, depth, propStop

def approxOptimalDepth(ds):
  depths = []
  for d in ds:
    depths.append(getOptimalDepth(d))

  frs = np.logspace(math.log(ds[0],10), math.log(ds[-1],10), 500)
  rDepths1 = []
  rDepths2 = []
  for fr in frs:
    rDepths1.append(math.log(fr,2))
    rDepths2.append(math.ceil(rDepths1[-1]))

  plt.plot(ds, depths, 'bD')
  plt.plot(frs, rDepths2, 'g', linewidth=2)
  plt.plot(frs, rDepths1, 'r', linewidth=2)
  plt.xscale("log")
  plt.xlabel("PriorStd / PostStd")
  plt.ylabel("Depth")
  plt.title("Optimal depths given d against log(d) and ceil(log(d))")
  plt.show()

def testOptExpImprovement(ds):
  imp = []
  df = []
  dr1 = []
  dr2 = []
  dr3 = []
  dr4 = []
  for d in ds:
    imp.append(getExpMovement(d, getOptimalDepth(d)))
    df.append(getExpMovement(d, 0))
    dr1.append(getExpMovement(d, 5))
    dr2.append(getExpMovement(d, 10))
    dr3.append(getExpMovement(d, 20))
    dr4.append(getMixExpMovement(d, [1,2,4,8,16,32,64]))

  ax1, = plt.plot(ds, imp, 'bD')
  ax2, = plt.plot(ds, df, 'r^', markersize=8)
  ax3, = plt.plot(ds, dr1, 'kh')
  ax4, = plt.plot(ds, dr2, 'gd')
  ax5, = plt.plot(ds, dr3, 'm*')
  ax6, = plt.plot(ds, dr4, 'co')
  plt.xscale("log")
  plt.yscale("log")
  plt.legend([ax1,ax2,ax3,ax4,ax5,ax6],["Optimal Depth", "Depth 0", "Depth 5", "Depth 10", "Depth 20", "Depth = 2^1..2^6"], loc=3)
  plt.xlabel("PriorStd / PostStd")
  plt.ylabel("Movement / PostStd")
  plt.title("Expected movement for optimal depth and several fixed depths")
  plt.show()
  
if __name__ == "__main__":
  global normalData
  normalData = loadData(stocPy.getCurDir(__file__) + "/normalData_2_001_1000")
  
  #getPost(9, -15, 15, 0.001, fn="normal9Post")
  #samples = stocPy.aggDecomp(stocPy.getTimedSamples(normal9Dec5, 10, alg="met", thresh=0.1), func= lambda xs: 10000 * ss.norm.cdf(sum(xs)))

  #samples = stocPy.getTimedSamples(normal8Part, 5, alg="met", thresh=0.1)
  #print samples
  #samples = stocPy.aggDecomp(stocPy.getTimedSamples(normal9Dec15, 1, alg="met", thresh=0.1))
  #stocPy.plotSamples(samples)
  #print samples
  #stocPy.saveRun(samples, "normal8U20_600")

  #stocPy.calcKSTests("normal8Post", ["./normal8U20_600"], aggFreq=np.logspace(1,math.log(50000,10),10), burnIn=1000, single=True, alpha=1, names=["gigi"])

  #with open("normal8Dec15MetRunsCorr", 'r') as f:
  #  print map(lambda x: (min(x.values()), max(x.values())), cPickle.load(f))

  #with open("normal8U20Timed10", 'r') as f:
  #  runs = [dict([(k, 10000 * ss.norm.cdf(v)) for (k,v) in run.items()]) for run in cPickle.load(f)]
 
  #with open("normal8U20Timed10Corr",'r') as f:
  #  runs = cPickle.load(f)
  #print map(len, runs)
  #print map(lambda r: sorted(r.items())[-10:], runs)
  """
  for name in ["Dec15"]:
    print name
    with open("normal9" + name + "Timed90", 'r') as fr:
      with open("normal9" + name + "Timed10", 'w') as fw:
        runs = cPickle.load(fr)
        print map(len, runs)
        smallRuns = []
        for run in runs:
          lim = len(run) / 9
          smallRuns.append(dict([(k,v) for (k,v) in run.items() if k <= lim]))
        cPickle.dump(smallRuns, fw)
        print map(len, smallRuns)
  """
  genRuns(normal8Part, "met", noRuns = 20, fn="normal8PartU20Timed10", time=600)
  #genRuns(normal8Part2, "met", noRuns = 20, fn="normal8Part10Timed10", time=600)
  #genRuns(normal8Old, "met", noRuns = 20, fn="normal8Timed10", time=600)
  #genRuns(normal7, "met", noRuns = 15, fn="normal7Timed1", time=60)
  #genRuns(normal8DecU20, "met", noRuns = 100, fn="normal8U20Timed5", agg=lambda xs: 10000 * ss.norm.cdf(sum(xs)), time=300)
  #genRuns(normal9Dec5, "met", noRuns = 100, fn="normal9Dec5Timed90", agg=True, time=90)
  #genRuns(normal9Dec10, "met", noRuns = 100, fn="normal9Dec10Timed90", agg=True, time=90)
  #genRuns(normal9Dec15, "met", noRuns = 100, fn="normal9Dec15Timed90", agg=True, time=90)
  #runKsRuns(9, term="Timed10")

  #testMovementProb(np.logspace(-5,5,50), 1000,"Probability of moving from mode given difference between prior and post")
  #testMovementExpectation(np.logspace(-5,5,50), 2000, priorStd = 0.3, title="Expected prop postStd moved from mode given difference between prior and post")
  #testMovementDiff(np.logspace(-3,3,50), 10, 1, iters = 1000, title="Expected optimalMovement/movement2 given optimalPartition/partition2")
  #testMovementDiffExp(10, 1, iters = 1000, title="Expected optimalMovement/movement2 given optimalPartition/partition2")
  #testMovementExpConv(10, 1, iters = 1000, title="", norm=True)  
  #getOptimalDepth(100)
  #testMovementExpDepth(33, 1, 9, title="")
  #diffs = []
  #for prior in np.arange(1, 10000000001, 1000000):
  #  diffs.append(getOptimalDepth(prior, dump=True))
  #print max(diffs)
  #approxOptimalDepth(np.logspace(0,9,100))
  #testOptExpImprovement(np.logspace(-6,10,100))

  #with open("normal8PartTimed5",'r') as f:
  #  runs = cPickle.load(f)
  #  print map(lambda r:sorted(r.items())[-10:], runs)
    #print runs
  #with open(stocPy.getCurDir(__file__) + "/normal8Post", 'r') as f:
  #  print cPickle.load(f)
  stocPy.calcKSSumms("./normal8Post", ["./normal8Timed10", "./normal8Part10Timed10", "./normal8PartU20Timed10"], aggFreq=np.logspace(1,math.log(100000,10),10), burnIn=0, names=["Depth0", "Depth10", "Depth_Unif(0,20)"], modelName = "NormalMean", postXlim=[0,10000], ylim=[2.0**(-3),1.1], xlim=10000)
  assert(False)
  samples = stocPy.readSamps("normal8Met600")
  print samples[:100]
  print len(filter(lambda x: x>10, samples))
  #stocPy.plotSamples(samples, filt=lambda x:x>1.5 and x<2.5)
  #print stocPy.readSamps("normal7Dec5Met600")

  #assert(False)  
  runKsTest(7)  
  
  #print genData(1000, m=2, v=0.01, fn="normalData_2_001_1000")
  #getPost(7, 0, 10, 0.1, data=normalData, fn="normal8Post")
  #with open("normal5Post", 'r') as f:
  #  print cPickle.load(f)
  assert(False)

  samples = stocPy.aggDecomp(stocPy.getTimedSamples(normal7Dec10, 600, alg="met", thresh=0.1), func= lambda xs: 100 * ss.norm.cdf(sum(xs)))
  #samples = stocPy.getTimedSamples(normal7, 600, alg="met", thresh=0.1)
  print samples
  stocPy.saveRun(samples, "normal7Dec10Met600")

  samples = stocPy.aggDecomp(stocPy.getTimedSamples(normal7Dec15, 600, alg="met", thresh=0.1), func= lambda xs: 100 * ss.norm.cdf(sum(xs)))
  #samples = stocPy.getTimedSamples(normal7, 600, alg="met", thresh=0.1)
  print samples
  stocPy.saveRun(samples, "normal7Dec15Met600")

  samples = stocPy.getTimedSamples(normal8, 600, alg="met", thresh=0.1)
  print samples
  stocPy.saveRun(samples, "normal8Met600")

  samples = stocPy.aggDecomp(stocPy.getTimedSamples(normal8Dec5, 600, alg="met", thresh=0.1), func= lambda xs: 10000 * ss.norm.cdf(sum(xs)))
  #samples = stocPy.getTimedSamples(normal7, 600, alg="met", thresh=0.1)
  print samples
  stocPy.saveRun(samples, "normal8Dec5Met600")

  samples = stocPy.aggDecomp(stocPy.getTimedSamples(normal8Dec10, 600, alg="met", thresh=0.1), func= lambda xs: 10000 * ss.norm.cdf(sum(xs)))
  #samples = stocPy.getTimedSamples(normal7, 600, alg="met", thresh=0.1)
  print samples
  stocPy.saveRun(samples, "normal8Dec10Met600")

  samples = stocPy.aggDecomp(stocPy.getTimedSamples(normal8Dec15, 600, alg="met", thresh=0.1), func= lambda xs: 10000 * ss.norm.cdf(sum(xs)))
  #samples = stocPy.getTimedSamples(normal7, 600, alg="met", thresh=0.1)
  print samples
  stocPy.saveRun(samples, "normal8Dec15Met600")

  getPost(8, 0, 10, 0.01, data=normalData, fn="normal8Post")

  #stocPy.plotSamples(samples, title="Normal7Dec5 Metropolis", xlabel = "Mean")
  #procRuns(2, "met")
  #genRuns(normal4, "slice", 25, 200000)
  #print stocPy.aggDecomp(stocPy.getSamples(normal1Dec, 100, alg="met"))
  #genRuns(normal4Dec2, "met", 100, 200000, fn="normal/normal4PerLLMetDec44")
  """
  genRuns(normal1, "met", 100, 20000)
  genRuns(normal2, "met", 100, 20000)
  genRuns(normal3, "met", 100, 20000)

  genRuns(normal1, "slice", 100, 20000)
  genRuns(normal2, "slice", 100, 20000)
  genRuns(normal3, "sliceNoTrans", 100, 20000)

  genRuns(normal1, "sliceMet", 100, 20000)
  genRuns(normal2, "sliceMet", 100, 20000)
  genRuns(normal3, "sliceMet", 100, 20000)

  genRuns(normal1, "sliceMet", 100, 20000, thresh=0.5)
  genRuns(normal2, "sliceMet", 100, 20000, thresh=0.5)
  genRuns(normal3, "sliceMet", 100, 20000, thresh=0.5)

  genRuns(normal4, "met", 25, 200000)
  genRuns(normal4, "slice", 25, 200000)
  genRuns(normal4, "sliceMet", 25, 200000, thresh=0.1)
  genRuns(normal4, "sliceMet", 25, 200000, thresh=0.5)
  genRuns(normal4, "sliceNoTrans", 25, 200000)
  """

  #genRuns(normal4, "slice", 25, 200000, fn="normal/normal4PerLLSliceV2")

  #genRuns(normal3, "slice", 100, 20000)

  #with open("normal/normal4PerLLSliceV2",'r') as f:
  #  runs1 = cPickle.load(f)
  #print map(len, runs1)

  #print np.mean(map(len,runs))
  #print stocPy.calcKSDiff(stocPy.getPostFun("normal/normal4Post"), runs[6].values())
  #stocPy.plotSamples(runs[5])
  #stocPy.plotSamples(stocPy.getSamplesByLL(normal4, 100000, alg="sliceMet"))
  """
  stocPy.getSamples(normal3, 100000, alg="met")
  print len(vs)
  vs = filter(lambda v: v<=3, vs)
  plt.hist(vs, 100, normed=True)
  plt.xlim([0,3])
  plt.show()
  """
  #getPost(4, -4, 7, 0.01, rfn = "normal/normal4Post")

  #with open("normal/normal4Post",'r') as f:
  #  print cPickle.load(f)

  #v = make_church_prime_ripl()
  #normalVenture(3, v, 5000, burn = 10000, lag = 10)

  #for v in np.arange(0.25,100,1):ist
  #  print v, ss.invgamma.pdf(v, 2, scale = 1)

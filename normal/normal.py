import sys
sys.path.append("/home/haggis/Desktop/StocPyDev/")
import stocPy
import math
import scipy.stats as ss
import numpy as np
from matplotlib import pyplot as plt
import stocPy
import ppUtils as pu
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

def normal5Dec():
  m1 = stocPy.normal(0, math.sqrt(0.5), obs=True)
  m2 = stocPy.normal(0, math.sqrt(0.25), obs=True)
  m3 = stocPy.normal(0, math.sqrt(0.125), obs=True)
  m4 = stocPy.normal(0, math.sqrt(0.0625), obs=True)
  m5 = stocPy.normal(0, math.sqrt(0.0625), obs=True)
  m = m1+m2+m3+m4+m5
  for datum in normalData:
    stocPy.normal(m, 1, datum)

def normal6Dec5():
  m1 = stocPy.normal(0, math.sqrt(5), obs=True)
  m2 = stocPy.normal(0, math.sqrt(2.5), obs=True)
  m3 = stocPy.normal(0, math.sqrt(1.25), obs=True)
  m4 = stocPy.normal(0, math.sqrt(0.625), obs=True)
  m5 = stocPy.normal(0, math.sqrt(0.625), obs=True)
  m = m1+m2+m3+m4+m5
  for datum in normalData:
    stocPy.normal(m, 1, datum)

def normal6Dec10():
  var = 10.0
  m1 = stocPy.normal(0, math.sqrt(10.0/2), obs=True)
  m2 = stocPy.normal(0, math.sqrt(10.0/4), obs=True)
  m3 = stocPy.normal(0, math.sqrt(10.0/8), obs=True)
  m4 = stocPy.normal(0, math.sqrt(10.0/16), obs=True)
  m5 = stocPy.normal(0, math.sqrt(10.0/32), obs=True)
  m6 = stocPy.normal(0, math.sqrt(10.0/64), obs=True)
  m7 = stocPy.normal(0, math.sqrt(10.0/128), obs=True)
  m8 = stocPy.normal(0, math.sqrt(10.0/256), obs=True)
  m9 = stocPy.normal(0, math.sqrt(10.0/512), obs=True)
  m10 = stocPy.normal(0, math.sqrt(10.0/512), obs=True)
  m = m1+m2+m3+m4+m5+m6+m7+m8+m9+m10

  for datum in normalData:
    stocPy.normal(m, 1, datum)

def normal6Dec15():
  var = 10.0
  m1 = stocPy.normal(0, math.sqrt(10.0/2), obs=True)
  m2 = stocPy.normal(0, math.sqrt(10.0/4), obs=True)
  m3 = stocPy.normal(0, math.sqrt(10.0/8), obs=True)
  m4 = stocPy.normal(0, math.sqrt(10.0/16), obs=True)
  m5 = stocPy.normal(0, math.sqrt(10.0/32), obs=True)
  m6 = stocPy.normal(0, math.sqrt(10.0/64), obs=True)
  m7 = stocPy.normal(0, math.sqrt(10.0/128), obs=True)
  m8 = stocPy.normal(0, math.sqrt(10.0/256), obs=True)
  m9 = stocPy.normal(0, math.sqrt(10.0/512), obs=True)
  m10 = stocPy.normal(0, math.sqrt(10.0/1024), obs=True)
  m11 = stocPy.normal(0, math.sqrt(10.0/2048), obs=True)
  m12 = stocPy.normal(0, math.sqrt(10.0/4096), obs=True)
  m13 = stocPy.normal(0, math.sqrt(10.0/8192), obs=True)
  m14 = stocPy.normal(0, math.sqrt(10.0/16384), obs=True)
  m15 = stocPy.normal(0, math.sqrt(10.0/16384), obs=True)
  m = m1+m2+m3+m4+m5+m6+m7+m8+m9+m10+m11+m12+m13+m14+m15

  for datum in normalData:
    stocPy.normal(m, 1, datum)

def normal7Dec5():
  m1 = stocPy.normal(0, math.sqrt(1.0/(2**1)), obs=True)
  m2 = stocPy.normal(0, math.sqrt(1.0/(2**2)), obs=True)
  m3 = stocPy.normal(0, math.sqrt(1.0/(2**3)), obs=True)
  m4 = stocPy.normal(0, math.sqrt(1.0/(2**4)), obs=True)
  m5 = stocPy.normal(0, math.sqrt(1.0/(2**4)), obs=True)
  m = 100 * ss.norm.cdf(m1+m2+m3+m4+m5, loc = 0, scale = 1)
  for datum in normalData:
    stocPy.normal(m, 1, datum)

def normal7Dec10():
  m1 = stocPy.normal(0, math.sqrt(1.0/(2**1)), obs=True)
  m2 = stocPy.normal(0, math.sqrt(1.0/(2**2)), obs=True)
  m3 = stocPy.normal(0, math.sqrt(1.0/(2**3)), obs=True)
  m4 = stocPy.normal(0, math.sqrt(1.0/(2**4)), obs=True)
  m5 = stocPy.normal(0, math.sqrt(1.0/(2**5)), obs=True)
  m6 = stocPy.normal(0, math.sqrt(1.0/(2**6)), obs=True)
  m7 = stocPy.normal(0, math.sqrt(1.0/(2**7)), obs=True)
  m8 = stocPy.normal(0, math.sqrt(1.0/(2**8)), obs=True)
  m9 = stocPy.normal(0, math.sqrt(1.0/(2**9)), obs=True)
  m10 = stocPy.normal(0, math.sqrt(1.0/(2**9)), obs=True)
  m = 100 * ss.norm.cdf(m1+m2+m3+m4+m5+m6+m7+m8+m9+m10, loc = 0, scale = 1)
  for datum in normalData:
    stocPy.normal(m, 1, datum)

def normal7Dec15():
  m1 = stocPy.normal(0, math.sqrt(1.0/(2**1)), obs=True)
  m2 = stocPy.normal(0, math.sqrt(1.0/(2**2)), obs=True)
  m3 = stocPy.normal(0, math.sqrt(1.0/(2**3)), obs=True)
  m4 = stocPy.normal(0, math.sqrt(1.0/(2**4)), obs=True)
  m5 = stocPy.normal(0, math.sqrt(1.0/(2**5)), obs=True)
  m6 = stocPy.normal(0, math.sqrt(1.0/(2**6)), obs=True)
  m7 = stocPy.normal(0, math.sqrt(1.0/(2**7)), obs=True)
  m8 = stocPy.normal(0, math.sqrt(1.0/(2**8)), obs=True)
  m9 = stocPy.normal(0, math.sqrt(1.0/(2**9)), obs=True)
  m10 = stocPy.normal(0, math.sqrt(1.0/(2**10)), obs=True)
  m11 = stocPy.normal(0, math.sqrt(1.0/(2**11)), obs=True)
  m12 = stocPy.normal(0, math.sqrt(1.0/(2**12)), obs=True)
  m13 = stocPy.normal(0, math.sqrt(1.0/(2**13)), obs=True)
  m14 = stocPy.normal(0, math.sqrt(1.0/(2**14)), obs=True)
  m15 = stocPy.normal(0, math.sqrt(1.0/(2**14)), obs=True)
  m = 100 * ss.norm.cdf(m1+m2+m3+m4+m5+m6+m7+m8+m9+m10+m11+m12+m13+m14+m15, loc = 0, scale = 1)
  for datum in normalData:
    stocPy.normal(m, 1, datum)

def normal8Dec5():
  m1 = stocPy.normal(0, math.sqrt(1.0/(2**1)), obs=True)
  m2 = stocPy.normal(0, math.sqrt(1.0/(2**2)), obs=True)
  m3 = stocPy.normal(0, math.sqrt(1.0/(2**3)), obs=True)
  m4 = stocPy.normal(0, math.sqrt(1.0/(2**4)), obs=True)
  m5 = stocPy.normal(0, math.sqrt(1.0/(2**4)), obs=True)
  m = 10000 * ss.norm.cdf(m1+m2+m3+m4+m5, loc = 0, scale = 1)
  for datum in normalData:
    stocPy.normal(m, 1, datum)

def normal8Dec10():
  m1 = stocPy.normal(0, math.sqrt(1.0/(2**1)), obs=True)
  m2 = stocPy.normal(0, math.sqrt(1.0/(2**2)), obs=True)
  m3 = stocPy.normal(0, math.sqrt(1.0/(2**3)), obs=True)
  m4 = stocPy.normal(0, math.sqrt(1.0/(2**4)), obs=True)
  m5 = stocPy.normal(0, math.sqrt(1.0/(2**5)), obs=True)
  m6 = stocPy.normal(0, math.sqrt(1.0/(2**6)), obs=True)
  m7 = stocPy.normal(0, math.sqrt(1.0/(2**7)), obs=True)
  m8 = stocPy.normal(0, math.sqrt(1.0/(2**8)), obs=True)
  m9 = stocPy.normal(0, math.sqrt(1.0/(2**9)), obs=True)
  m10 = stocPy.normal(0, math.sqrt(1.0/(2**9)), obs=True)
  m = 10000 * ss.norm.cdf(m1+m2+m3+m4+m5+m6+m7+m8+m9+m10, loc = 0, scale = 1)
  for datum in normalData:
    stocPy.normal(m, 1, datum)

def normal8Dec15():
  m1 = stocPy.normal(0, math.sqrt(1.0/(2**1)), obs=True)
  m2 = stocPy.normal(0, math.sqrt(1.0/(2**2)), obs=True)
  m3 = stocPy.normal(0, math.sqrt(1.0/(2**3)), obs=True)
  m4 = stocPy.normal(0, math.sqrt(1.0/(2**4)), obs=True)
  m5 = stocPy.normal(0, math.sqrt(1.0/(2**5)), obs=True)
  m6 = stocPy.normal(0, math.sqrt(1.0/(2**6)), obs=True)
  m7 = stocPy.normal(0, math.sqrt(1.0/(2**7)), obs=True)
  m8 = stocPy.normal(0, math.sqrt(1.0/(2**8)), obs=True)
  m9 = stocPy.normal(0, math.sqrt(1.0/(2**9)), obs=True)
  m10 = stocPy.normal(0, math.sqrt(1.0/(2**10)), obs=True)
  m11 = stocPy.normal(0, math.sqrt(1.0/(2**11)), obs=True)
  m12 = stocPy.normal(0, math.sqrt(1.0/(2**12)), obs=True)
  m13 = stocPy.normal(0, math.sqrt(1.0/(2**13)), obs=True)
  m14 = stocPy.normal(0, math.sqrt(1.0/(2**14)), obs=True)
  m15 = stocPy.normal(0, math.sqrt(1.0/(2**14)), obs=True)
  m = 10000 * ss.norm.cdf(m1+m2+m3+m4+m5+m6+m7+m8+m9+m10+m11+m12+m13+m14+m15, loc = 0, scale = 1)
  for datum in normalData:
    stocPy.normal(m, 1, datum)


def normal1():
  m = stocPy.normal(0, 1, obs=True)
  cond = None
  if init:
    cond = obs
  stocPy.normal(m, 1, cond)

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

def normal8():
  m = stocPy.unifCont(0, 10000, obs=True)
  for datum in normalData:
    stocPy.normal(m, 1, datum)

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

  samples = pu.posterior_samples(v, "m" ,sample, burn, lag)

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
  #pu.save_samples(samples, os.getcwd(), "cont600")

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

    if aprox and model < 5:
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

def genRuns(model, alg, noRuns = 100, length = 20000, thresh=0.1, fn=None, agg=False, name=None):

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
      runs.append(stocPy.getSamplesByLL(model, length, alg=alg, thresh=thresh)[name])
    else:
      if agg:
        samples = stocPy.aggDecomp(stocPy.getSamplesByLL(model, length, alg=alg, thresh=thresh))
        runs.append(samples)
      else:
        samples = stocPy.getSamplesByLL(model, length, alg=alg, thresh=thresh)
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

def runKsRuns(mi):
  mi = str(mi)
  expDir = stocPy.getCurDir(__file__) + "/"
  paths = ["normal" + mi + "MetRuns", "normal" + mi + "Dec5MetRuns", "normal" + mi + "Dec10MetRuns", "normal" + mi + "Dec15MetRuns"]
  titles = ["Metropolis", "Metropolis Dec5", "Metropolis Dec10", "Metropolis Dec15"]
  paths = [expDir + path for path in paths]

  #stocPy.calcKSTest(expDir + "normal" + mi + "Post", paths, names = titles)
  #stocPy.calcKSTests(expDir + "normal" + mi + "Post", paths, aggFreq=np.logspace(1,math.log(2000,10),10), burnIn=100, postXlim = [0,10000], names=titles)
  stocPy.calcKSSumms(expDir + "normal" + mi + "Post", paths, aggFreq=np.logspace(1,math.log(2000,10),10), burnIn=100, postXlim = [0,10000], names=titles, modelName = "NormalMean8")

if __name__ == "__main__": #9northerniighT
  global normalData
  normalData = loadData("normalData_2_001_1000")
  getPost(8, 0, 10, 0.01, data=normalData, fn="normal8Post")
  #samples = stocPy.getTimedSamples(normal8, 3600, alg="met", thresh=0.1)
  #print samples
  #stocPy.saveRun(samples, "normal8Met3600")

  #with open("normal8Dec15MetRunsCorr", 'r') as f:
  #  print map(lambda x: (min(x.values()), max(x.values())), cPickle.load(f))

  #with open("normal8Dec15MetRuns", 'r') as f:
  #  runs = [dict([(k, 10000 * ss.norm.cdf(v)) for (k,v) in run.items()]) for run in cPickle.load(f)]
  
  #with open("normal8Dec15MetRunsCorr", 'w') as f:
  #  cPickle.dump(runs, f)
  #genRuns(normal8Dec5, "met", noRuns = 100, length = 2000, fn="normal8Dec5MetRuns", agg=True)
  #genRuns(normal8Dec10, "met", noRuns = 100, length = 2000, fn="normal8Dec10MetRuns", agg=True)
  #genRuns(normal8Dec15, "met", noRuns = 100, length = 2000, fn="normal8Dec15MetRuns", agg=True)
  runKsRuns(8)
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

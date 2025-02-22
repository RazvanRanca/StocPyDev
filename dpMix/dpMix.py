import sys
sys.path.append("/home/haggis/Desktop/StocPyDev/")
import stocPyDev as stocPy
import random
import math
import scipy.stats as ss
import numpy as np
import copy
import itertools
import cPickle

obs = [1.0, 1.1, 1.2, -10.0, -15.0, -20.0, 0.01, 0.1, 0.05, 0.0]
post = {1: 0.04571063618028917, 2: 0.21363892248912586, 3: 0.32803178751362133, 4: 0.2536086916674492, 5: 0.1168560934038502, 6: 0.034586518747180765, 7: 0.006687390777635802, 8: 0.0008201140565686028, 9: 5.803774292549257e-05, 10: 1.8074213536324885e-06}

def allPartSizes(data):
  if len(data) == 1:
    return [[[data[0]]]]
  parts = []
  nParts = allParts(data[1:])
  for nPart in nParts:
    parts.append([[data[0]] + nPart[0]] + nPart[1:])
    parts.append([[data[0]]] + nPart)
  return parts

def addelement(partlist, e):
  newpartlist = []
  for part in partlist:
    npart = part + [[e]]
    newpartlist += [npart]
    for i in xrange(len(part)):
      npart = copy.deepcopy(part)
      npart[i] += [e]
      newpartlist += [npart]
  return newpartlist

def allParts(data):
  if len(data) == 0: 
    return []
  partlist = [[[data[0]]]]
  for i in xrange(1, len(data)):
    partlist = addelement(partlist, data[i])
  return map(lambda ps: map(tuple, ps), partlist)

def powerSet(data):
  ps = []
  for i in range(1,len(data)+1):
    ps += itertools.combinations(data, i)
  return ps

def testCrpProbs():
  crp = CRP(1.37)
  for i in range(1,20):
    print i, any(map(lambda x: abs(math.log(crp.pdf(x)) - crp.logpdf(x)) < 0.000001, allParts(range(i))))

llDict = None
def getLL(cl, fn = None):
  global llDict
  if fn:
    try:
      return llDict[cl]
    except:
      with open(fn, 'r') as f:
        llDict = cPickle.load(f)
      return llDict[cl]
  else:
    ll = math.log(ss.invgamma.expect(lambda v: ss.norm.expect(lambda m: np.product([ss.norm.pdf(d, loc=m, scale=10*v) for d in cl]) ,loc=0, scale=10*v), 1, loc=0, scale=10))
  return ll

def storeLLs(data, fn):
  lls = {}
  for cl in powerSet(data):
    lls[cl] = getLL(cl)
    print cl, lls[cl]
  with open(fn, 'w') as f:
    cPickle.dump(lls, f)

def getPost(ds, a, fn=None):
  crp = stocPy.CRP(a)
  parts = allParts(ds)
  post = {}

  for p in range(len(parts)):
    part = parts[p]
    ll = crp.logpmf(part)
    if p % 1000 == 0:
      print p, part, ll, 
    sys.stdout.flush()
    for cl in part:
      ll += getLL(cl, fn)
    try:
      post[len(part)].append(ll)
    except:
      post[len(part)] = [ll]
    if p % 1000 == 0:
      print ll
    sys.stdout.flush()
  post = dict([(k, sum(map(lambda x: math.e**x, v))) for (k, v) in post.items()])
  print post
  post = stocPy.norm(post)
  return post

obsLens = []
def dpmLazy():
  crp = stocPy.crp(1.72)
  sds = {}
  ms = {}
  for i in range(len(obs)):
    c = crp(i)
    if c not in ms:
      sds[c] = math.sqrt(10 * stocPy.stocPrim("invgamma", (1, 0, 10), part=4))
      ms[c] = stocPy.stocPrim("normal", (0, sds[c]), part=4)
    stocPy.normal(ms[c], sds[c], obs[i])
  obsLens.append(len(ms))

def dpmEager():
  crp = stocPy.crp(1.72, 10)
  sds = {}
  ms = {}
  cs = {}
  for ps in range(len(crp)):
    sds[ps] = math.sqrt(10 * stocPy.stocPrim("invgamma", (1, 0, 10), part=2))
    ms[ps] = stocPy.stocPrim("normal", (0, sds[ps]), part=2)
    for p in crp[ps]:
     cs[p] = ps

  for i in range(len(obs)):
    stocPy.normal(ms[cs[i]], sds[cs[i]], obs[i])
  obsLens.append(len(ms))

def genRuns(model, noRuns, time, fn, alg="met"):
  global obsLens
  runs = []
  for i in range(noRuns):
    print str(model), "Run", i
    samples, traceAcc = stocPy.getTimedSamples(model, time, alg=alg, outTraceAcc=True)
    runs.append(stocPy.procUserSamples(obsLens, traceAcc))
    obsLens = []
  cd = stocPy.getCurDir(__file__)
  print map(lambda run: (min(run.values()), max(run.values())), runs)
  with open(cd + "/" + fn, 'w') as f:
    cPickle.dump(runs, f)

if __name__ == "__main__":
  #print allParts(range(3))
  #print powerSet(range(3))
  #print len(powerSet(range(10)))
  #storeLLs(obs, "llDict")
  #for n in range(1,11):
  #  print len(allParts(range(n)))
  #print getPost(obs, 1.72, "dpMixLLS")
  #_, traceAcc = stocPy.getSamples(dpmLazy, 100, alg="met", outTraceAcc=True)
  #print stocPy.procUserSamples(obsLens, traceAcc)
  noRuns = 5
  runTime = 60
  term = "_" + str(noRuns) + "_" + str(runTime)

  genRuns(dpmLazy, noRuns, runTime, "Lazy_Met" + term, alg="met")
  genRuns(dpmEager, noRuns, runTime, "Eager_Met" + term, alg="met")
  cd = stocPy.getCurDir(__file__)
  stocPy.calcKLSumms(post , [cd + "Eager_Met" + term, cd + "Lazy_Met" + term], names = ["Eager", "Lazy"], burnIn=0, modelName="DP Mixture")

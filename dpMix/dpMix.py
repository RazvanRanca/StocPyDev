import sys
sys.path.append("/home/haggis/Desktop/StocPyDev/")
import stocPyDev as stocPy
import random
import math
import scipy.stats as ss
import numpy as np

obs = [1.0, 1.1, 1.2, -10.0, -15.0, -20.0, 0.01, 0.1, 0.05, 0.0]

def allParts(data):
  if len(data) == 1:
    return [[[data[0]]]]
  parts = []
  nParts = allParts(data[1:])
  for nPart in nParts:
    parts.append([[data[0]] + nPart[0]] + nPart[1:])
    parts.append([[data[0]]] + nPart)
  return parts

class CRP: #anglican crp formulation: http://www.robots.ox.ac.uk/~fwood/anglican/examples/dp_mixture_model/index.html
  def __init__(self, a):
    self.a = float(a)

  def getClass(self, x):
    while self.n <= x:
      self.getNext()
    for c in range(len(self.pns)):
      if x in self.pns[c]:
        return c
    assert(False)

  def getNext(self):
    r = random.random()
    accum = 0
    for c in range(len(self.pns)):
      accum += len(self.pns[c]) / (self.n + self.a)
      if r < accum:
        return self.setClass(c)
    return self.setClass()

  def setClass(self, c=None):
    if c != None:
      self.pns[c].append(self.n)
    else:
      c = len(self.pns)
      self.pns.append([self.n])
    self.n += 1
    return c

  def reset(self):
    self.pns = []
    self.n = 0

  def rvs(self):
    self.reset()

  def getParams(self, part = None):
    if part == None:
      return self.n, self.pns
    else:
      return sum(map(len, part)), map(len, part)

  def pdf(self, part = None):
    n, pns = self.getParams(part)
    prob = (math.gamma(self.a) * self.a**(len(pns))) / math.gamma(self.a + n)
    for pn in pns:
      prob *= math.gamma(pn)
    return prob

  def logpdf(self, part = None):
    n, pns = self.getParams(part)
    lprob = math.lgamma(self.a) + len(pns)*math.log(self.a) - math.lgamma(self.a + n)
    for pn in pns:
      lprob += math.lgamma(pn)
    return lprob

def testCrpProbs():
  crp = CRP(1.37)
  for i in range(1,20):
    print i, any(map(lambda x: abs(math.log(crp.pdf(x)) - crp.logpdf(x)) < 0.000001, allParts(range(i))))

def getLL(cl):
  ll = math.log(ss.invgamma.expect(lambda v: ss.norm.expect(lambda m: np.product([ss.norm.pdf(d, loc=m, scale=10*v) for d in cl]) ,loc=0, scale=10*v), 1, loc=0, scale=10))
  return ll

def getPost(ds, a):
  crp = CRP(a)
  parts = allParts(ds)
  post = {}

  for p in range(len(parts)):
    part = parts[p]
    ll = crp.logpdf(part)
    print p, part, ll, 
    sys.stdout.flush()
    for cl in part:
      ll += getLL(cl)
    try:
      post[len(part)] += ll
    except:
      post[len(part)] = ll
    print ll
    sys.stdout.flush()
  post = stocPy.norm(post)
  return post

if __name__ == "__main__":
  print getPost(obs, 1.72)

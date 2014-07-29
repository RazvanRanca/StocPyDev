from venture.shortcuts import *
import ppUtils as pu
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as ss
import math
import utils

def branching(v, sample, burn = 0, lag = 1):
  v.assume("fib", "(lambda (n) (if (= n 0) 0 (if (= n 1) 1 (+ (fib (- n 1)) (fib (- n 2))))))")
  v.assume("r", "(poisson 4)")
  v.assume("l", "(if (< 4 r) 6 (+ (fib (* 3 r)) (poisson 4)))")

  v.observe("(poisson l)", "6")
  samples = pu.posterior_samples(v, "r" ,sample, burn, lag)

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

def testPois(r, v, sample, burn = 0, lag = 1):
  v.assume("fib", "(lambda (n) (if (= n 0) 1 (if (= n 1) 1 (+ (fib (- n 1)) (fib (- n 2))))))")
  v.assume("r", str(r))
  if r < 4:
    v.assume("l", "6")
  else:
    v.assume("l", "(+ (fib (* 3 r)) (poisson 4))")

  v.assume("pl", "(poisson l)")
  v.observe("(poisson l)", "6")
  samples = pu.posterior_samples(v, "pl" ,sample, burn, lag)

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

def fib0(n):
  if n == 0:
    return 0
  if n < 3:
    return 1
  else:
    a = 1
    b = 1
    for i in range(3,n):
      c = a+b
      a = b
      b = c
    return a+b

"""
def fib1(n):
  if n < 2:
    return 1
  else:
    a = 1
    b = 1
    for i in range(2,n):
      c = a+b
      a = b
      b = c
    return a+b
"""

def anlPost(disp=True):
  size = 21
  pois = ss.poisson
  pois4 = {}
  for i in range(size):
    prob = pois.pmf(i,4)
    pois4[i] = prob

  rs = {}  
  for r in range(size):
    pr = pois4[r]
    l = {}
    if r > 4:
      l[6] = 1
    else:
      f = fib0(3*r)
      for k,v in pois4.items():
        l[k+f] = v
    assert(abs(sum(l.values()) - 1) < 0.0001)

    for val,prob in l.items():
      #print r, val, pois.pmf(6,val), prob
      pl = pois.pmf(6,val) * prob
      if math.isnan(pl):
        pl = 0
      try:
        rs[r] += pl*pr
      except:
        rs[r] = pl*pr

  rs = utils.norm(rs)
  if disp:
    print sorted(rs.items())
    print sum(rs.values())
    plotDict(rs)
  return rs

def anlPrior():
  pois = ss.poisson
  pois4 = {}
  fib3 = {}
  for i in range(101):
    pois4[i] = 0
    fib3[i] = 0
  for i in range(101):
    prob = pois.pmf(i,4)
    pois4[i] = prob
    if i < 5:
      fib3[fib0(3*i)] = prob
  
  fib3 = utils.norm(fib3)
  p = pois.cdf(3,4)
  print sum(pois4.values())
  print sorted(fib3.items())
  print sum(fib3.values())

  l2 = {}
  for v1, p1 in pois4.items():
    for v2, p2 in fib3.items():
      try:
        l2[v1+v2] += p1*p2
      except:
        l2[v1+v2] = p1*p2

  print sum(l2.values())

  l = {}
  for k,v in l2.items():
    l[k] = v * (1-p)

  try:
    l[6] += p
  except:
    l[6] = p

  print sum(l.values())
  plotDict(l, 2000)

def anlPrior1():
  pois = ss.poisson
  pois4 = {}
  fib3 = {}
  for i in range(101):
    pois4[i] = 0
    fib3[i] = 0
  for i in range(101):
    prob = pois.pmf(i,4)
    pois4[i] = prob
    if i < 5:
      fib3[fib1(3*i)] = prob
  
  fib3 = utils.norm(fib3)
  p = pois.cdf(3,4)
  print sum(pois4.values())
  print sorted(fib3.items())
  print sum(fib3.values())

  l2 = {}
  for v1, p1 in fib3.items():
    poisCur = {}
    for i in range(101):
      prob = pois.pmf(i,v1)
      if math.isnan(prob):
        prob = 0
        print v1, i
      poisCur[i] = prob
    #print v1, v2, p1, p2
    for v2, p2 in poisCur.items():
      try:
        l2[v1+v2] += p1*p2
      except:
        l2[v1+v2] = p1*p2

  l2 = utils.norm(l2)
  print sorted(l2.items())
  print sum(l2.values())

  l = {}
  for k,v in l2.items():
    l[k] = v * (1-p)

  try:
    l[6] += p
  except:
    l[6] = p

  print sum(l.values())
  plotDict(l, 2000)

def anlLik(plot=True, trans=True, twoDim = True):
  rng = 15
  pois = ss.poisson
  pois4 = {}
  for i in range(rng):
    pois4[i] = pois.pmf(i, 4)

  lik = {}
  for r, pr in pois4.items():
    for p, pp in pois4.items():
      if r > 4:
        l = 6
      else:
        l = fib0(3*r) + p
      pl = pois.pmf(6, l)
      if r > 4 and trans: #2nd poisson not evaluated
        prob = pr*pl
      else:
        prob = pr*pp*pl
      if r == 0 and p == 0:
        prob = 0
      
      if twoDim:
        lik[(r,p)] = prob
      else:
        try:
          lik[r] += prob
        except:
          lik[r] = prob

  if plot:
    if twoDim:
      Xs2, Ys2 = np.meshgrid(range(rng),range(rng))
      print len(Xs2), len(Ys2)
      Zs2 = []
      for i in range(len(Xs2)):
        Zs2.append([])
        for j in range(len(Ys2)):
          Zs2[i].append(lik[Xs2[i][j], Ys2[i][j]])

      fig = plt.figure()
      ax = fig.gca(projection='3d')
      ax.plot_surface(Xs2, Ys2, Zs2, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)
      plt.xlabel("pois1", size=20)
      plt.ylabel("pois2", size=20)
      ax.set_zlabel("Trace likelihood", size=20)
      plt.title("Trace likelihood when ignoring trans-dimensionality", size=20)
    else:
      Xs = []
      Ys = []
      lik = utils.norm(lik)
      [(Xs.append(x), Ys.append(y)) for x,y in sorted(lik.items())]
      plt.plot(Xs, Ys, 'D')
      plt.xlabel("pois1", size=20)
      plt.ylabel("Probability", size=20)
      plt.title("Posterior implied by ignoring trans-dimensionality", size=22)
    plt.show()
  return lik

def plotDict(dic, lim=float("inf")):
  xs = []
  ys = []
  [(xs.append(k), ys.append(v)) for k,v in sorted(dic.items()) if k < lim]
  plt.plot(xs,ys, 'D')
  plt.xlabel("pois1", size = 20)
  plt.ylabel("Probability", size=20)
  plt.title("True posterior for branching model", size=30)
  #plt.xscale("log")
  #plt.yscale("log")
  plt.show()

def getTransProb(liks, tr):
  probs = []
  for (r,p), oldProb in liks.items():
    newProb = liks[(tr,p)]
    if newProb <= 0 or oldProb <= 0:
      continue
    probs.append((r, p, newProb/oldProb, math.log(newProb), math.log(oldProb)))

  print '\n'.join(map(str, sorted(probs)))

if __name__ == "__main__":
  #v = make_church_prime_ripl()
  #branching(v, 10000)
  #testPois(3, v, 100)
  #print anlPost()
  #anlPrior()
  #getTransProb(anlLik(False), tr = 0)
  anlLik(trans=True, twoDim=True)

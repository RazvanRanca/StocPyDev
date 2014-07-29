import stocPy
import time
import scipy.stats as ss
import math

conds = None

def tdf(init=False):
  ys = []
  dof = stocPy.unifCont(2, 100, obs=init)
  for i in range(len(conds)):
    cond = None
    if init:
      cond = conds[i]
    ys.append(stocPy.studentT(dof, cond=cond))
  return ys

def tdfDecomp(init=False):
  ys = []
  d1 = stocPy.normal(0, 1, obs=init)
  d2 = stocPy.normal(0, 2, obs=init)
  d3 = stocPy.normal(0, 4, obs=init)
  d4 = stocPy.normal(0, 8, obs=init)
  d5 = stocPy.normal(0, 16, obs=init)
  dof = 2 + 98 * ss.norm.cdf(d1+d2+d3+d4+d5, loc = 0, scale = math.sqrt(sum([x**2 for x in [1,2,4,8,16]])))
  for i in range(len(conds)):
    cond = None
    if init:
      cond = conds[i]
    ys.append(stocPy.studentT(dof, cond=cond))
  return ys

def tdfExpName(init=False):
  ys = []
  dof = stocPy.unifCont(2, 100, obs=init, name = stocPy.getExplicitName("tdf", 8, 0))
  for i in range(len(conds)):
    cond = None
    if init:
      cond = conds[i]
    ys.append(stocPy.studentT(dof, cond, name = stocPy.getExplicitName("tdf", 13, i)))
  return ys

def getConds(fn):
  with open(fn,'r') as f:
    data = [float(val.strip()) for val in f.read().strip().split(',')]
  return data

aggFunc = lambda xs: 2 + 98 * ss.norm.cdf(sum(xs), loc = 0, scale = 1)

if __name__ == "__main__":
  conds = getConds("tdfData")
  #samps = stocPy.getTimedSamples(tdf, 600)['tdf-4-0']
  #print samps
  #stocPy.plotSamples(filter(lambda x: x<8, samps))
  samps = stocPy.aggDecomp(stocPy.getTimedSamples(tdfDecomp, 10, alg="met"), func=aggFunc)
  print samps
  stocPy.plotSamples(filter(lambda x: x<8, samps))
  #print tdf()
  """  
  times = []
  for i in range(50):
    startTime = time.time()
    samples = stocPy.getSamples(tdf, 100)
    times.append(time.time() - startTime)
  print times
  """

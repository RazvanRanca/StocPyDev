import stocPy
from matplotlib import pyplot as plt
import cPickle
import branchingVenture as bv
import numpy as np
import stocPy
import math
import scipy.stats as ss

post = {0: 0.020851615261930488, 1: 0.11980537300682528, 2: 0.067744481473721987, 3: 9.992589688842966e-10, 4: 7.8933974285075793e-54, 5: 0.33333507114328936, 6: 0.22222338076219267, 7: 0.12698478900696725, 8: 0.06349239450348361, 9: 0.028218842001548314, 10: 0.011287536800619315, 11: 0.0041045588365888419, 12: 0.0013681862788629479, 13: 0.00042098039349629197, 14: 0.00012028011242751155, 15: 3.2074696647336408e-05, 16: 8.0186741618340985e-06, 17: 1.8867468616080252e-06, 18: 4.192770803573402e-07, 19: 8.8268859022597819e-08, 20: 1.7653771804519576e-08, 21: 3.3626232008608741e-09, 22: 6.1138603652015677e-10, 23: 1.0632800635133162e-10, 24: 1.7721334391888611e-11, 25: 2.835413502702193e-12, 26: 4.3621746195418139e-13, 27: 6.4624809178397933e-14, 28: 9.2321155969138434e-15, 29: 1.2733952547467591e-15, 30: 1.6978603396623357e-16, 31: 2.1907875350481778e-17, 32: 2.7384844188102261e-18, 33: 3.3193750531032237e-19, 34: 3.9051471212980181e-20, 35: 4.4630252814834445e-21, 36: 4.9589169794260022e-22, 37: 5.36099132910932e-23, 38: 5.6431487674833744e-24, 39: 5.7878448897265764e-25, 40: 5.7878448897265503e-26, 41: 5.6466779411966489e-27, 42: 5.3777885154254306e-28, 43: 5.0025939678375241e-29, 44: 4.5478126980341361e-30, 45: 4.0425001760301941e-31, 46: 3.5152175443742627e-32, 47: 2.9916745058503504e-33, 48: 2.4930620882086573e-34, 49: 2.0351527250683204e-35, 50: 1.6281221800546502e-36, 51: 1.2769585725918323e-37, 52: 9.8227582507064752e-39, 53: 7.4134024533633357e-40, 54: 5.4914092247136307e-41, 55: 3.9937521634280061e-42, 56: 2.8526801167343401e-43, 57: 2.0018807836732729e-44, 58: 1.3806074370160323e-45, 59: 9.3600504204477849e-47, 60: 6.2400336136315869e-48, 61: 4.0918253204142188e-49, 62: 2.6398873034931542e-50, 63: 1.6761189228527715e-51, 64: 1.0475743267829744e-52, 65: 6.4466112417415276e-54, 66: 3.9070371162068563e-55, 67: 2.3325594723622722e-56, 68: 1.3720938072719211e-57, 69: 7.9541669986777969e-59, 70: 4.5452382849590175e-60, 71: 2.5606976253289085e-61, 72: 1.4226097918493977e-62, 73: 7.7951221471198371e-64, 74: 4.2135795389839391e-65, 75: 2.2472424207913712e-66, 76: 1.1827591688375536e-67, 77: 6.1442034744807972e-69, 78: 3.1508735766569453e-70, 79: 1.5953790261554935e-71, 80: 7.9768951307766224e-73, 81: 3.9392074719883955e-74, 82: 1.9215646204821048e-75, 83: 9.2605523878661387e-77, 84: 4.4097868513649239e-78, 85: 2.0751938124069896e-79, 86: 9.6520642437535323e-81, 87: 4.4377306867830686e-82, 88: 2.0171503121741864e-83, 89: 9.0658440996597338e-85, 90: 4.0292640442930584e-86, 91: 1.7711050744144346e-87, 92: 7.7004568452800955e-89, 93: 3.3120244495829887e-90, 94: 1.409372106205564e-91, 95: 5.9341983419182025e-93, 96: 2.4725826424657694e-94, 97: 1.0196217082333034e-95, 98: 4.1617212580952852e-97, 99: 1.6815035386243325e-98, 100: 6.726014154497484e-100}

def fib(n):
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

ls = []
def branchingDec2(init=False):
  global ls
  r1 = stocPy.normal(0, math.sqrt(0.5), obs=init)
  r2 = stocPy.normal(0, math.sqrt(0.25), obs=init)
  r3 = stocPy.normal(0, math.sqrt(0.125), obs=init)
  r4 = stocPy.normal(0, math.sqrt(0.0625), obs=init)
  r5 = stocPy.normal(0, math.sqrt(0.0625), obs=init)
  rn = ss.norm.cdf(r1+r2+r3+r4+r5, loc=0, scale=1)
  r = int(ss.poisson.ppf(rn, 4)) 
  #print r
  if r > 4:
    l = 6
  else:
    p1 = stocPy.normal(0, math.sqrt(0.5))
    p2 = stocPy.normal(0, math.sqrt(0.25))
    p3 = stocPy.normal(0, math.sqrt(0.125))
    p4 = stocPy.normal(0, math.sqrt(0.0625))
    p5 = stocPy.normal(0, math.sqrt(0.0625))
    pn = ss.norm.cdf(p1+p2+p3+p4+p5, loc=0, scale=1)
    p = int(ss.poisson.ppf(pn, 4)) 
    l = fib(3*r) + p 

  #print r, fib(3*r), l
  cond = None
  if init:
    cond = 6
  stocPy.poisson(l, cond=cond)
  ls.append(l)

def branchingDec1(init=False):
  global ls
  r1 = stocPy.normal(0, math.sqrt(0.5), obs=init)
  r2 = stocPy.normal(0, math.sqrt(0.25), obs=init)
  r3 = stocPy.normal(0, math.sqrt(0.125), obs=init)
  r4 = stocPy.normal(0, math.sqrt(0.0625), obs=init)
  r5 = stocPy.normal(0, math.sqrt(0.0625), obs=init)
  rn = ss.norm.cdf(r1+r2+r3+r4+r5, loc=0, scale=1)
  r = int(ss.poisson.ppf(rn, 4)) 
  #print r
  if r > 4:
    l = 6
  else:
    p2 = stocPy.poisson(4)
    l = fib(3*r) + p2 

  #print r, fib(3*r), l
  cond = None
  if init:
    cond = 6
  stocPy.poisson(l, cond=cond)
  ls.append(l)

def branching(init=False):
  global ls
  r = stocPy.poisson(4, obs=init)
  #print r
  if r > 4:
    l = 6
  else:
    p2 = stocPy.poisson(4)
    l = fib(3*r) + p2 

  #print r, fib(3*r), l
  cond = None
  if init:
    cond = 6
  stocPy.poisson(l, cond=cond)
  ls.append(l)

def branching1(init=False):
  global ls
  r = stocPy.poisson(4, name=stocPy.getExplicitName("branching", 19, 0), obs=init)
  #print r
  p2 = stocPy.poisson(4, name=stocPy.getExplicitName("branching", 23, 0))
  if r > 4:
    l = 6
  else:
    l = fib(3*r) + p2 

  #print r, fib(3*r), l
  cond = None
  if init:
    cond = 6
  stocPy.poisson(l, name=stocPy.getExplicitName("branching", 26, 0), cond=cond)
  ls.append(l)

def testPois(init=False):
  global ls
  r = stocPy.poisson(4, name=stocPy.getExplicitName("branching", 19, 0), obs=init)
  ls.append(r)

def testUnif(init=False):
  global ls
  r = stocPy.unifCont(0,5, name=stocPy.getExplicitName("branching", 19, 0), obs=init)
  ls.append(r)

def testStudentT(init=False):
  global ls
  r = stocPy.studentT(4, name=stocPy.getExplicitName("branching", 19, 0), obs=init)
  ls.append(r)

aggFunc = lambda xs: int(ss.poisson.ppf(ss.norm.cdf(sum(xs), loc=0, scale=1), 4))

def genRuns(model, fn, noRuns = 100, length = 20000):
  """
  runs = []
  for i in range(100):
    print "Run", i
    runs.append(stocPy.getSamples(branching, length, discAll=True, alg="met")['branching-19-0'])

  with open("metRunsNoTrans",'w') as f:
    cPickle.dump(runs, f)
  """
  runs = []
  for i in range(noRuns):
    print "Run", i
    if model == branching:
      runs.append(stocPy.getSamplesByLL(model, length, discAll=True, alg="slice", thresh=0.5)['branching-19-0'])
    else:
      runs.append(stocPy.aggDecomp(stocPy.getSamplesByLL(model, length, discAll=True, alg="met")), func=aggFunc)

  with open("branching/" + fn,'w') as f:
    cPickle.dump(runs, f)

def procRuns():
  with open("metRuns",'r') as f:
    runs1 = cPickle.load(f)

  with open("sliceMet01Runs",'r') as f:
    runs2 = cPickle.load(f)

  runs1Proc = []
  for run in runs1:
    runs1Proc.append(dict([(k,v) for k,v in run.items() if k <= 20000]))

  runs2Proc = []
  for run in runs2:
    runs2Proc.append({})      
    for k,v in sorted(run.items()):
      runs2Proc[-1][len(runs2Proc[-1])+1] = v
      if len(runs2Proc[-1]) >= 20000:
        break
  
  #print map(len,runs1Proc)
  #print map(len,runs2Proc)
  #stocPy.calcKLTests(post, runs1Proc, runs2Proc, burnIn = 1000, xlim=20000)
  stocPy.calcKLSumms(post, [runs1Proc, runs2Proc], burnIn = 1000, xlim=20000)
  #print runs2[0]

def testBranchingSliceNoTrans(samps):
  post1 = bv.norm(dict(filter(lambda (k,v): k<5, post.items())))
  post2 = bv.norm(dict(filter(lambda (k,v): k>=5, post.items())))
  test = lambda dic: 1 if dic.values()[0] > 4 else 0
  stocPy.calcKLCondTests([post1, post2], samps, test)

def transLLtoSamp(run, lim=float("inf")):
  nrun = {}
  for k,v in sorted(run.items()):
    nrun[len(nrun)] = v
    if len(nrun) >= lim:
      break
  return nrun

if __name__ == "__main__":
  #genRuns(branchingDec2, "metRunsByLLDec44", noRuns=100, length=20000)
  #with open("branching/sliceRunsV1ByLL",'r') as f:
  #  print map(len, cPickle.load(f))
  #procRuns()
  #stocPy.plotSamples(stocPy.getSamples(branching, 1000, discAll=True))#, 'branching-23-0')
  #print stocPy.calcKLTest(post, stocPy.getSamplesByLL(branching, 20000, discAll=True, alg="sliceNoTrans")['branching-19-0'])

  #stocPy.plotSamples(stocPy.getSamplesByLL(branching, 10000, discAll=True, alg="slice"))

  #print sorted(stocPy.getSamplesByLL(branching, 30, discAll=True, alg="sliceNoTrans")['branching-19-0'].items())
  #with open("branching/sliceMet01Runs",'r') as f:
  #  runs2 = cPickle.load(f)

  #print np.mean(map(len, runs2))
  #genRuns()

  #print sorted(runs2[5].items())
  #stocPy.calcKLTest(post, runs2[5])

  #samps = [stocPy.getSamplesByLL(branching, 10000, discAll=True, alg="sliceStat")['branching-19-0'] for i in range(100)]
  #print samps.count(0), samps.count(1)
  #testBranchingSliceNoTrans(samps)
  #procRuns()
  #print ls
  #plt.hist(ls,100)
  #plt.show()
  
  with open("branching/metRuns",'r') as f:
    runs1 = cPickle.load(f)

  #runs1 = map(lambda run: transLLtoSamp(run, lim=20000), runs1)
  #print map(len, runs1)
  #with open("branching/sliceMet01RunsByLL",'r') as f:
  #  runs2 = cPickle.load(f)

  #runs2 = map(lambda run: transLLtoSamp(run, lim=20000), runs2)
  #print map(len, runs2)
  with open("branching/sliceMet05RunsByLL",'r') as f:
    runs3 = cPickle.load(f)

  #runs3 = map(lambda run: transLLtoSamp(run, lim=20000), runs3)
  #print map(len, runs3)
  with open("branching/sliceRunsByLL",'r') as f:
    runs4 = cPickle.load(f)

  #with open("branching/sliceRunsV11ByLL",'r') as f:
  #  runs5 = cPickle.load(f)

  with open("branching/metRunsByLLDec44",'r') as f:
    runs6 = cPickle.load(f)
  
  #print map(len, runs3)
  stocPy.calcKLSumms(post, [runs6, runs1,runs3,runs4], ["Metropolis Decomp44", "Metropolis","1:1 Metropolis:Slice","Slice"], xlim=20000, burnIn = 1000)
  #stocPy.calcKLTests(post, [runs5, runs1,runs3,runs4], ["Slice TD1", "Metropolis","1:1 Metropolis:Slice","Slice"], xlim=100000, burnIn = 1000)

  #print sorted(runs2[17].items())
  #print sorted([(i, zip(*stocPy.calcKLTest(post, runs2[i], plot=False))[-1]) for i in range(len(runs2))], key=lambda (i,(k,v)): v)
  #plt.hist(runs2[17].values(),100)
  #plt.show()
  
  #print runs1[0]
  #stocPy.calcKLTest(post, runs1[0], cutOff = 20)

  #print np.mean(map(len,runs))
  #print sorted(runs2[68].items())
  #print sorted(runs2[68].items(), key = lambda (k,v): v)
  #print sorted([(sorted(runs1[i].items(), key = lambda (k,v): v)[-1][1], i) for i in range(len(runs1))])
  #print sorted([(sorted(runs2[i].items(), key = lambda (k,v): v)[-1], i) for i in range(len(runs2))], key = lambda ((k,v),i): v)
  #stocPy.calcKLSumms(post, [runs1, runs2], burnIn = 10000)

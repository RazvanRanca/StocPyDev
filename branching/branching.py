import sys
sys.path.append("/home/haggis/Desktop/StocPyDev")
import stocPyDev as stocPy
from matplotlib import pyplot as plt
import cPickle
import branchingVenture as bv
import numpy as np
import math
import scipy.stats as ss

post = {0: 0.020851615261930488, 1: 0.11980537300682528, 2: 0.067744481473721987, 3: 9.992589688842966e-10, 4: 7.8933974285075793e-54, 5: 0.33333507114328936, 6: 0.22222338076219267, 7: 0.12698478900696725, 8: 0.06349239450348361, 9: 0.028218842001548314, 10: 0.011287536800619315, 11: 0.0041045588365888419, 12: 0.0013681862788629479, 13: 0.00042098039349629197, 14: 0.00012028011242751155, 15: 3.2074696647336408e-05, 16: 8.0186741618340985e-06, 17: 1.8867468616080252e-06, 18: 4.192770803573402e-07, 19: 8.8268859022597819e-08, 20: 1.7653771804519576e-08, 21: 3.3626232008608741e-09, 22: 6.1138603652015677e-10, 23: 1.0632800635133162e-10, 24: 1.7721334391888611e-11, 25: 2.835413502702193e-12, 26: 4.3621746195418139e-13, 27: 6.4624809178397933e-14, 28: 9.2321155969138434e-15, 29: 1.2733952547467591e-15, 30: 1.6978603396623357e-16, 31: 2.1907875350481778e-17, 32: 2.7384844188102261e-18, 33: 3.3193750531032237e-19, 34: 3.9051471212980181e-20, 35: 4.4630252814834445e-21, 36: 4.9589169794260022e-22, 37: 5.36099132910932e-23, 38: 5.6431487674833744e-24, 39: 5.7878448897265764e-25, 40: 5.7878448897265503e-26, 41: 5.6466779411966489e-27, 42: 5.3777885154254306e-28, 43: 5.0025939678375241e-29, 44: 4.5478126980341361e-30, 45: 4.0425001760301941e-31, 46: 3.5152175443742627e-32, 47: 2.9916745058503504e-33, 48: 2.4930620882086573e-34, 49: 2.0351527250683204e-35, 50: 1.6281221800546502e-36, 51: 1.2769585725918323e-37, 52: 9.8227582507064752e-39, 53: 7.4134024533633357e-40, 54: 5.4914092247136307e-41, 55: 3.9937521634280061e-42, 56: 2.8526801167343401e-43, 57: 2.0018807836732729e-44, 58: 1.3806074370160323e-45, 59: 9.3600504204477849e-47, 60: 6.2400336136315869e-48, 61: 4.0918253204142188e-49, 62: 2.6398873034931542e-50, 63: 1.6761189228527715e-51, 64: 1.0475743267829744e-52, 65: 6.4466112417415276e-54, 66: 3.9070371162068563e-55, 67: 2.3325594723622722e-56, 68: 1.3720938072719211e-57, 69: 7.9541669986777969e-59, 70: 4.5452382849590175e-60, 71: 2.5606976253289085e-61, 72: 1.4226097918493977e-62, 73: 7.7951221471198371e-64, 74: 4.2135795389839391e-65, 75: 2.2472424207913712e-66, 76: 1.1827591688375536e-67, 77: 6.1442034744807972e-69, 78: 3.1508735766569453e-70, 79: 1.5953790261554935e-71, 80: 7.9768951307766224e-73, 81: 3.9392074719883955e-74, 82: 1.9215646204821048e-75, 83: 9.2605523878661387e-77, 84: 4.4097868513649239e-78, 85: 2.0751938124069896e-79, 86: 9.6520642437535323e-81, 87: 4.4377306867830686e-82, 88: 2.0171503121741864e-83, 89: 9.0658440996597338e-85, 90: 4.0292640442930584e-86, 91: 1.7711050744144346e-87, 92: 7.7004568452800955e-89, 93: 3.3120244495829887e-90, 94: 1.409372106205564e-91, 95: 5.9341983419182025e-93, 96: 2.4725826424657694e-94, 97: 1.0196217082333034e-95, 98: 4.1617212580952852e-97, 99: 1.6815035386243325e-98, 100: 6.726014154497484e-100}

post_1_100 = {0: 0.99999999999995282, 1: 4.7157565438967786e-14, 2: 4.0169031669314827e-214, 3: 0.0, 4: 0.0, 5: 5.4215158728254636e-138, 6: 3.6143439152169724e-138, 7: 2.0653393801239843e-138, 8: 1.032669690061992e-138, 9: 4.5896430669421932e-139, 10: 1.8358572267768754e-139, 11: 6.6758444610068197e-140, 12: 2.2252814870022745e-140, 13: 6.8470199600070031e-141, 14: 1.9562914171448508e-141, 15: 5.2167771123862678e-142, 16: 1.3041942780965666e-142, 17: 3.068692419050748e-143, 18: 6.8193164867794599e-144, 19: 1.435645576164095e-144, 20: 2.8712911523281919e-145, 21: 5.4691260044346559e-146, 22: 9.943865462608427e-147, 23: 1.7293679065405965e-147, 24: 2.8822798442343289e-148, 25: 4.61164775077495e-149, 26: 7.0948426934998904e-150, 27: 1.0510878064444394e-150, 28: 1.501554009206318e-151, 29: 2.0711089782156468e-152, 30: 2.7614786376208464e-153, 31: 3.563198242091419e-154, 32: 4.4539978026142803e-155, 33: 5.3987852152899057e-156, 34: 6.3515120179882992e-157, 35: 7.2588708777009061e-158, 36: 8.0654120863342644e-159, 37: 8.7193644176588712e-160, 38: 9.1782783343775652e-161, 39: 9.4136188044898728e-162, 40: 9.4136188044898294e-163, 41: 9.1840183458437588e-164, 42: 8.7466841388989012e-165, 43: 8.1364503617662809e-166, 44: 7.3967730561512021e-167, 45: 6.5749093832452704e-168, 46: 5.7173125071700948e-169, 47: 4.8657978784424935e-170, 48: 4.0548315653687963e-171, 49: 3.3100665839745747e-172, 50: 2.6480532671796495e-173, 51: 2.0769045232780748e-174, 52: 1.5976188640600699e-175, 53: 1.2057500860830645e-176, 54: 8.9314821191338915e-178, 55: 6.495623359369957e-179, 56: 4.6397309709786228e-180, 57: 3.2559515585815736e-181, 58: 2.2454838335045045e-182, 59: 1.5223619210200205e-183, 60: 1.0149079473466367e-184, 61: 6.6551340809616537e-186, 62: 4.2936348909431814e-187, 63: 2.7261173910749952e-188, 64: 1.7038233694218595e-189, 65: 1.0485066888750143e-190, 66: 6.3545859931816937e-192, 67: 3.7937826824964847e-193, 68: 2.2316368720567495e-194, 69: 1.2937025345256506e-195, 70: 7.3925859115755979e-197, 71: 4.164837133281839e-198, 72: 2.3137984073788059e-199, 73: 1.2678347437691844e-200, 74: 6.8531607771311207e-202, 75: 3.6550190811364953e-203, 76: 1.9236942532297179e-204, 77: 9.9932168998946384e-206, 78: 5.124726615330789e-207, 79: 2.5947982862435648e-208, 80: 1.2973991431216448e-209, 81: 6.4069093487487654e-211, 82: 3.1253216335359192e-212, 83: 1.5061791004993314e-213, 84: 7.1722814309493538e-215, 85: 3.3751912616231707e-216, 86: 1.569856400754978e-217, 87: 7.217730578183541e-219, 88: 3.280786626447168e-220, 89: 1.4745108433471181e-221, 90: 6.5533815259869375e-223, 91: 2.8806072641699248e-224, 92: 1.2524379409434368e-225, 93: 5.3868298535204059e-227, 94: 2.2922680227747006e-228, 95: 9.6516548327356424e-230, 96: 4.0215228469729433e-231, 97: 1.6583599368960532e-232, 98: 6.7688160689637584e-234, 99: 2.7348751793792561e-235, 100: 1.0939500717517272e-236}

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

def branchingRep():
  r = stocPy.poisson(4, obs=True)
  if r > 4:
    l = 6
  else:
    p2 = stocPy.poisson(4)
    l = fib(3*r) + p2
  for i in range(100):
    stocPy.poisson(l, cond=1)

def branchingRepPart1U15():
  r = stocPy.stocPrim("poisson", (4,), obs=True, part=stocPy.stocPrim("randint", (0,16)))
  if r > 4:
    l = 6
  else:
    p2 = stocPy.poisson(4)
    l = fib(3*r) + p2
  for i in range(100):
    stocPy.poisson(l, cond=1)

def branchingRepPart12U15():
  r = stocPy.stocPrim("poisson", (4,), obs=True, part=stocPy.stocPrim("randint", (0,16)))
  if r > 4:
    l = 6
  else:
    p2 = stocPy.stocPrim("poisson", (4,), part=stocPy.stocPrim("randint", (0,16)))
    l = fib(3*r) + p2
  for i in range(100):
    stocPy.poisson(l, cond=1)

def branching():
  r = stocPy.poisson(4, obs=True)
  if r > 4:
    l = 6
  else:
    p2 = stocPy.poisson(4)
    l = fib(3*r) + p2 
  stocPy.poisson(l, cond=6)

def branchingPart1U15():
  r = stocPy.stocPrim("poisson", (4,), obs=True, part=stocPy.stocPrim("randint", (0,16)))
  if r > 4:
    l = 6
  else:
    p2 = stocPy.poisson(4)
    l = fib(3*r) + p2 
  stocPy.poisson(l, cond=6)

def branchingPart2U15():
  r = stocPy.poisson(4, obs=True)
  if r > 4:
    l = 6
  else:
    p2 = stocPy.stocPrim("poisson", (4,), part=stocPy.stocPrim("randint", (0,16)))
    l = fib(3*r) + p2 
  stocPy.poisson(l, cond=6)

def branchingPart12U15():
  r = stocPy.stocPrim("poisson", (4,), obs=True, part=stocPy.stocPrim("randint", (0,16)))
  if r > 4:
    l = 6
  else:
    p2 = stocPy.stocPrim("poisson", (4,), part=stocPy.stocPrim("randint", (0,16)))
    l = fib(3*r) + p2 
  stocPy.poisson(l, cond=6)

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

def genRuns(model, fn, noRuns = 100, length = 20000, agg=False, time=None, alg="met", thresh=0.1, aggFunc=lambda x:sum(x)):
  if time:
    length = time
    stocPyFunc = stocPy.getTimedSamples
  else:
    stocPyFunc = stocPy.getSamplesByLL
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
    if not agg:
      samples = stocPyFunc(model, length, alg=alg, thresh=thresh)
      assert(len(samples.keys()) == 1)
      runs.append(samples[samples.keys()[0]])
    else:
      runs.append(stocPy.aggDecomp(stocPyFunc(model, length, discAll=True, alg="met")), func=aggFunc)

  with open(stocPy.getCurDir(__file__) + "/"  + fn,'w') as f:
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

def displayExperiments(xlim=300000):
  paths = ["branchingRepTimed05", "branchingRepP1U15Timed05", "branchingRepP12U15Timed05"]#, "branchingP12U15Timed1"]
  titles = ["Depth_0_0", "Depth_U20_0", "Depth_U20_U20"]#, "Depth_U20_U20"]

  runs = []
  for path in paths:
    with open(stocPy.getCurDir(__file__) + "/" + path,'r') as f:
      runs.append(cPickle.load(f))
  #stocPy.calcKLTests(post_1_100, runs, titles, xlim=xlim, burnIn = 0) # show all runs
  stocPy.calcKLSumms(post_1_100, runs, titles, xlim=xlim, burnIn = 0) # show run quartiles
  

if __name__ == "__main__":
  with open(stocPy.getCurDir(__file__) + "/branchingRepP12U15Timed05", 'r') as f:
	  print cPickle.load(f)
  #genRuns(branchingRep, "branchingRepTimed05", time=30, noRuns=20)
  #genRuns(branchingRepPart1U15, "branchingRepP1U15Timed05", time=30, noRuns=20)
  #genRuns(branchingPart2U15, "branchingP2U15Timed1", time=60, noRuns=20)
  #genRuns(branchingRepPart12U15, "branchingRepP12U15Timed05", time=30, noRuns=20)
  cd = stocPy.getCurDir(__file__)
  #with open(cd + "/posterior", 'r') as fr:
  #  with open(cd + "/listPosterior", 'w') as fw:
  #    cPickle.dump(tuple(map(list, zip(*sorted(cPickle.load(fr).items())))), fw)
  #stocPy.calcKLSumms(post , [cd + "/branchingTimed1", cd + "/branchingP1U15Timed1", cd + "/branchingP12U15Timed1"], aggFreq=np.logspace(1,math.log(1000000,10),10), burnIn=1000, names=["Depth_0_0", "Depth_U20_0", "Depth_U20_U20"], modelName = "Branching")
  displayExperiments()
  assert(False)
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

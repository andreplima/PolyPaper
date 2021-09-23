import pickle
import codecs
import os
import os.path
import re
import itertools

import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot    as plt
import matplotlib.animation as manimation
import bootstrapped.bootstrap       as bs
import bootstrapped.stats_functions as bs_stats

from datetime      import datetime
from collections   import OrderedDict, defaultdict
from configparser  import RawConfigParser

from math          import modf
from scipy.stats   import pearsonr, spearmanr

from copy          import copy
from itertools     import permutations, chain, combinations
from random        import seed, sample, shuffle, randint, random

from shapely.geometry        import Polygon
from matplotlib.collections  import LineCollection, PolyCollection
from matplotlib.ticker       import AutoMinorLocator
from scipy.stats.mstats      import gmean
from scipy.cluster.hierarchy import dendrogram
from scipy.optimize          import differential_evolution
from sklearn.cluster         import AgglomerativeClustering
from sklearn.neural_network  import MLPRegressor

ECO_SEED = 23
ECO_PRECISION = 1E-9
ECO_DATETIME_FMT = '%Y%m%d%H%M%S'
ECO_FIELDSEP = '|'
ECO_LISTSEP  = ','

# constants identifying polygon drawing patterns
ECO_PTRN_DEMAND  = 0
ECO_PTRN_OFFER   = 1
ECO_PTRN_BENEFIT = 2

#--------------------------------------------------------------------------------------------------
# General purpose definitions - I/O interfaces used in logging and serialisation
#--------------------------------------------------------------------------------------------------

# buffer where all tsprint messages are stored
LogBuffer = []

def stimestamp():
  return(datetime.now().strftime(ECO_DATETIME_FMT))

def stimediff(finishTs, startTs):
  return str(datetime.strptime(finishTs, ECO_DATETIME_FMT) - datetime.strptime(startTs, ECO_DATETIME_FMT))

def tsprint(msg, verbose=True):
  buffer = '[{0}] {1}'.format(stimestamp(), msg)
  if(verbose):
    print(buffer)
  LogBuffer.append(buffer)

def resetLog():
  LogBuffer = []

def saveLog(filename):
  saveAsText('\n'.join(LogBuffer), filename)

def serialise(obj, name):
  f = open(name + '.pkl', 'wb')
  p = pickle.Pickler(f)
  p.fast = True
  p.dump(obj)
  f.close()
  p.clear_memo()

def deserialise(name):
  f = open(name + '.pkl', 'rb')
  p = pickle.Unpickler(f)
  obj = p.load()
  f.close()
  return obj

def file2List(filename, separator = ',', erase = '"', _encoding = 'iso-8859-1'):

  contents = []
  f = codecs.open(filename, 'r', encoding=_encoding)
  if(len(erase) > 0):
    for buffer in f:
      contents.append(buffer.replace(erase, '').rstrip().split(separator))
  else:
    for buffer in f:
      contents.append(buffer.rstrip().split(separator))
  f.close()

  return(contents)

def saveAsText(content, filename, _encoding='utf-8'):
  f = codecs.open(filename, 'w', encoding=_encoding)
  f.write(content)
  f.close()

#-------------------------------------------------------------------------------------------------------------------------------------------
# General purpose definitions - interface to handle parameter files
#-------------------------------------------------------------------------------------------------------------------------------------------

# Essay Parameters hashtable
EssayParameters = {}

def setupEssayConfig(configFile = ''):

  # initialises the random number generator
  seed(ECO_SEED)

  # defines default values for some configuration parameters
  setEssayParameter('ESSAY_ESSAYID',  'None')
  setEssayParameter('ESSAY_CONFIGID', 'None')
  setEssayParameter('ESSAY_SCENARIO', 'None')
  setEssayParameter('ESSAY_RUNS',     '1')

  # overrides default values with user-defined configuration
  loadEssayConfig(configFile)

  return listEssayConfig()

def setEssayParameter(param, value):
  """
  Purpose: sets the value of a specific parameter
  Arguments:
  - param: string that identifies the parameter
  - value: its new value
    Premises:
    1) When using inside python code, declare value as string, independently of its true type.
       Example: 'True', '0.32', 'Rastrigin, normalised'
    2) When using parameters in Config files, declare value as if it was a string, but without the enclosing ''.
       Example: True, 0.32, Rastrigin, only Reproduction
  Returns: None
  """

  so_param = param.upper()

  # boolean-valued parameters
  if(so_param in ['PARAM_SAVEIT', 'PARAM_PLOTGRIDS', 'PARAM_RETRACEOPT']):

    so_value = eval(value[0]) if isinstance(value, list) else bool(value)

  # integer-valued parameters
  elif(so_param in ['ESSAY_RUNS', 'PARAM_MAXCORES', 'PARAM_SAMPLESIZE', 'PARAM_RETRACEDPI',
                    'PARAM_RETRACEFPS', 'PARAM_NUMREPLICAS', 'PARAM_NUMEXPERTS',
                    'PARAM_NUMOFCASES', 'PARAM_COMMONCASES', 'PARAM_QUANTINPUT',
                    'PARAM_QUANTOUTPUT', 'PARAM_ROUNDSCORES']):

    so_value = eval(value[0])

  # floating-point-valued parameters
  elif(so_param in ['PARAM_FLOAT', 'PARAM_CUTOFF', 'PARAM_SPLIT']):

    so_value = float(eval(value[0]))

  # parameters that requires eval expansion
  elif(so_param in ['PARAM_SOURCEPATH',  'PARAM_TARGETPATH',  'PARAM_CASEIDS',     'PARAM_EXCLUDE',
                    'PARAM_FONTGON',     'PARAM_UNITSIZES',   'PARAM_XOFFSET',     'PARAM_YOFFSET',
                    'PARAM_INNERSEPS',   'PARAM_INNERSEPSR',  'PARAM_TITLEOFFSET',
                    'PARAM_TAGOFFSET',   'PARAM_RETRACEGRID', 'PARAM_ROW_SCRIPT',
                    'PARAM_CUTOFFS',     'PARAM_DEOPTS',      'PARAM_KNN',         'PARAM_METRIC_KNN',
                    'PARAM_MLP',         'PARAM_TIMEPERTASK', 'PARAM_TASKORDER',
                    'PARAM_DOMAINS',     'PARAM_ULIMITS',     'PARAM_SCORES']):

    so_value = value

  # parameters that represent text
  else:

    so_value = value[0]

  EssayParameters[so_param] = so_value

def getEssayParameter(param):
  return EssayParameters[param.upper()]

def overrideEssayParameter(param):

  if(param in os.environ):
    param_value = os.environ[param]
    tsprint('-- option {0} replaced from {1} to {2} (environment variable setting)'.format(param,
                                                                                           getEssayParameter(param),
                                                                                           param_value))
    setEssayParameter(param, [str(param_value)])

  return getEssayParameter(param)

class OrderedMultisetDict(OrderedDict):

  def __setitem__(self, key, value):

    try:
      item = self.__getitem__(key)
    except KeyError:
      super(OrderedMultisetDict, self).__setitem__(key, value)
      return

    if isinstance(value, list):
      item.extend(value)
    else:
      item.append(value)

    super(OrderedMultisetDict, self).__setitem__(key, item)

def loadEssayConfig(configFile):

  """
  Purpose: loads essay configuration coded in a essay parameters file
  Arguments:
  - configFile: name and path of the configuration file
  Returns: None, but EssayParameters dictionary is updated
  """

  if(len(configFile) > 0):

    if(os.path.exists(configFile)):

      # initialises the config parser and set a custom dictionary in order to allow multiple entries
      # of a same key (example: several instances of GA_ESSAY_ALLELE
      config = RawConfigParser(dict_type = OrderedMultisetDict)
      config.read(configFile)

      # loads parameters codified in the ESSAY section
      for param in config.options('ESSAY'):
        setEssayParameter(param, config.get('ESSAY', param))

      # loads parameters codified in the PROBLEM section
      for param in config.options('PROBLEM'):
        setEssayParameter(param, config.get('PROBLEM', param))

      # expands parameter values that requires evaluation
      # parameters that may occur once, and hold lists or tuples
      if('PARAM_SOURCEPATH' in EssayParameters):
        EssayParameters['PARAM_SOURCEPATH']  = eval(EssayParameters['PARAM_SOURCEPATH'][0])

      if('PARAM_TARGETPATH' in EssayParameters):
        EssayParameters['PARAM_TARGETPATH']  = eval(EssayParameters['PARAM_TARGETPATH'][0])

      if('PARAM_CASEIDS' in EssayParameters):
        EssayParameters['PARAM_CASEIDS']  = eval(EssayParameters['PARAM_CASEIDS'][0])

      if('PARAM_FONTGON' in EssayParameters):
        EssayParameters['PARAM_FONTGON']  = eval(EssayParameters['PARAM_FONTGON'][0])

      if('PARAM_UNITSIZES' in EssayParameters):
        EssayParameters['PARAM_UNITSIZES']  = eval(EssayParameters['PARAM_UNITSIZES'][0])

      if('PARAM_XOFFSET' in EssayParameters):
        EssayParameters['PARAM_XOFFSET']  = eval(EssayParameters['PARAM_XOFFSET'][0])

      if('PARAM_YOFFSET' in EssayParameters):
        EssayParameters['PARAM_YOFFSET']  = eval(EssayParameters['PARAM_YOFFSET'][0])

      if('PARAM_INNERSEPS' in EssayParameters):
        EssayParameters['PARAM_INNERSEPS']  = eval(EssayParameters['PARAM_INNERSEPS'][0])

      if('PARAM_INNERSEPSR' in EssayParameters):
        EssayParameters['PARAM_INNERSEPSR']  = eval(EssayParameters['PARAM_INNERSEPSR'][0])

      if('PARAM_RETRACEGRID' in EssayParameters):
        EssayParameters['PARAM_RETRACEGRID']  = eval(EssayParameters['PARAM_RETRACEGRID'][0])

      if('PARAM_ROW_SCRIPT' in EssayParameters):
        EssayParameters['PARAM_ROW_SCRIPT']  = eval(EssayParameters['PARAM_ROW_SCRIPT'][0])

      if('PARAM_TITLEOFFSET' in EssayParameters):
        EssayParameters['PARAM_TITLEOFFSET']  = eval(EssayParameters['PARAM_TITLEOFFSET'][0])

      if('PARAM_TAGOFFSET' in EssayParameters):
        EssayParameters['PARAM_TAGOFFSET']  = eval(EssayParameters['PARAM_TAGOFFSET'][0])

      if('PARAM_CUTOFFS' in EssayParameters):
        EssayParameters['PARAM_CUTOFFS']  = eval(EssayParameters['PARAM_CUTOFFS'][0])

      if('PARAM_DEOPTS' in EssayParameters):
        EssayParameters['PARAM_DEOPTS']  = eval(EssayParameters['PARAM_DEOPTS'][0])

      if('PARAM_KNN' in EssayParameters):
        EssayParameters['PARAM_KNN']  = eval(EssayParameters['PARAM_KNN'][0])

      if('PARAM_METRIC_KNN' in EssayParameters):
        EssayParameters['PARAM_METRIC_KNN']  = eval(EssayParameters['PARAM_METRIC_KNN'][0])

      if('PARAM_MLP' in EssayParameters):
        EssayParameters['PARAM_MLP']  = eval(EssayParameters['PARAM_MLP'][0])

      if('PARAM_TIMEPERTASK' in EssayParameters):
        EssayParameters['PARAM_TIMEPERTASK']  = eval(EssayParameters['PARAM_TIMEPERTASK'][0])

      if('PARAM_TASKORDER' in EssayParameters):
        EssayParameters['PARAM_TASKORDER']  = eval(EssayParameters['PARAM_TASKORDER'][0])

      if('PARAM_EXCLUDE' in EssayParameters):
        EssayParameters['PARAM_EXCLUDE']  = eval(EssayParameters['PARAM_EXCLUDE'][0])

      if('PARAM_DOMAINS' in EssayParameters):
        EssayParameters['PARAM_DOMAINS']  = eval(EssayParameters['PARAM_DOMAINS'][0])

      if('PARAM_ULIMITS' in EssayParameters):
        EssayParameters['PARAM_ULIMITS']  = eval(EssayParameters['PARAM_ULIMITS'][0])

      if('PARAM_SCORES' in EssayParameters):
        EssayParameters['PARAM_SCORES']  = eval(EssayParameters['PARAM_SCORES'][0])

      # checks if configuration is ok
      (check, errors) = checkEssayConfig(configFile)
      if(not check):
        print(errors)
        exit(1)

    else:

      print('*** Warning: Configuration file [{1}] was not found'.format(configFile))

def checkEssayConfig(configFile):

  check = True
  errors = []
  errorMsg = ""

  # insert criteria below
  if(EssayParameters['ESSAY_ESSAYID'] not in EssayParameters['ESSAY_SCENARIO']):
    check = False
    #errors.append("Parameter {0} must respect restriction: {1}\n".format('ESSAY_ESSAYID', 'be part of the ESSAY_SCENARIO identification'))
    param_name = 'ESSAY_ESSAYID'
    restriction = 'be part of the ESSAY_SCENARIO identification'
    errors.append("Parameter {0} (set as {2}) must respect restriction: {1}\n".format(param_name, restriction, EssayParameters[param_name]))

  if(EssayParameters['ESSAY_CONFIGID'] not in EssayParameters['ESSAY_SCENARIO']):
    check = False
    #errors.append("Parameter {0} must respect restriction: {1}\n".format('ESSAY_CONFIGID', 'be part of the ESSAY_SCENARIO identification'))
    param_name = 'ESSAY_CONFIGID'
    restriction = 'be part of the ESSAY_SCENARIO identification'
    errors.append("Parameter {0} (set as {2}) must respect restriction: {1}\n".format(param_name, restriction, EssayParameters[param_name]))

  if(EssayParameters['ESSAY_CONFIGID'].lower() not in configFile.lower()):
    check = False
    #errors.append("Parameter {0} must respect restriction: {1}\n".format('ESSAY_CONFIGID', 'be part of the configuration filename'))
    param_name = 'ESSAY_CONFIGID'
    restriction = 'be part of the config filename'
    errors.append("Parameter {0} (set as {2}) must respect restriction: {1}\n".format(param_name, restriction, EssayParameters[param_name]))

  # summarises errors found
  if(len(errors) > 0):
    separator = "=============================================================================================================================\n"
    errorMsg = separator
    for i in range(0, len(errors)):
      errorMsg = errorMsg + errors[i]
    errorMsg = errorMsg + separator

  return(check, errorMsg)

# recovers the current essay configuration
def listEssayConfig():

  res = ''
  for e in sorted(EssayParameters.items()):
    res = res + "{0} : {1} (as {2})\n".format(e[0], e[1], type(e[1]))

  return res

def getMountedOn():

  if('PARAM_MOUNTEDON' in os.environ):
    res = os.environ['PARAM_MOUNTEDON'] + os.sep
  else:
    res = os.getcwd().split(os.sep)[-0] + os.sep

  return res

def headerfy(mask):
  res = re.sub('\:\d+\.\d+f', '', mask)
  res = re.sub('\:\d+d', '', res)
  return res

#-------------------------------------------------------------------------------------------------------------------------------------------
# General purpose definitions - metric spaces
#-------------------------------------------------------------------------------------------------------------------------------------------

def mmatrix(V, param_metric, params):
  return mm_generic(V, fmetric(param_metric), params)

def fmetric(metricID):

  if(metricID   == 'euclidean'):
    metricfunc = df_euclidean

  elif(metricID == 'cosine'):
    metricfunc = df_cosine

  elif(metricID == 'jaccard'):
    metricfunc = df_jaccard

  elif(metricID == 'aitchison'):
    metricfunc = df_aitchison

  elif(metricID == 'kullback-leibler'):
    metricfunc = df_kl

  elif(metricID == 'jensen-shannon'):
    metricfunc = df_js

  elif(metricID == 'areal'):
    metricfunc = df_areal

  else:
    raise ValueError

  return metricfunc

def mm_generic(V, df, params = None):
  # please note that this is a simplified version, not suited to large datasets
  nd = V.shape[0]  # number of units of V
  mm = np.zeros((nd, nd))
  for i in range(nd):
    v = V[i]
    for j in range(nd):
      w = V[j]
      mm[i][j] = df(v, w, params)

  return mm

def df_euclidean(v, w, params = None):
  return np.linalg.norm(v - w)

def df_cosine(v, w, params = None):

  # first computes the cosine similarity
  sim = v.dot(w) / (np.linalg.norm(v) * np.linalg.norm(w))

  # then converts it to unit distance (translate and rescale)
  res = 0.5 * (1 - sim)

  return res

def df_jaccard(v, w, params = None):

  # first computes the Jaccard similarity
  ub = v.shape[0]
  accmin = 0.0
  accmax = 0.0
  for k in range(ub):
    accmin += min(v[k], w[k])
    accmax += max(v[k], w[k])
  sim = accmin/accmax

  # then converts it to unit distance
  res = 1 - sim

  return res

def df_aitchison(v, w, params = None):

  v_  = clr(bmt(v))
  w_  = clr(bmt(w))
  res = np.linalg.norm(v_ - w_)

  return res

def df_kl(v, w, params = None):

  v_  = bmt(v)
  w_  = bmt(w)
  res = kldiv(v_, w_)

  return res

def df_js(v, w, params = None):

  v_  = bmt(v)
  w_  = bmt(w)
  vw  = (v_ + w_) / 2
  res = kldiv(v_, vw) + kldiv(w_, vw)

  return res

def df_areal(v, w, params):

  tomains, ulimits = params

  offer    = coords2poly(denseScores2coords(v, tomains, ulimits))
  demand   = coords2poly(denseScores2coords(w, tomains, ulimits))
  ardiff1  = matchp(offer, demand)
  ardiff2  = matchp(demand, offer)
  res      = ardiff1 + ardiff2

  return res

def kldiv(v, w):
  return sum([v[k] * np.log(v[k]/w[k]) for k in range(len(v))])

def bmt(v):

  # applies unit closure to the vector v
  v_prime = v / np.linalg.norm(v, 1)

  # checks the presence of zero-components in the unit-closed vector
  numOfZeros = sum(v_prime == 0)
  if(numOfZeros == 0):
    # no need to apply BMT to the unit-closed vector
    w = v_prime
  else:
    # need to apply BMT to get rid of zero-components
    D = v.shape[0]
    si = 1             # using Perks prior
    alpha_ij = 1 / D   # using Perks prior
    Ti = 1             # because v_prime is unit-closed

    # computes the adjustment to be made to non-zero-components
    sum_ik = numOfZeros * alpha_ij / (Ti + si)

    # creates a zero-free, unit-closed simplex vector
    w = np.zeros(D)
    for k in range(D):
      if(v_prime[k] == 0):
        w[k] = (Ti * alpha_ij) / (Ti + si)
      else:
        w[k] = v_prime[k] * (1 - sum_ik)

  return w

def clr(v):

  # obtains the centred log-ratio representation of v
  w = np.log(v / gmean(v))

  return w

def optimisedmm(mm, case_ids):
  """ metric matrix optimised for neighbourhood search
  """
  nd = mm.shape[0]
  if(nd != len(case_ids)):
    raise ValueError('Metric matrix do not conform to list of cases')

  om = defaultdict(list)
  for i in range(nd):
    case_id = case_ids[i]
    for j in range(nd):
      if(i != j):
        neighbour = case_ids[j]
        om[case_id].append((neighbour, mm[i,j]))

  for case_id in om:
    om[case_id].sort(key=lambda e: e[1])

  return om

def mm2text(mm):

  nd = mm.shape[0]

  header = 'mm\t' + '\t'.join(['v{0}'.format(i) for i in range(nd)])
  content = [header]

  for i in range(nd):
    buffer = ['v{0}'.format(i)]
    for j in range(nd):
      buffer.append('{0:7.3f}'.format(mm[i][j]))
    content.append('\t'.join(buffer))

  return '\n'.join(content)

def quantise(v, nseg):

  f,i = modf(v)
  lastdist = 1.0
  nearest  = 0.0
  for p in np.linspace(0.0, 1.0, nseg + 1):
    dist = abs(f - p)
    if(lastdist > dist):
      lastdist  = dist
      nearest   = p

  return i + nearest


#-------------------------------------------------------------------------------------------------------------------------------------------
# Problem-specific definitions: metric spaces
#-------------------------------------------------------------------------------------------------------------------------------------------

def computeDistMatrix(case_s, case_ids, instrument, param_metric):

  domains = instrument.tomains()
  ulimits = instrument.ulimits()

  V  = sparse2dense(case_s, case_ids, domains)
  params_mm = (domains, ulimits)
  mm = mmatrix(V, param_metric, params_mm)
  check, report = isMetric(mm, case_ids, case_s)
  return mm, check, report

def sparse2dense(case_s, case_ids, domains):

  nr = len(case_ids)
  nc = len(domains)
  V = np.zeros((nr, nc))

  for i in range(nr):
    case_id = case_ids[i]
    for j in range(nc):
      domain = domains[j]
      V[i][j] = case_s[case_id][domain]

  return V

def isMetric(mm, case_ids, case_s):

  violations = []
  m = mm.shape[0] # number of rows in the square distance matrix mm
  axioms = [None,
            'd(a,b) >= 0',               # Axiom 1
            'd(a,b) = 0 iif a = b',      # Axiom 2
            'd(a,b) = d(b,a)',           # Axiom 3
            'd(a,b) <= d(a,c) + d(c,b)'  # Axiom 4
           ]

  for i in range(m):
    for j in range(m):
      k = -1

      # checks if axiom 1 is satisfied: d(a,b) >= 0
      if(mm[i, j] < 0):
        violations.append((axioms[1], i, j, k, mm[i, j]))

      # checks if axiom 2 is satisfied: d(a,b) = 0 iif a = b
      if(mm[i, j] == 0 and i != j):
        scores_i = case_s[case_ids[i]]
        scores_j = case_s[case_ids[j]]
        if(scores_i != scores_j):
          violations.append((axioms[2], i, j, k, 0.0))

      # checks if axiom 3 is satisfied: d(a,b) = d(b,a)
      if(mm[i, j] != mm[j, i]):
        violations.append((axioms[3], i, j, k, mm[i, j] - mm[j, i]))

      # checks if axiom 4 is satisfied: d(a,b) <= d(a,c) + d(c,b)
      for k in range(m):
        if(k != i and k != j):
          diff = mm[i, j] - (mm[i, k] + mm[k, j])
          if(diff > ECO_PRECISION):
            violations.append((axioms[4], i, j, k, diff))

  res = len(violations) == 0

  # transforms violations into a readable format
  header  = 'Axiom\ti\tj\tk\tCase ID i\tCase ID j\tCase ID k\tValue'
  content = [header]
  for (axiom, i, j, k, val) in violations:
    case_id_k = case_ids[k] if k >= 0 else '-'
    buffer = '{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}'.format(axiom, i, j, k, case_ids[i], case_ids[j], case_id_k, val)
    content.append(buffer)

  return(res, '\n'.join(content))


#--------------------------------------------------------------------------------------------------
# Problem-specific definitions: polygonal representation, transformations, and operations
#--------------------------------------------------------------------------------------------------

def scores2coords(scores, tomains, ulimits):
  """
  Maps domain scores to coordinates that represent points over the diagonals of of a zero-centred,
  regular n-gon (with n as the number of domains explored by the instrument)
  -- assumes that adopting a regular n-gon as a template is an adequate trade-off between cogency of
     explanations and the correlational structure observed among the random variables that correspond
     to each instrument domain.
  # WARNING: axes run in counter clock-wise order because the computational geometry lib (shapely)
             follows the convention of using CCW primitives.
  """

  nd = len(tomains)     # number of domains
  ra = 2 * np.pi / nd   # angle between two axes, in radians
  axes  = [(tomains[i], i * ra) for i in range(nd)] # ~ [(domain LABEL, axis angle), ...]

  # converts the scores to the cartesian coordinates describing the vertices of a regular nd-polygon
  L = []
  for (domain, theta) in axes:
    r = scores[domain] / ulimits[domain] # scales the score of the current domain to the [0, 1] interval
    x = r * np.cos(theta)                # the x-coord of the vertice that corresponds to the current score
    y = r * np.sin(theta)                # the y-coord of the vertice that corresponds to the current score
    L.append((x, y))

  return L

def denseScores2coords(scores, tomains, ulimits):

  # maps each domain to the angular component of its polar coordinate
  # WARNING: axes must run in counter clock-wise order because shapely relies on CCW primitives
  nd = len(tomains)     # number of domains
  ra = 2 * np.pi / nd   # angle between two axes, in radians
  axes  = [(i, tomains[i], i * ra) for i in range(nd)] # ~ [(domain INDEX, domain LABEL, axis angle), ...]

  # converts the scores to the cartesian coordinates describing the vertices of a regular nd-polygon
  L = []
  for (domain_idx, domain_lbl, theta) in axes:
    r = scores[domain_idx] / ulimits[domain_lbl] # scales the score of the current domain to the [0, 1] interval
    x = r * np.cos(theta)                        # the x-coord of the vertice that corresponds to the current score
    y = r * np.sin(theta)                        # the y-coord of the vertice that corresponds to the current score
    L.append((x, y))

  return L

def coords2scores(coords, tomains, ulimits):

  scores = {}
  for i in range(len(tomains)):
    domain = tomains[i]
    scores[domain] = np.linalg.norm(np.array(coords[i])) * ulimits[domain]

  return scores

def coords2poly(coords):
  return Polygon(coords)

def poly2coords(polygon):

  xs, ys = polygon.exterior.coords.xy
  L = [(xs[i], ys[i]) for i in range(len(xs) - 1)]

  return L

def match(effects, scores, tomains, ulimits):
  """
  Estimates the expected benefit of treatment to a patient.
  Arguments:
     effects - the expected effects of the treatment,   according to a particular instrument
     scores  - the current health state of the patient, according to a particular instrument
     tomains - the representational template that must be adopted
     ulimits - the upper bounds for each domain

  Returns a positive real value representing the expected benefit (point estimate)
  -- assumes that the underlying representational model* has an adequate correspondence to reality.

  A technical note:
  -- when the representation for a treatment is induced from expert advice (e.g., clinical judgements
     provided by a medical professional), their advice is taken as the ground truth and the treatment
     representation is chosen so that, under this operation (match), it produces a score that
     corresponds to a measure of expected benefit of the treatment to the patient:

     U = R+^d, the set into which the domain scores are embedded, with d as the number of domains.

     M: U x U -> R                                 (this is the match operation)
       (t , p) |-> || P(t) \ P(p) ||               (the result is an estimate of expected benefit)

     P: U -> Poly(d)                               (d corresponds to the number of vertices)
        p |-> polygon representing the patient scores obtained by applying an arbitrary instrument.
        t |-> polygon representing the treatment expected effects, obtained as described earlier.

     \: Poly(d) x Poly(d) -> Polygons
            (p ,  t) |-> a set of polygons representing points in p that are not covered by t
                         the resulting polygons are not necessarily regular, and do not contain (0,0)

     ||.||: Polygons -> R+
       s |-> non-negative real number that represents the sum of the area of each non-overlapping
             polygon in s

     Example: suppose we want to choose a representation for an arbitrary treatment A about which we
     have a set of tuples (patient, best treatment) such that some of these tuples refer to treatment
     A and other refer to other treatments, collectively referred to as not-A. The strategy we follow
     to choose a proper representation for treatment A is to find a representation such that:

     (1) under the match operation with scores obtained from all of the patients to whom an expert
         indicated A as their best treatment, the resulting estimate of expected benefit is larger
         than any estimates obtained for other treatments, namely not-A. Formally:
         M(A, p) > M(not-A, p) \forall p | best_treatment(p) = A.

     (2) under the match operation with scores obtained from all of the patients to whom an expert
         indicated not-A as their best treatment, some of the resulting estimates of expected
         benefit is smaller than the estimate obtained for treatment A. Formally:
         \exists p | best_treatment(p) \neq A and M(A, p) < M(not-A, p).

     It follows from these properties that the score produced by M can be employed to rank the
     available treatments in order of expected benefit to a particular patient, and the produced
     ranking agrees with the expert advice, which is taken as the ground truth at this stage of
     the project. It must also be stated that the process of searching for such a representation may
     fail because such representation may not exist. To the purpose of this project, sub-optimal
     solutions are acceptable, and the search process is modelled as a discrete optimisation problem
     (see optimiseOffers function).
  """

  # estimates the expected benefit of treatment
  offer    = coords2poly(scores2coords(effects, tomains, ulimits))
  demand   = coords2poly(scores2coords(scores,  tomains, ulimits))

  try:
    expected = offer.difference(demand)
    res = expected.area
  except ValueError:
    res = 0.0

  return res

def matchp(offer, demand):

  # estimates the expected benefit of treatment
  expected = offer.difference(demand)

  return expected.area

def absorb(effects, scores, tomains, ulimits):
  """
  Estimates the expected outcome of treatment to a patient.
  Arguments:
     effects - the expected effects of the treatment,   according to a particular instrument
     scores  - the current health state of the patient, according to a particular instrument
     tomains - the representational template that must be adopted
     ulimits - the upper bounds for each domain

  Returns an estimate of the scores obtained by the patient after the completion of the treatment.
  -- assumes that the underlying representational model* has an adequate correspondence to reality.
  -- assumes that the patient fully adheres to the treatment.

  Some technical notes:

  -- when the match operation is used to estimate the expected benefit of a treatment to a patient,
     the estimate is obtained by computing the sum of the area of a set of polygons resulting from a
     difference operation in abstract space. The problem is that this set of polygons are not embedded
     in the representational space (i.e., they are not elements of Poly(d)). This means that a
     representation of the treatment outcome (i.e., an estimate of the future scores obtained by the
     patient after completing the treatment) is not a byproduct of the match operation.

  -- to tackle this problem, the absorb operation is defined as follows: the patient scores, obtained
     before and after the patient is submitted to the treatment, if compared, must ideally produce a
     measure of improvement (i.e., observed benefit) that is equal to the estimate of expected benefit
     obtained by the match operation. Formally:

     A: U x U -> U                                 (this is the absorb operation)
       (t x p) |-> p' | B(p', p) = B(t, p)         (the result is an estimate of the patient scores
                                                    obtained after the completion of the treatment)

  -- we assume that the treatment cannot produce harm to the patient. A corollary to this is that it
     must the case that all scores obtained in the reassessment of the patient are equal to or larger
     than the respective scores obtained in the previous assessment (i.e., p'_i >= p_i, i = 1 .. d).

  -- in solving the system of equations, we begin by assuming a maximum absorption hypothesis, in
     which the expected treatment effects are entirely absorbed by the patient: p'_i = max(p_i, t_i).
     An important corollary to this assumption is that the treatment produces a benefit that is
     equal to or larger than the expected benefit estimated by the match operation:
     p'_i = max(p_i, t_i) => B(p', p) >= B(t, p)).

  -- to find a more suitable solution than the initial estimate described above, the obtained scores
     (p') are proportionally shrinked until this condition is satisfied: B(p', p) <= B(t, p). The
     maximum length of | B(p', p) - B(t, p) | is controlled by the parameter n, which is used to
     determine the number of contractions to perform. It must also be noted that, as a result from
     this process of approximation:
     (1) the results from the absorb operation are biased towards underestimation;
     (2) the corollary holds: t_i >= p'_i \foralll i: t_i > p_i.
  """

  # identifies which domains are improved by the current treatment
  watermark = (-1, None, None)
  adjust  = []
  outcome = {}
  for domain in scores:
    diff = effects[domain] - scores[domain]
    if(diff > 0):
      # assumes the treatment produces maximal benefit (following the maximum absorption hypothesis),
      # but this is temporary: this initial estimate is adjusted later if necessary.
      outcome[domain] = effects[domain]
      adjust.append(domain)
      if(watermark[0] == -1 or diff < watermark[0]):
        watermark = (diff, scores[domain], effects[domain])
    else:
      # assumes the treatment does not produce any benefit,
      # and also that the treatment cannot harm the patient
      outcome[domain] = scores[domain]

  # estimates the expected benefit of treatment
  offer    = coords2poly(scores2coords(effects, tomains, ulimits))
  demand   = coords2poly(scores2coords(scores,  tomains, ulimits))
  expected = offer.difference(demand)

  # proportionally shrinks the level of those domains that require adjustment,
  # until the expected benefit is approximately attained
  (_, a, b) = watermark
  if(a is not None):
    n = 100
    shrinking = np.exp(np.log(a/b) / n)
    for i in range(n):
      for area in adjust:
        outcome[area] *= shrinking
      offer   = coords2poly(scores2coords(outcome, tomains, ulimits))
      benefit = offer.difference(demand)

      if(expected.area >= benefit.area):
        break

  return outcome

#--------------------------------------------------------------------------------------------------
# Problem-specific definitions: dataset preprocessing
#--------------------------------------------------------------------------------------------------

def loadMask(sourcepath, filename, encoding, separator = ECO_FIELDSEP):

  content = []
  for e in file2List(os.path.join(*sourcepath, filename), separator, '', encoding):
    content.append(tuple(e))

  return content

def loadCaseHistory(sourcepath, filename, encoding, case_t, separator = ECO_FIELDSEP):

  content = file2List(os.path.join(*sourcepath, filename), separator, '', encoding)

  # transforms the header row into a column-name dictionary
  header = content[0]
  ncols  = len(header)
  field2pos = {header[pos]: pos for pos in range(ncols)}

  # transforms the remaining rows into the case history, descriptions of which follow the case template
  case_h   = {}
  case_e  = defaultdict(list)
  case_ids = []
  for detail in content[1:]:

    if(len(detail) == ncols):

      # assembles a new case following the case template
      case = {}
      msgs = []
      for (target, source, lom, lov, opt) in case_t:

        if(len(source) > 0):
          raw_val = detail[field2pos[source]]
        else:
          raw_val = ''

        val, msg = recast(source, raw_val, lom, lov, opt)

        case[target] = val
        msgs += msg

      # checks uniqueness of case id
      if(case['CaseID'] in case_ids):
        msgs.append('Found duplicated caseID: [{0}]'.format(case['CaseID']))
      else:
        case_ids.append(case['CaseID'])

      # adds the newly assembled case into the case history
      case_h[case['CaseID']] = case
      case_e[case['CaseID']] += msgs

    else:
      tsprint('-- Inconsistent detail row: {0}'.format(detail))

  return case_h, case_e

def loadTreatmentHistory(sourcepath, filename, encoding, treatment_t, separator = ECO_FIELDSEP):

  content = file2List(os.path.join(*sourcepath, filename), separator, '', encoding)

  # transforms the header row into a column-name dictionary
  header = content[0]
  field2pos = {header[pos]: pos for pos in range(len(header))}

  # remaining rows converted into the treatment history, descriptions of which follow the treatment template
  treatment_h  = {}
  treatment_e  = defaultdict(list)
  for detail in content[1:]:

    # assembles a new recommended treatment instance following the treatment template
    treatment = {}
    msgs = []
    for (target, source, lom, lov, opt) in treatment_t:

      if(len(source) > 0):
        raw_val = detail[field2pos[source]]
      else:
        raw_val = ''

      val, msg = recast(source, raw_val, lom, lov, opt)

      treatment[target] = val
      msgs += msg

    # adds the newly assembled treatment instance into the treatment history
    treatment_h[treatment['CaseID']] = treatment
    treatment_e[treatment['CaseID']] += msgs

  return treatment_h, treatment_e

def recast(source, raw_val, lom, lov, opt):

  msg = []
  if(len(raw_val) > 0):

    if(lom == 'Ordinal'):
      if(raw_val in lov):
        val = int(raw_val)
      else:
        val = np.nan
        msg.append('value is outwith its range in field [{0}] ({1} not in {2})'.format(source, raw_val, lov))

    elif(lom == 'Categorical'):
      if(raw_val in lov):
        val = raw_val.strip()
      else:
        val = None
        msg.append('value is outwith its range in field [{0}] ({1} not in {2})'.format(source, raw_val, lov))

    elif(lom == 'Multi-valued'):

      val = []
      for term in raw_val.split(ECO_LISTSEP):
        term = term.strip()
        if(term in lov):
          val.append(term)
        else:
          msg.append('value is outwith its range in field [{0}] ({1} not in {2})'.format(source, term, lov))

    elif(lom == 'Textual'):
      val = raw_val

    elif(lom == 'Date'):
      val = raw_val

    else:
      val = raw_val

  else:

    # the value is missing
    if(lom == 'Ordinal'):
      val = np.nan

    elif(lom == 'Textual'):
      val = None

    elif(lom == 'Date'):
      val = None

    else:
      val = None

    if(opt == 'mandatory'):
      msg.append('mandatory value is missing for field [{0}]'.format(source))

  return val, msg

def computeCaseScores(case_h, case_e, instrument, precision):

  case_s = {}
  case_ids = sorted(list(case_h))
  for case_id in case_ids:

    case = case_h[case_id]
    scores, msgs = instrument.scorer(case, precision)

    if(scores is not None):
      case_s[case_id] = scores

    if(len(msgs) > 0):
      case_e[case_id] += msgs

  return case_s, case_e

def sampleCaseIDs(case_s, param_exclude, param_caseids, param_samplesize):

  if(len(param_exclude) > 0):
    tsprint('-- the following cases have been excluded: {0}'.format(param_exclude))
    for case_id in param_exclude:
      case_s.pop(case_id)

  if(len(param_caseids) > 0):
    case_ids = param_caseids
  elif(param_samplesize > 0):
    case_ids = sorted(sample(list(case_s), param_samplesize))
  else:
    case_ids = sorted(list(case_s))

  sample_order = {case_ids[i] : i for i in range(len(case_ids))}
  sample_redro = {i : case_ids[i] for i in range(len(case_ids))}
  tsprint('** case history  has {0:3d} samples remaining after resampling'.format(len(case_ids)))

  return (case_ids, sample_order, sample_redro)

def errors2text(case_e):

  content = []
  case_ids = sorted(list(case_e))
  for case_id in case_ids:
    content.append('Case [{0}]:'.format(case_id))
    for msg in case_e[case_id]:
      content.append('  {0}'.format(msg))

  return '\n'.join(content)

def scores2text(case_s):

  case_ids = sorted(list(case_s))

  domains = []
  for case_id in case_ids:
    for domain in case_s[case_id]:
      domains.append(domain)
  domains = sorted(set(domains))

  content = ['Case ID\t' + '\t'.join(domains)]
  for case_id in case_ids:
    buffer = [case_id]
    for domain in domains:
      buffer.append('{0:6.3f}'.format(case_s[case_id][domain]))
    content.append('\t'.join(buffer))

  return '\n'.join(content)

def assignTreatments(case_ids, labels, sample_order):

  treatment_h = {}
  assignments  = defaultdict(list)
  for case_id in case_ids:
    intervention = chr(ord('A') + labels[sample_order[case_id]])
    treatment_h[case_id] = {'CaseID': case_id, 'Treatments': [intervention]}
    assignments[intervention].append(case_id)

  return treatment_h, assignments

def splitDataset(assignments, param_split):

  splits = defaultdict(list)
  detailed_splits = defaultdict(lambda: defaultdict(list))
  for intervention in assignments:
    aux = assignments[intervention]
    ss = int(len(aux) * param_split)
    ss = 1 if (ss == 0 and len(aux) > 1) else ss
    test_sample  = sample(aux, ss)
    train_sample = [case_id for case_id in aux if case_id not in test_sample]

    splits['Test']  += test_sample
    splits['Train'] += train_sample

    detailed_splits['Test'][intervention]  = test_sample
    detailed_splits['Train'][intervention] = train_sample

  return splits, detailed_splits

def splits2text(case_ids_ao, treatment_h, splits):

  header = 'Case ID\tInterventions\tPartition'
  content = [header]

  for case_id in case_ids_ao:

    if(case_id in splits['Train']):
      partition = 'Train'
    elif(case_id in splits['Test']):
      partition = 'Test'
    else:
      partition = 'None'

    buffer = '{0}\t{1}\t{2}'.format(case_id, ', '.join(treatment_h[case_id]['Treatments']), partition)
    content.append(buffer)

  return '\n'.join(content)


#--------------------------------------------------------------------------------------------------
# Problem-specific definitions: demand (i.e., patient needs) stratification
#--------------------------------------------------------------------------------------------------

def plotDendrogram(mm, case_ids, metric, cutoff, linkage, sample_redro, params, plotTitle, filename):

  # unpacks parameters
  (unitsizes, fontgon, innerseps, xoff, yoff, titleoffset, tagoffset, saveit, targetpath) = params
  (unitsizew, unitsizeh) = unitsizes
  (left, bottom, right, top, wspace, hspace) = innerseps
  (xoffcol, yoffcol, xoffrow, yoffrow) = titleoffset

  # applies hierarchical clustering using the external metric matrix mm
  if(linkage == 'ward'):
    model = AgglomerativeClustering(distance_threshold = cutoff,
                                    compute_full_tree  = True,
                                    n_clusters = None,
                                    linkage    = linkage)
  else:
    model = AgglomerativeClustering(distance_threshold = cutoff,
                                    compute_full_tree  = True,
                                    n_clusters = None,
                                    linkage    = linkage,
                                    affinity   = 'precomputed')
  model = model.fit(mm)
  res = copy(model.labels_)

  # plots a dendrogram based on the clustering results and obtains the order
  # in which cases should be plotted so as to improve the sense of similarity (affinity order) in the reader
  fig = plt.figure(figsize = (6 * unitsizew, 2 * unitsizeh))
  plt.title(plotTitle)
  plt.gca().minorticks_on()
  leaves = createDendrogram(model, truncate_mode = None, distance_sort = 'ascending')
  case_ids_ao = [sample_redro[i] for i in leaves]
  plt.xlabel("Number of points in node (or index of point if no parenthesis).")


  if(saveit):
    #print('-- saving the plot.')
    if(not os.path.exists(os.path.join(*targetpath))):
      os.makedirs(os.path.join(*targetpath))
    plt.savefig(os.path.join(*targetpath, '{0}'.format(filename)), bbox_inches = 'tight')
    plt.close(fig)

  else:
    print('-- rendering the plot.')
    plt.show()
    print('   figure width is {0} and height is {1}'.format(fig.get_figwidth(), fig.get_figheight()))

  return case_ids_ao, res

def createDendrogram(model, **kwargs):

    # creates the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # plots the corresponding dendrogram
    #res = dendrogram(linkage_matrix, **kwargs)
    res = dendrogram(linkage_matrix, truncate_mode = None, distance_sort = 'ascending', orientation = 'top')
    return res['leaves']

def plotDemandGrid(case_ids, case_s, templates, ulimits, sample_order, params, plotTitle, filename):

  # unpacks parameters
  (unitsizes, fontgon, innerseps, xoff, yoff, titleoffset, tagoffset, saveit, targetpath) = params
  (unitsizew, unitsizeh) = unitsizes
  (xoffcol, yoffcol, xoffrow, yoffrow) = titleoffset

  # determines the grid size
  nd    = len(ulimits)    # number of domains
  nrows = len(case_ids)   # number of rows    in the grid
  ncols = len(templates)  # number of columns in the grid
  fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * unitsizew, nrows * unitsizeh))
  #plt.suptitle(plotTitle, fontsize = 1.8 * fontgon['size'])

  template_ids = sorted(list(templates))
  for i in range(nrows):
    case_id = case_ids[i]

    for j in range(ncols):
      template_id = template_ids[j]
      tomains = templates[template_id][0]

      pos = ncols * i + j + 1
      plt.subplot(nrows, ncols, pos)
      plt.subplots_adjust(left   = innerseps['left'],
                          bottom = innerseps['bottom'],
                          right  = innerseps['right'],
                          top    = innerseps['top'],
                          wspace = innerseps['wspace'],
                          hspace = innerseps['hspace'])

      # plots the guiding elements of the diagram
      gpc, glc = drawGuides(tomains, ulimits, xoff, yoff, fontgon)
      plt.gca().add_collection(gpc)
      plt.gca().add_collection(glc)

      # plots the n-gon induced from the scores
      dpc = drawDiagram(case_s[case_id], tomains, ulimits, ECO_PTRN_DEMAND)
      plt.gca().add_collection(dpc)

      # adds the template name as column title
      if(i == 0):
        plt.text(xoffcol, yoffcol, template_id,
                 transform=plt.gca().transAxes,
                 fontsize= 1.5 * fontgon['size'],
                 horizontalalignment='center')

      # adds case_id as row titles
      if(j == 0):
        plt.text(xoffrow, yoffrow, '{0}\n({1})'.format(case_id, sample_order[case_id]),
                 transform=plt.gca().transAxes,
                 fontsize= 1.5 * fontgon['size'],
                 rotation='vertical',
                 verticalalignment='center',
                 horizontalalignment='center')

      # disables the cartesian axes and rescales the plotting, for visual clarity
      plt.gca().autoscale()
      plt.gca().axis('off')

  if(saveit):
    #print('-- saving the plot.')
    if(not os.path.exists(os.path.join(*targetpath))):
      os.makedirs(os.path.join(*targetpath))
    plt.savefig(os.path.join(*targetpath, '{0}'.format(filename)), bbox_inches = 'tight')
    plt.close(fig)

  else:
    print('-- rendering the plot.')
    plt.show()
    print('   figure width is {0} and height is {1}'.format(fig.get_figwidth(), fig.get_figheight()))

def ao2text(case_ids_ao, labels, sample_order):

  header = 'Case ID\tSample Order\tCluster'
  content = [header]

  for case_id in case_ids_ao:
    buffer = '{0}\t{1}\t{2}'.format(case_id, sample_order[case_id], chr(ord('A') + labels[sample_order[case_id]]))
    content.append(buffer)

  return '\n'.join(content)

def treatment2text(case_ids_ao, treatment_h):

  header = 'Case ID\tInterventions'
  content = [header]

  for case_id in case_ids_ao:
    buffer = '{0}\t{1}'.format(case_id, ', '.join(treatment_h[case_id]['Treatments']))
    content.append(buffer)

  return '\n'.join(content)

#--------------------------------------------------------------------------------------------------
# Problem-specific definitions: learning offer (i.e., treatment) representation
#--------------------------------------------------------------------------------------------------

def chromosome2scores(chromosome, treatment_ids, tomains, ulimits):

  treatment_s = {}

  nd = len(ulimits)          # number of dimensions
  nt = len(chromosome) // nd # number of treatments
  for i in range(nt):
    treatment_id = treatment_ids[i]
    scores = {}
    for j in range(nd):
      domain = tomains[j]
      scores[domain] = chromosome[i * nd + j]
    treatment_s[treatment_id] = scores

  return treatment_s

def callbackfn1(chromosome, convergence):

  GlbBuffer.append(chromosome)
  return False

def callbackfn2(chromosome, convergence):

  GlbBuffer.append((chromosome, convergence))
  return False

def optimiseOffers(case_ids, case_s, ulimits, llimits, tomains, treatment_h, params, workers, overridePopsize=0):

  # a global variable that will be used by a callback function to store intermediary solutions from
  # the scipy.differential_evolution implementation
  global GlbBuffer
  GlbBuffer = []

  # unpacks parameters
  (strategy, maxiter, popsize, mutation, recombination) = params
  if(overridePopsize > 0):
    popsize = overridePopsize

  # computes the demand representations for the sample of cases
  demands = {}
  for case_id in case_ids:
    demands[case_id] = coords2poly(scores2coords(case_s[case_id], tomains, ulimits))

  # recovers the list of treatments recommended in the sample of cases
  L = [treatment_h[case_id]['Treatments'] for case_id in case_ids]
  treatment_ids = sorted(set([item for sublist in L for item in sublist]))

  # prepares the remaining parameters required by the optimiser
  bounds = [(llimits[domain], ulimits[domain]) for domain in tomains for _ in treatment_ids]
  args   = (case_ids, demands, treatment_ids, treatment_h, tomains, ulimits)

  # seeks for offer representations that satisfy the constraints posed by the objective function
  res = differential_evolution(objfn2,
                               bounds,
                               args     = args,
                               strategy = strategy,
                               mutation = mutation,
                               recombination = recombination,
                               maxiter  = maxiter,
                               popsize  = popsize,
                               callback = callbackfn2,
                               polish   = True,
                               #init     = 'random',
                               workers  = workers, # only available in scipy 1.2+
                               updating = 'deferred',
                               disp     = False,
                               seed     = ECO_SEED)

  treatment_s = chromosome2scores(res.x, treatment_ids, tomains, ulimits)
  offers  = {}
  for treatment in treatment_s:
    offers[treatment] = coords2poly(scores2coords(treatment_s[treatment], tomains, ulimits))

  return (treatment_s, offers, GlbBuffer)

def objfn1(chromosome, *args):
  """
  objective function that seeks to optimise precision@1 (by searching for treatment reps that
     obtains larger expected benefit for those cases in which it is the preferred treatment,
     compared to the remaining cases).
  since this function is being used with scipy DE routine, which seeks to minimize the objective
     function, the returned value is the precision@1 obtained from applying the treatments coded
     in the chromosome to a sample of cases, multiplied by -1.
  """

  # unpacks parameters
  (case_ids, demands, treatment_ids, treatment_h, tomains, ulimits) = args

  # converts the current solution to its corresponding treatment scores
  treatment_s = chromosome2scores(chromosome, treatment_ids, tomains, ulimits)

  # computes the fitness of the current solution by applying "vertical" constraints
  fitness = 1.0
  for treatment_id in treatment_s:

    # converts the scores of the current treatment to their corresponding offer representation
    offer = coords2poly(scores2coords(treatment_s[treatment_id], tomains, ulimits))

    # recovers the list of cases in which the current treatment is the preferred one
    R = {case_id for case_id in case_ids if treatment_id == treatment_h[case_id]['Treatments'][0]}

    # recovers the list of cases in which the current treatment is NOT the preferred one
    R_c = set(case_ids).difference(R)

    # computes the expected benefit of the current treatment to all cases
    alpha = {case_id: matchp(offer, demands[case_id]) for case_id in case_ids}

    # assesses the quality of the boundary between R and R_c provided by the current offer rep
    # by counting the number of times the expected benefit of a case in R is larger than one in R_c
    (c, d) = (0, 0)
    for p in R:
      for q in R_c:
        c += 1 if (alpha[p] > alpha[q]) else 0
        d += 1

    fitness *= c/d if d != 0 else 0

    tsprint('treatment {0}: {1} out of {2} hits'.format(treatment_id, c, d))

  return -fitness

def objfn2(chromosome, *args):
  """
  objective function that seeks to optimise precision@k, with k being the number of treatments
    informed in each case thistory).
  since this function is being used with scipy DE routine, which seeks to minimize the objective
     function, the returned value is the precision@k obtained from applying the treatments coded
     in the chromosome to a sample of cases, multiplied by -1.
  """

  # unpacks parameters
  (case_ids, demands, treatment_ids, treatment_h, tomains, ulimits) = args

  # converts the current solution to its corresponding treatment scores
  treatment_s = chromosome2scores(chromosome, treatment_ids, tomains, ulimits)

  # converts the scores of each treatment to their corresponding offer representations
  offers = {}
  for treatment in treatment_s:
    offers[treatment] = coords2poly(scores2coords(treatment_s[treatment], tomains, ulimits))

  # computes the number of hits achieved by the solution
  tries = 0
  hits  = 0
  for case_id in demands:
    demand = demands[case_id]
    expert_advice = treatment_h[case_id]['Treatments']
    nt = len(expert_advice) # number of treatments associated with the case in hand

    # applies the current solution to obtain a ranking of available treatments
    L = []
    for treatment_id in offers:
      offer   = offers[treatment_id]
      alpha   = matchp(offer, demand)
      L.append((treatment_id, alpha))
    L = sorted([(treatment_id, alpha) for (treatment_id, alpha) in L], key = lambda e: -e[1])
    system_advice = [treatment_id for (treatment_id, alpha) in L if alpha != 0.0]

    acc = 0.0
    for i in range(nt):
      treatment_id = expert_advice[i]
      try:
        j = system_advice.index(treatment_id)
        acc += 1 / (abs(i - j) + 1)
      except ValueError:
        #acc += 0.0
        pass

    hits  += acc
    tries += nt

  fitness = hits/tries

  return -fitness

def details2text(details, sample_order):

  header  = 'CaseID\tSample Order\tPrecision\tExpert Advice\tSystem Advice\tDetails'
  content = [header]
  for case_id in details:
    (precision, expert_advice, system_advice, L) = details[case_id]
    content.append('{0}\t{1}\t{2:5.3f}\t{3}\t{4}\t{5}'.format(case_id, sample_order[case_id], precision, expert_advice, system_advice, L))

  return '\n'.join(content)

def plotOfferGrid(case_ids, case_s, treatment_s, details, tomains, ulimits, sample_order, params, plotTitle, filename):

  # | (1) empty      | (2) offer1 | (3) offer2 | ... | (m) offer m | -> i == 0
  # | (m+1) demand 1 | (m+2) d*o  | (m+3) d*0  | ... | (2m) d*0    |
  #  ----------------
  #         v
  # |     j == 0     |

  # unpacks parameters
  (unitsizes, fontgon, innerseps, xoff, yoff, titleoffset, tagoffset, saveit, targetpath) = params
  (unitsizew, unitsizeh) = unitsizes
  (xoffcol, yoffcol, xoffrow, yoffrow) = titleoffset
  (xofftag, yofftag, rofftag) = tagoffset

  treatment_ids = sorted(list(treatment_s))

  # determines the grid size
  nd    = len(ulimits)            # number of domains
  nrows = len(case_ids)      + 1  # number of rows    in the grid, plus one (top  header)
  ncols = len(treatment_ids) + 1  # number of columns in the grid, plus one (left header)
  fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * unitsizew, nrows * unitsizeh))


  for i in range(nrows):
    case_id = case_ids[i - 1] if (i > 0) else None

    for j in range(ncols):
      treatment_id = treatment_ids[j - 1] if (j > 0) else None

      pos = ncols * i + j + 1
      plt.subplot(nrows, ncols, pos)
      plt.subplots_adjust(left   = innerseps['left'],
                          bottom = innerseps['bottom'],
                          right  = innerseps['right'],
                          top    = innerseps['top'],
                          wspace = innerseps['wspace'],
                          hspace = innerseps['hspace'])

      if(pos == 1):

        # this cell remains empty; no case_id or treatment_id is assignable to it
        # plt.suptitle(plotTitle, fontsize = 1.8 * fontgon['size'])
        None

      elif(i == 0):

        # top header - plots a diagram for the treatment (offer)
        gpc, glc = drawGuides(tomains, ulimits, xoff, yoff, fontgon)
        plt.gca().add_collection(gpc)
        plt.gca().add_collection(glc)

        opc = drawDiagram(treatment_s[treatment_id], tomains, ulimits, ECO_PTRN_OFFER)
        plt.gca().add_collection(opc)

        # adds the treatment ID name as column title
        plt.text(xoffcol, yoffcol, treatment_id,
                 transform=plt.gca().transAxes,
                 fontsize= 1.5 * fontgon['size'],
                 horizontalalignment='center')

      elif(j == 0):

        # left header - plots a diagram for the patient (demand)
        gpc, glc = drawGuides(tomains, ulimits, xoff, yoff, fontgon)
        plt.gca().add_collection(gpc)
        plt.gca().add_collection(glc)

        dpc = drawDiagram(case_s[case_id], tomains, ulimits, ECO_PTRN_DEMAND)
        plt.gca().add_collection(dpc)

        # adds the case ID name as row  title
        plt.text(xoffrow, yoffrow, '{0}\n({1})'.format(case_id, sample_order[case_id]),
                 transform=plt.gca().transAxes,
                 fontsize= 1.5 * fontgon['size'],
                 rotation='vertical',
                 verticalalignment='center',
                 horizontalalignment='center')

      else:

        # plots a detail cell (in this order: guiding, demand, offer, and benefit)

        # plots the guiding elements
        gpc, glc = drawGuides(tomains, ulimits, xoff, yoff, fontgon)
        plt.gca().add_collection(gpc)
        plt.gca().add_collection(glc)

        # plots the demand over the guiding elements
        dpc = drawDiagram(case_s[case_id], tomains, ulimits, ECO_PTRN_DEMAND)
        plt.gca().add_collection(dpc)

        # plots the offer over the demand
        opc = drawDiagram(treatment_s[treatment_id], tomains, ulimits, ECO_PTRN_OFFER)
        plt.gca().add_collection(opc)

        # plots the benefit over the offer
        demand  = coords2poly(scores2coords(case_s[case_id],           tomains, ulimits))
        offer   = coords2poly(scores2coords(treatment_s[treatment_id], tomains, ulimits))
        benefit = offer.difference(demand)
        val     = benefit.area

        if(benefit != None):
          if(type(benefit) == Polygon):
            benefit = [benefit]
          for poly in benefit:
            vertices = list(poly.exterior.coords)
            pc = drawPolygon(vertices, ECO_PTRN_BENEFIT)
            plt.gca().add_collection(pc)

        # plots the expected benefit as text tag
        if(val > 0.0001):
          system_advice = details[case_id][3][0][0]
          if(treatment_id == system_advice):
            bbox = {'fc': (.8, 1., .8), 'ec': (.5, 1., .5), 'boxstyle': 'round'}
          else:
            bbox = {'fc': (1., .8, .8), 'ec': (1., .5, .5), 'boxstyle': 'round'}
          plt.text(xofftag, yofftag, '{0:6.4f}'.format(val), fontgon,
                   transform = plt.gca().transAxes,
                   rotation  = rofftag,
                   bbox = bbox,
                   verticalalignment   ='center',
                   horizontalalignment ='center')

      # disables the cartesian axes and rescales the plotting, for visual clarity
      plt.gca().autoscale()
      plt.gca().axis('off')

  if(saveit):
    #print('-- saving the plot.')
    if(not os.path.exists(os.path.join(*targetpath))):
      os.makedirs(os.path.join(*targetpath))
    plt.savefig(os.path.join(*targetpath, '{0}'.format(filename)), bbox_inches = 'tight')
    plt.close(fig)

  else:
    print('-- rendering the plot.')
    plt.show()
    print('   figure width is {0} and height is {1}'.format(fig.get_figwidth(), fig.get_figheight()))

def drawGuides(tomains, ulimits, xoff, yoff, fontgon):

  nd = len(ulimits)

  # plots the guiding elements of the diagram - concentric n-gons
  guiding, lws, lss = [], [], []
  for r in [.2, .4, .6, .8, 1.0]:
    scores = {domain: r * ulimits[domain] for domain in tomains}
    vertices = scores2coords(scores, tomains, ulimits)
    guiding.append(vertices)
    lws.append(.5  if r < 1.0 else  2)
    lss.append(':' if r < 1.0 else '-')

  pc = PolyCollection(guiding,
                      edgecolors = ['black' for _ in vertices],
                      facecolors = ['none'  for _ in vertices],
                      linewidths = lws,
                      linestyle  = lss)

  # plots the guiding elements of the diagram - domain axes
  lines  = []
  origin = (0.0, 0.0)
  for k in range(nd):
    domain = tomains[k]

    # adds the domain axis
    vertex = vertices[k] # vertices is the list of vertices that corresponds to the outer guiding element
    lines.append([origin, vertex])

    # adds the domain label
    (x, y) = vertex
    plt.text(x + xoff[k], y + yoff[k], domain, fontgon)

  lc = LineCollection(lines, colors = ['black' for _ in lines], linewidth = .5)

  return pc, lc

def drawDiagram(scores, tomains, ulimits, pattern):

  # plots the n-gon induced from the scores using the specified visual pattern
  vertices = scores2coords(scores, tomains, ulimits)
  pc = drawPolygon(vertices, pattern)

  return pc

def drawPolygon(vertices, pattern):

  if(pattern   == ECO_PTRN_DEMAND):
    (ptrn_edgecolor, ptrn_facecolor, alpha) = ('blue', 'lightblue', 1.0)
  elif(pattern == ECO_PTRN_OFFER):
    (ptrn_edgecolor, ptrn_facecolor, alpha) = ('red',  'none', 0.80)
  elif(pattern == ECO_PTRN_BENEFIT):
    (ptrn_edgecolor, ptrn_facecolor, alpha) = ('red', 'lemonchiffon', 0.80)
  else:
    raise ValueError

  # plots the n-gon specified by the vertices according to a visual pattern
  pc = PolyCollection([vertices],
                      edgecolors = [ptrn_edgecolor for _ in vertices],
                      facecolors = [ptrn_facecolor for _ in vertices],
                      alpha      = alpha,
                      linewidths = 1)

  return pc

def retraceOfferOptimisation(history, splits, dsplits, demands, tomains, ulimits, treatment_h, params, plotTitle, filename):

  # unpacks parameters
  (unitsizes, fontgon, innerseps, xoff, yoff, titleoffset, tagoffset, dpi, fps,
   grid, row_script, saveit, targetpath) = params
  (unitsizew, unitsizeh) = unitsizes
  (xoffcol, yoffcol, xoffrow, yoffrow) = titleoffset
  (xofftag, yofftag, rofftag) = tagoffset
  (nrows, ncols) = grid # grid size

  # recovers the list of treatments recommended in the sample of cases
  treatment_ids = sorted(dsplits['Train'])

  # recovers the trail of the evolution of offer representation during the optimisation
  data = history # data stored in the GlbBuffer during optimiseOffer
  nsteps = len(data)
  tsprint('   optimisation process ran for {0} generations'.format(nsteps))

  # determines the grid size
  nd = len(ulimits)   # number of domains
  fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * unitsizew, nrows * unitsizeh))

  # initialises the movie writer
  FFMpegWriter = manimation.writers['ffmpeg']
  metadata = dict(title=plotTitle, artist='Matplotlib', comment=plotTitle)
  writer = FFMpegWriter(fps=fps, metadata=metadata)

  # initialises the background by plotting the grid guidelines
  it = 0
  fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * unitsizew, nrows * unitsizeh))
  newplotTitle = plotTitle + ' (generation {0:4d})'.format(it)

  for (treatment_id, pos) in row_script:
    plt.subplot(nrows, ncols, pos)

    if(treatment_id == '.'):

      plt.gca().set_title('Progress')
      plt.gca().set_xlim([0.0, nsteps])
      plt.gca().set_ylim([0.0, 1.01])
      plt.gca().set_xlabel('iteration')

    else:
      plt.subplots_adjust(left   = innerseps['left'],
                          bottom = innerseps['bottom'],
                          right  = innerseps['right'],
                          top    = innerseps['top'],
                          wspace = innerseps['wspace'],
                          hspace = innerseps['hspace'])

      # draws the guiding elements
      gpc, glc = drawGuides(tomains, ulimits, xoff, yoff, fontgon)
      plt.gca().add_collection(gpc)
      plt.gca().add_collection(glc)

      # adds the treatment ID name as column title
      plt.text(xoffcol, yoffcol, treatment_id,
               transform=plt.gca().transAxes,
               fontsize= 1.5 * fontgon['size'],
               horizontalalignment='center')

      # disables the cartesian axes and rescales the plotting, for visual clarity
      plt.gca().autoscale()
      plt.gca().axis('off')

  # plots the offer representation at each optimisation step
  with writer.saving(fig, os.path.join(*targetpath, filename + '.mp4'), dpi):

    last_precision = 0.0
    for (chromosome, convergence) in data:
      it += 1
      treatment_s = chromosome2scores(chromosome, treatment_ids, tomains, ulimits)

      col = []
      for (treatment_id, pos) in row_script:
        plt.subplot(nrows, ncols, pos)

        if(treatment_id == '.'):

          offers = {}
          for treatment in treatment_s:
            offers[treatment] = coords2poly(scores2coords(treatment_s[treatment], tomains, ulimits))

          # computes the performance of the obtained offer representations
          hits, tries, _ = estimatePrecisionPoly(splits['Train'], treatment_h, demands, offers)
          precision = hits/tries

          # updates the progress chart with convergence and precision

          # a blue dot represents population convergence
          # (i.d., the fractional value of the population convergence)
          plt.gca().scatter(it, convergence, marker='.', s=2, color='blue')
          if(last_precision < precision):
            # a red dot represents improvement in precision relative to the last iteration
            plt.gca().scatter(it, precision,   marker='+', s=4, color='red')
            last_precision = precision
          else:
            # a yellow dot represents no improvement in precision relative to the last iteration
            plt.gca().scatter(it, precision,   marker='.', s=2, color='yellow')

        else:
          opc = drawDiagram(treatment_s[treatment_id], tomains, ulimits, ECO_PTRN_OFFER)
          col.append(plt.gca().add_collection(opc))

      writer.grab_frame()
      #for e in col: e.remove()
      for e in col: e.set_edgecolor('lightgray')

  return None


#--------------------------------------------------------------------------------------------------
# Problem-specific definitions: interface for instruments (scales, inventories, batteries)
#--------------------------------------------------------------------------------------------------

class Instrument:
  """
  Instrument model
  name (str) .....: instrument identifier
  tomains (tuple) : order in which domains of a particular instrument instance are assigned to
                    polygon vertices
  limits (dict) ..: the upper and lower bounds for values of each domain of the instrument
  """

  def __init__(self, name, tomains, limits):

    self.name = name
    self._tomains = tomains
    self._limits  = limits

  def tomains(self):
    return self._tomains

  def ulimits(self):

    res = {}
    for domain in self._tomains:
      res[domain] = self._limits[domain][1]

    return res

  def llimits(self):

    res = {}
    for domain in self._tomains:
      res[domain] = self._limits[domain][0]

    return res

  def clip(self, scores):
    """
    ensures the results obtained with an instrument are within specified limits
    """
    new_scores = {}
    for domain in self._limits:
      (lb, ub) = self._limits[domain]
      if(  scores[domain] < lb):
        new_scores[domain] = lb
      elif(scores[domain] > ub):
        new_scores[domain] = ub
      else:
        new_scores[domain] = scores[domain]

    return new_scores

#--------------------------------------------------------------------------------------------------
# Problem-specific definitions: comparative evaluation of the Polygon method with other methods
#--------------------------------------------------------------------------------------------------

def estimatePrecisionPoly(case_ids, treatment_h, demands, offers, details = {}):

  # computes the number of hits achieved by the solution
  tries = 0
  hits  = 0
  for case_id in case_ids:
    demand = demands[case_id]

    expert_advice = treatment_h[case_id]['Treatments']
    nt = len(expert_advice) # number of treatments associated with the case in hand

    # applies the model to obtain a ranking of available treatments
    L = []
    for treatment_id in offers:
      offer   = offers[treatment_id]
      alpha   = matchp(offer, demand)
      L.append((treatment_id, alpha))
    L = sorted([(treatment_id, alpha) for (treatment_id, alpha) in L], key = lambda e: -e[1])
    system_advice = [treatment_id for (treatment_id, alpha) in L if alpha != 0.0]

    # computes the precision of the model in the given test sample
    # -- this is precision@N, with N = len(system_advice)
    acc = precision(system_advice, expert_advice)
    tries += nt
    hits  += acc * nt

    details[case_id] = (acc, expert_advice, system_advice, L)

  return hits, tries, details

def estimatePrecisionRandom(case_ids, treatment_h, assignments, details = {}):

  # computes the number of hits achieved by the solution
  tries = 0
  hits  = 0
  for case_id in case_ids:

    expert_advice = treatment_h[case_id]['Treatments']
    nt = len(expert_advice) # number of treatments associated with the case in hand

    # applies the model to obtain a ranking of available treatments
    interventions = list(assignments.keys())
    L = sorted(zip(interventions, np.random.random(len(interventions))), key = lambda e: -e[1])
    system_advice = [treatment for (treatment, _) in L]

    # computes the precision of the model in the given test sample
    # -- this is precision@N, with N = len(system_advice)
    acc = precision(system_advice, expert_advice)
    tries += nt
    hits  += acc * nt

    details[case_id] = (acc/nt, expert_advice, system_advice, L)

  return hits, tries, details

def estimatePrecisionKnn(case_ids, known_case_ids, case_s, treatment_h, tomains, assignments, metric, param_metric, k, details = {}):

  # computes the number of hits achieved by the solution
  tries = 0
  hits  = 0
  for case_id in case_ids:

    expert_advice = treatment_h[case_id]['Treatments']
    nt = len(expert_advice) # number of treatments associated with the case in hand

    # applies the model to obtain a ranking of available treatments
    reverseAssignments = {case_id: treatment for treatment in assignments
                                             for case_id in assignments[treatment]}

    counter    = defaultdict(int)
    v = np.array([case_s[case_id][domain] for domain in tomains])
    neighbours = []
    for known_case_id in known_case_ids:
      w = np.array([case_s[known_case_id][domain] for domain in tomains])
      neighbours.append((known_case_id, fmetric(metric)(v, w, param_metric)))
    neighbours.sort(key = lambda e: e[1])

    for (neighbour, _) in neighbours[0:k]:
      counter[reverseAssignments[neighbour]] += 1
    system_advice = [treatment for (treatment, _) in sorted(counter.items(), key = lambda e: -e[1])]

    # computes the precision of the model in the given test sample
    # -- this is precision@N, with N = len(system_advice)
    acc = precision(system_advice, expert_advice)
    tries += nt
    hits  += acc * nt

    details[case_id] = (acc/nt, expert_advice, system_advice, neighbours)

  return hits, tries, details

def estimatePrecisionSVD(case_ids, case_s, treatment_h, tomains, model, details = {}):

  # unpacks parameters
  (P, S, V, interventions) = model

  # computes the number of hits achieved by the solution
  tries = 0
  hits  = 0
  for case_id in case_ids:

    expert_advice = treatment_h[case_id]['Treatments']
    nt = len(expert_advice) # number of treatments associated with the case in hand

    # applies the model to obtain a ranking of available treatments
    Xnew = np.array([case_s[case_id][domain] for domain in tomains]).reshape(1, len(tomains))
    (_, Yn) = plsPredict(Xnew, P, S, V, interventions)
    system_advice = [treatment for (treatment, _) in Yn[0]]

    # computes the precision of the model in the given test sample
    # -- this is precision@N, with N = len(system_advice)
    acc = precision(system_advice, expert_advice)
    tries += nt
    hits  += acc * nt

    details[case_id] = (acc/nt, expert_advice, system_advice, Yn)

  return hits, tries, details

def estimatePrecisionLIR(case_ids, case_s, treatment_h, tomains, model, details = {}):

  # unpacks parameters
  (W, interventions) = model

  # computes the number of hits achieved by the solution
  tries = 0
  hits  = 0
  for case_id in case_ids:

    expert_advice = treatment_h[case_id]['Treatments']
    nt = len(expert_advice) # number of treatments associated with the case in hand

    # applies the model to obtain a ranking of available treatments
    Xnew = np.array([case_s[case_id][domain] for domain in tomains]).reshape(1, len(tomains))
    (_, Yn) = LIRPredict(Xnew, W, interventions)
    system_advice = [treatment for (treatment, _) in Yn[0]]

    # computes the precision of the model in the given test sample
    # -- this is precision@N, with N = len(system_advice)
    acc = precision(system_advice, expert_advice)
    tries += nt
    hits  += acc * nt

    details[case_id] = (acc/nt, expert_advice, system_advice, Yn)

  return hits, tries, details

def estimatePrecisionMLP(case_ids, case_s, treatment_h, tomains, model, details = {}):

  # unpacks parameters
  (mlp, mu, sd, interventions) = model

  # computes the number of hits achieved by the solution
  tries = 0
  hits  = 0
  for case_id in case_ids:

    expert_advice = treatment_h[case_id]['Treatments']
    nt = len(expert_advice) # number of treatments associated with the case in hand

    # applies the model to obtain a ranking of available treatments
    Xnew = np.array([case_s[case_id][domain] for domain in tomains]).reshape(1, len(tomains))
    Xnew = (Xnew - mu) / sd
    (_, Yn) = MLPPredict(Xnew, mlp, interventions)
    system_advice = [treatment for (treatment, _) in Yn[0]]

    # computes the precision of the model in the given test sample
    # -- this is precision@N, with N = len(system_advice)
    acc = precision(system_advice, expert_advice)
    tries += nt
    hits  += acc * nt

    details[case_id] = (acc/nt, expert_advice, system_advice, Yn)

  return hits, tries, details

def filterBestResults(results):

  bestResults = defaultdict(lambda: defaultdict(list))
  for entry in results:
    (essay, cutoff, numOfGroups, model, params) = entry

    if(len(bestResults[essay][model]) > 0):

      refEntry = bestResults[essay][model][0] # all references have the same mean precision
      refEstimate = results[refEntry][2]      # estimated mean precision of the reference
      newEstimate = results[entry][2]         # estimated mean precision of the comparandum

      if(newEstimate > refEstimate):
        # updates the reference to reflect another performance group
        bestResults[essay][model] = [entry]

      elif(newEstimate == refEstimate):
        # updates the reference to add another member to the performance group
        bestResults[essay][model].append(entry)

    else:
      # updates the reference to add the first member of the performance group
      bestResults[essay][model] = [entry]

  return bestResults

def precision(judgements, truth, N = 0):

  if(N <= 0): N = len(judgements) # corresponds to N -> +inf

  acc = 0.0
  n = len(truth)
  for i in range(n):
    try:
      j = judgements[0:N].index(truth[i])
      acc += 1 / (abs(i - j) + 1)
    except ValueError:
      None

  return acc/n

def solveineqs(inequalities): # future use in SVM
  """
  solves a system of inequalities
  'inequalities' is a list of fundamental inequalities in descending order, e.g.,
     [['C', 'A'], ['C', 'B'], ['A', 'B']] is interpreted as a system with C > A, C > B, A > B

  returns a list representing the ordering of the variables, e.g.,
     ['C', 'A', 'B'], meaning C > A > B

  if no ordering of the variables satisfies the inequalities, None is returned

  """
  # recovers the list of variables
  vars = set(chain(*inequalities))

  # tests if all inequalities hold together
  L = list(permutations(vars))
  for (var1, var2) in inequalities: # assumes (var1 > var2)
    i = len(L) - 1
    while(i >= 0):
      # searches for the positions of the variables in the current inequality
      posvar1 = L[i].index(var1)
      posvar2 = L[i].index(var2)
      # removes the permutations in which 'var1' is succeeded by 'var2' in a 'x > y > z' system
      if(posvar1 > posvar2):
        L.pop(i)
      i -= 1

  res = list(L[0]) if len(L) == 1 else None
  return res

def applyHC(mm, case_ids, cutoff, linkage, sample_redro):

  # applies hierarchical clustering using the external metric matrix mm
  if(linkage == 'ward'):
    model = AgglomerativeClustering(distance_threshold = cutoff,
                                    compute_full_tree  = True,
                                    n_clusters = None,
                                    linkage    = linkage)
  else:
    model = AgglomerativeClustering(distance_threshold = cutoff,
                                    compute_full_tree  = True,
                                    n_clusters = None,
                                    linkage    = linkage,
                                    affinity   = 'precomputed')
  model = model.fit(mm)
  res = copy(model.labels_)
  leaves = createDendrogram(model, truncate_mode = None, distance_sort = 'ascending')
  case_ids_ao = [sample_redro[i] for i in leaves]

  return case_ids_ao, res

def plsSVD(case_ids, case_s, treatment_h, tomains):

  # build matrix X from case scores (numerical variables), and matrix Y from treatment history (nominal variables)
  X  = np.array([ [case_s[case_id][domain] for domain in tomains] for case_id in case_ids])
  Yn = [treatment_h[case_id]['Treatments'] for case_id in case_ids]

  # factorises the matrix A=X|Yn and computes the X->U transformation
  A, _, interventions = applyCoding(X, Yn)
  (U,S,V, _) = plsFactorise(A)
  P = np.linalg.inv(X.transpose() @ X) @ X.transpose() @ U

  return (P, S, V, interventions)

def LIR(case_ids, case_s, treatment_h, tomains):

  # build matrix X from case scores (numerical variables), and matrix Y from treatment history (nominal variables)
  X  = np.array([ [case_s[case_id][domain] for domain in tomains] for case_id in case_ids])
  Yn = [treatment_h[case_id]['Treatments'] for case_id in case_ids]
  _, Y, interventions = applyCoding(X, Yn)

  # solves (a probably overdetermined) system of linear equations by means of least squares approximation
  W = np.linalg.inv(X.transpose() @ X) @ X.transpose() @ Y

  return (W, interventions)

def MLP(case_ids, case_s, treatment_h, tomains, layersize):

  # build matrix X from case scores (numerical variables), and matrix Y from treatment history (nominal variables)
  X  = np.array([ [case_s[case_id][domain] for domain in tomains] for case_id in case_ids])
  mu = np.mean(X, 0)
  sd = np.std(X, 0, ddof=1)
  X  = (X - mu) / sd

  Yn = [treatment_h[case_id]['Treatments'] for case_id in case_ids]
  _, Y, interventions = applyCoding(X, Yn)

  # induces a input/output mapping from the training sample
  ni = len(interventions)
  model = MLPRegressor(hidden_layer_sizes = (layersize,),
                       activation         = 'relu', # 'logistic', #
                       learning_rate_init = 0.001,
                       learning_rate      = 'constant',
                       solver             = 'sgd',
                       alpha              = 0.0001,
                       batch_size         = 1,
                       shuffle            = True,
                       max_iter           = 1000,
                       tol                = 1E-4,
                       n_iter_no_change   = 30,
                       random_state       = ECO_SEED
                      ).fit(X, Y)

  return (model, mu, sd, interventions)

def applyCoding(X, Ynominal):

  # transforming nominal variables (Ynominal) to numerical variables (Y) using dummy coding
  # https://stats.idre.ucla.edu/spss/faq/coding-systems-for-categorical-variables-in-regression-analysis-2/#DUMMYCODING
  interventions = sorted(set([item for sublist in Ynominal for item in sublist]))
  ni = len(interventions)
  nominal2numerical = {}
  for i in range(ni):
    intervention = interventions[i]
    nominal2numerical[intervention] = np.array([1 if j==i else 0 for j in range(ni)])

  Y = np.vstack([sum([nominal2numerical[intervention] for intervention in sublist]) for sublist in Ynominal])
  A = np.hstack((X,Y))

  return A, Y, interventions

def plsFactorise(A):

  # training the model
  (U,S,Vt) = np.linalg.svd(A)
  singularValues = S

  dimColNullspace = Vt.shape[0] - S.shape[0]
  if(dimColNullspace > 0):
    aux = np.zeros((S.shape[0], dimColNullspace))
    S = np.hstack((np.diag(S), aux))
  else:
    S = np.diag(S)

  dimRowNullspace = U.shape[1] - S.shape[0]
  if(dimRowNullspace > 0):
    aux = np.zeros((dimRowNullspace, S.shape[1]))
    S = np.vstack((S, aux))
  V = Vt.transpose()

  return (U,S,V, singularValues)

def plsPredict(Xnew, P, S, V, interventions):

  Anew = Xnew @ P @ S @ V.transpose()
  Ynew = Anew[:, P.shape[0]:]

  ni = len(interventions)
  Ynominalnew = []
  for i in range(Ynew.shape[0]):
    aux = [(interventions[j], Ynew[i,j]) for j in range(ni)]
    Ynominalnew.append(sorted(aux, key = lambda e: -e[1]))

  return Anew, Ynominalnew

def LIRPredict(Xnew, W, interventions):

  Ynew = Xnew @ W

  ni = len(interventions)
  Ynominalnew = []
  for i in range(Ynew.shape[0]):
    aux = [(interventions[j], Ynew[i,j]) for j in range(ni)]
    Ynominalnew.append(sorted(aux, key = lambda e: -e[1]))

  return Ynew, Ynominalnew

def MLPPredict(Xnew, mlp, interventions):

  Ynew = mlp.predict(Xnew)

  ni = len(interventions)
  Ynominalnew = []
  for i in range(Ynew.shape[0]):
    aux = [(interventions[j], Ynew[i,j]) for j in range(ni)]
    Ynominalnew.append(sorted(aux, key = lambda e: -e[1]))

  return Ynew, Ynominalnew

def plotBestResults(bestResults, results, pivot, targetpath, filename):

  # initialises subplots
  listOfEssays = sorted(set([essay for (essay, _, _, _, _) in results]))#cand
  listOfModels = list(set([model for (_, _, _, model, _) in results]))

  ncolsmax = 5
  nrows = (len(listOfEssays) - 1)// ncolsmax + 1
  ncols = min(ncolsmax, len(listOfEssays))
  figsizew, figsizeh = 3.5 * ncols, 3.0 * nrows
  fig, axes = plt.subplots(nrows, ncols, figsize=(figsizew, figsizeh), sharey='all')

  # sets up the left-right order in which each model appears in a subplot
  orderOfModels = [model for model in ['kNN', 'MLP', 'Poly', 'SVD', 'Rnd'] if model in listOfModels]

  pos = 0                                 # position of the current plot in the grid
  for essay in listOfEssays:              # the results of each essay are shown in a separate plot
    pos += 1
    print()
    print('Essay {0}'.format(essay))

    (x, x_label, x_colour) = ([], [], []) # x-coordinate (order), label, and colour for a model
    (y, y_lb, y_ub) = ([], [], [])        # model mean precision estimate,
                                          # lower and upper bounds of confidence interval

    # recovers the confidence interval for the pivot model
    entry = bestResults[essay][pivot][0]
    (pivot_lb, pivot_ub, _, _) = results[entry]

    for i in range(len(orderOfModels)):   # each plot shows the results of all models, in the specified order
      model = orderOfModels[i]

      # recovers the data to be illustrated in the current plot
      entry = bestResults[essay][model][0]
      (essay, cutoff, numOfGroups, model, params) = entry
      (ci_lb, ci_ub, ci_mu, _) = results[entry]
      bar_colour = 'lightsteelblue' if overlapCI(pivot_lb, pivot_ub, ci_lb, ci_ub) else 'silver'

      x.append(i+1)                       # horizontal position of the model's error bar
      x_label.append(model)               # the label that identifies the model
      x_colour.append(bar_colour)         # the color of the bar

      y.append(ci_mu)                     # the mean precision estimate shown in the error bar
      y_lb.append(ci_mu - ci_lb)          # the lower bound estimate for the mean precision (only the delta)
      y_ub.append(ci_ub - ci_mu)          # the upper bound estimate for the mean precision (only the delta)

      print('Model {0: <6}, mu {1} lb {2} ub {3}'.format(x_label[i], y[i], y[i] - y_lb[i], y[i] + y_ub[i]))

    # draws the subplot
    lw, fst, fsa, tsp = 1, 15, 12, .30   # sets up sizes of some graphical elements
    plt.subplot(nrows, ncols, pos)
    #plt.grid(True, axis='y', color='silver', linestyle='solid', linewidth=lw/5)
    #plt.gca().set_axisbelow(True)
    plt.gca().patch.set_facecolor('0.95')
    plt.title('{0} groups (cutoff {1})'.format(numOfGroups, cutoff), fontsize=fst)
    plt.xticks(fontsize=fsa)
    plt.yticks(fontsize=fsa)
    plt.gca().set_xlim(0.0, len(orderOfModels) + 1)
    plt.gca().set_ylim(0.0, 1.05) #xxx replace lb for estimate for lowest(random/lb - 0.1) @step:8
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator(4))
    plt.subplots_adjust(left=.06, bottom=.06, right=.94, top=.94, wspace=.30, hspace=.40)
    #plt.bar(x, y, width=0.7, color='silver')
    #plt.errorbar(x, y, yerr = [y_lb, y_ub], fmt='+', markersize=6, color=x_colour[i], ecolor='red', elinewidth=6, capsize=0)
    plt.bar(x, y, width=0.7, color=x_colour)
    plt.errorbar(x, y, yerr = [y_lb, y_ub], fmt='+', markersize=6, color='red', ecolor='red', elinewidth=6, capsize=0)

    #if('Poly' in orderOfModels):
    #  i = orderOfModels.index('Poly')
    #  plt.axhline(y = y[i] - y_lb[i], color='green', linestyle='-.', lw=lw/4)
    #  plt.axhline(y = y[i] + y_ub[i], color='green', linestyle='-.', lw=lw/4)

    plt.xticks(x, x_label)#xxx not needed if we keep the bar

  plt.savefig(os.path.join(*targetpath, filename), bbox_inches = 'tight')
  plt.close(fig)

  return None

def overlapCI(ci1_lb, ci1_ub, ci2_lb, ci2_ub):
  # returns true if two confidence intervals overlap
  lb = max(ci1_lb, ci2_lb)
  ub = min(ci1_ub, ci2_ub)
  return((ub - lb) >= 0.0)

def evaluations2text(evaluations):

  header = '{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}'.format('essay',
                                                      'cutoff',
                                                      '#groups',
                                                      'run',
                                                      'model',
                                                      'precision',
                                                      'params')
  content = [header]

  for (essay, cutoff, numOfGroups, runID, model, precision, params) in evaluations:

    buffer = '{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}'.format(essay,
                                                        cutoff,
                                                        numOfGroups,
                                                        runID,
                                                        model,
                                                        precision,
                                                        params)
    content.append(buffer)

  return '\n'.join(content)

def results2text(results):

  header = '{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}'.format('essay',
                                                                'cutoff',
                                                                '#groups',
                                                                'model',
                                                                'params',
                                                                'ci precision lb',
                                                                'ci precision ub',
                                                                'ci precision mu',
                                                                'sample size')
  content = [header]

  for entry in results:
    (essay, cutoff, numOfGroups, model, params) = entry
    (ci_lb, ci_ub, ci_mu, ss) = results[entry]
    buffer = '{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}'.format(essay,
                                                                  cutoff,
                                                                  numOfGroups,
                                                                  model,
                                                                  params,
                                                                  ci_lb,
                                                                  ci_ub,
                                                                  ci_mu,
                                                                  ss)
    content.append(buffer)

  return '\n'.join(content)


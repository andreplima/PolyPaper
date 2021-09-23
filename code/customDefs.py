import pickle
import codecs

import numpy as np
import matplotlib
import matplotlib.pyplot    as plt
import matplotlib.animation as manimation

from datetime      import datetime
from shapely.geometry        import Polygon
from matplotlib.collections  import LineCollection, PolyCollection
from scipy.cluster.hierarchy import dendrogram
from scipy.optimize          import differential_evolution
from sklearn.cluster         import AgglomerativeClustering

ECO_SEED = 23
ECO_MAXCORES = 4
ECO_DATETIME_FMT = '%Y%m%d%H%M%S'

# constants identifying polygon drawing patterns
ECO_PTRN_DEMAND  = 0
ECO_PTRN_OFFER   = 1
ECO_PTRN_BENEFIT = 2

#--------------------------------------------------------------------------------------------------
# General purpose definitions - I/O interfaces used in logging and serialisation
#--------------------------------------------------------------------------------------------------

def stimestamp():
  return(datetime.now().strftime(ECO_DATETIME_FMT))

def tsprint(msg):
  print('[{0}] {1}'.format(stimestamp(), msg))

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

def saveAsText(content, filename, _encoding='utf-8'):
  f = codecs.open(filename, 'w', encoding=_encoding)
  f.write(content)
  f.close()

#--------------------------------------------------------------------------------------------------
# Problem-specific definitions: recovering application parameters
#--------------------------------------------------------------------------------------------------
def getDistParams():

  # access to clinical dataset employed in the article is subject to local regulations
  # instead, we use a generative description of the data, obtained as follows:

  # targetpath = ...
  # case_s = deserialise(join(*targetpath, 'case_s'))  # file that holds the actual case scores
  # L = [(case_s[case_id]['DOM1'], case_s[case_id]['DOM2'], case_s[case_id]['DOM3'], case_s[case_id]['DOM4']) for case_id in case_s]
  # data = np.array(L)
  # mu = data.mean(0)
  # Sigma = np.cov(data.T)

  # these are the obtained results
  mu    = np.array( [15.225     , 15.29907407, 14.88148148, 14.84814815])
  Sigma = np.array([[ 6.45852804,  3.74806075,  3.63252336,  3.51373832],
                    [ 3.74806075,  4.88626082,  3.53979578,  3.04275528],
                    [ 3.63252336,  3.53979578,  8.59124264,  3.13473174],
                    [ 3.51373832,  3.04275528,  3.13473174,  5.51822084]])

  return (mu, Sigma)

def getPlotParams1():
  unitsizes   = [3, 2.8]
  fontgon     = {'family': 'arial', 'color':  'black', 'weight': 'normal', 'size': 10}
  innerseps   = {'left': 0.06, 'bottom': 0.06, 'right': 0.94, 'top': 0.96, 'wspace': 0.52, 'hspace': 0.30}
  xoffset     = [ 0.02, -0.20, -0.48, -0.20]
  yoffset     = [-0.04,  0.04, -0.05, -0.12]
  titleoffset = [ 0.50,  1.15, -0.32,  0.51]
  tagoffset   = [ 0.78,  0.78, -45]

  return (unitsizes, fontgon, innerseps, xoffset, yoffset, titleoffset, tagoffset)

#--------------------------------------------------------------------------------------------------
# Problem-specific definitions: patient stratification
#--------------------------------------------------------------------------------------------------
def plotDendrogram(X, case_ids, cutoff, sample_redro):

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
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    # plots the corresponding dendrogram
    res = dendrogram(linkage_matrix, truncate_mode = None,
                                     distance_sort = 'ascending',
                                     orientation = 'top')
    return res['leaves']

  # unpacks parameters
  (unitsizes, fontgon, innerseps, xoff, yoff, titleoffset, tagoffset) = getPlotParams1()
  (unitsizew, unitsizeh) = unitsizes
  (left, bottom, right, top, wspace, hspace) = innerseps
  (xoffcol, yoffcol, xoffrow, yoffrow) = titleoffset

  # applies hierarchical clustering using the external metric matrix mm
  model = AgglomerativeClustering(distance_threshold = cutoff,
                                  n_clusters = None,
                                  linkage    = 'ward')
  model = model.fit(X)
  res = copy(model.labels_)

  # plots a dendrogram based on the clustering results and obtains the order
  # in which cases should be plotted so as to improve the sense of similarity (affinity order) in the reader
  fig = plt.figure(figsize = (6 * unitsizew, 2 * unitsizeh))
  plt.title(plotTitle)
  leaves = createDendrogram(model, truncate_mode = None, distance_sort = 'ascending')
  case_ids_ao = [sample_redro[i] for i in leaves]
  plt.xlabel("Number of points in node (or index of point if no parenthesis).")

  plt.savefig('dendrogram', bbox_inches = 'tight')
  plt.close(fig)

  return case_ids_ao, res

#--------------------------------------------------------------------------------------------------
# Problem-specific definitions: polygonal representation, transformations, and operations
# (e.g., Projection and Match operators described in the article)
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

def coords2poly(coords):
  return Polygon(coords)

def projectOp(scores, tomains, ulimits):
  coords = scores2coords(scores, tomains, ulimits)
  poly   = coords2poly(coords)
  return poly

def matchOp(offer, demand):
  # estimates the expected benefit of treatment
  # area and intersection transformations being handled by the shapely package
  expected = offer.difference(demand)
  return expected.area

#--------------------------------------------------------------------------------------------------
# Problem-specific definitions: learning intervention (i.e., treatment) representation
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

def callbackfn(chromosome, convergence):

  GlbBuffer.append((chromosome, convergence))
  return False

def learn(case_ids, case_s, ulimits, llimits, tomains, treatment_h, params, workers, overridePopsize=0):

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

def objfn(chromosome, *args):
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

def details2text(details, test, sample_order):
  #xxx add column partition
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

def retraceOfferGrid(history, splits, dsplits, demands, tomains, ulimits, treatment_h, params, plotTitle, filename):

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
# Problem-specific definitions: evaluation of learned representations
#--------------------------------------------------------------------------------------------------
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

def evaluate(case_ids, treatment_h, demands, offers, details = {}):

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


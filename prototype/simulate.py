"""
  This script implements the learning algorithm described in:

  Andre Lima, Laurentino Dantas, Brunela Orlandi, Paula Castro, Maria Pimentel,
  and Marcelo Manzato. 2021. An InterpretableRecommendation Model for Gerontological Care.
  In Fifteenth ACM Conference on Recommender Systems (RecSys '21), 2021, Amsterdam, Netherlands.
  https://doi.org/10.1145/3460231.34788501

  Our objective in sharing this code is to offer a didactic implementation of the operators
  described in the article. Given that access to the clinical data employed in the study is
  not publicly available, this prototype does not support full reproduction.

  Note: in sake of simplicity, in the comments below, you may interpret the term:
        'case'         as being synonymous with 'patient', and
        'intervention' as being synonymous with 'treatment'.

"""
import os
import sys
import numpy  as np
import pandas as pd

from customDefs import ECO_SEED, tsprint, saveAsText, serialise
from customDefs import projectionOp, learn, evaluate
from customDefs import getDistParams, plotDendrogram, details2text
from sklearn.model_selection import train_test_split

# defines the domains assessed by the WHOQOL-BREF instrument, as well as their boundaries
ECO_DOMAIN_LB =  4.0
ECO_DOMAIN_UB = 20.0
ECO_DOMAINS = {'DOM1': (ECO_DOMAIN_LB, ECO_DOMAIN_UB),
               'DOM2': (ECO_DOMAIN_LB, ECO_DOMAIN_UB),
               'DOM3': (ECO_DOMAIN_LB, ECO_DOMAIN_UB),
               'DOM4': (ECO_DOMAIN_LB, ECO_DOMAIN_UB)}

def main(ss, cutoff):

  print()
  tsprint('Job started')

  tsprint('Creating a synthetic dataset')

  # synthetises a sample of patient assessments obtained with the WHOQOL-BREF instrument
  # -- each assessment has 4 variables, DOM1 to DOM4 (see Section 3.1 in the paper)
  tsprint('-- drawing sample with {0} clinical cases.'.format(ss))
  np.random.seed(ECO_SEED)
  (mu, Sigma) = getDistParams()
  X = np.random.multivariate_normal(mu, Sigma, ss)
  X[X < ECO_DOMAIN_LB] = ECO_DOMAIN_LB
  X[X > ECO_DOMAIN_UB] = ECO_DOMAIN_UB

  # assigns each case to a single intervention
  # (in Section 4, see the paragraph describing the Dataset; also Figure 3)
  tsprint('-- clustering cases with hierarchical method, ward linkage, cutoff at {0:}.'.format(cutoff))
  case_ids = ['P{0}'.format(i+1)  for i in range(ss)]
  sample_order = {case_ids[i] : i for i in range(ss)}
  sample_redro = {i : case_ids[i] for i in range(ss)}
  filename  = 'dendrogram'
  plotTitle = 'Dendrogram showing similarity among patient assessments (euclidean distance, ward linkage)'
  case_ids_ao, labels = plotDendrogram(X, case_ids, cutoff, sample_redro, plotTitle, filename)

  # creates a synthetic clinical dataset
  # (in Section 4, see the paragraph describing the Dataset)
  (dom1, dom2, dom3, dom4) = zip(*X)
  interventions = [chr(ord('A') + label) for label in labels]
  data = pd.DataFrame(list(zip(dom1, dom2, dom3, dom4, interventions)), index = case_ids,
                      columns=['DOM1', 'DOM2', 'DOM3', 'DOM4', 'intervention'])
  tsprint('   dataset with {0} cases created.'.format(ss))
  tsprint('   cases were assigned to either one of {0} distinct interventions'.format(len(set(labels))))

  print()
  print(data.head())
  print()

  # splits the dataset into training and test partitions
  # -- Note: this does not correspond to the splitting used in the article, in which
  #          stratified sampling (per intervention) with an 8:2 rule was employed
  train, test = train_test_split(data, test_size=0.33)

  # learns representations for interventions
  tsprint('Learning representations for interventions using the dataset')
  tomains = tuple(ECO_DOMAINS)
  llimits = {domain: ECO_DOMAINS[domain][0] for domain in tomains}
  ulimits = {domain: ECO_DOMAINS[domain][1] for domain in tomains}
  (treatment_s, offers, history) = learn(train, tomains, llimits, ulimits)

  # creates data structures to support the evaluation of the learned representations
  case_s   = {}
  demands  = {}
  treatment_h = {}
  for case_id, row in data.iterrows():
    scores = row[['DOM1', 'DOM2', 'DOM3', 'DOM4']]
    case_s[case_id]  = scores
    demands[case_id] = projectionOp(scores, tomains, ulimits)
    treatment_h[case_id] = [row['intervention']] # each case is assigned to a single intervention,
                                                 # but the learning process can handle lists with
                                                 # multiple interventions

  # evaluates the learned representations
  tsprint('Evaluating learned representations')
  details  = {}
  case_ids = train.index
  hits, tries, details = evaluate(case_ids, treatment_h, demands, offers, details)
  tsprint('   [Train] average precision of {2:5.3f} ({0:7.3f} hit(s) out of {1})'.format(hits, tries, hits/tries))

  case_ids = test.index
  hits, tries, details = evaluate(case_ids, treatment_h, demands, offers, details)
  tsprint('   [Test]  average precision of {2:5.3f} ({0:7.3f} hit(s) out of {1})'.format(hits, tries, hits/tries))

  # this saves a file containing the expected benefit of each treatment to all cases
  saveAsText(details2text(details, sample_order, case_ids), 'performance.csv')

  # serialises data required to plot the patient vs intervention diagram, and the animated grid
  tsprint('Saving data')
  dataset = (data, train, test)
  serialise(dataset, 'dataset')
  plotGridParams = (case_ids_ao, case_s, treatment_s, details, tomains, ulimits, sample_order, history)
  serialise(plotGridParams, 'plotGridParams')

  print()
  tsprint('Job completed.')

if __name__ == "__main__":

  try:

    ss = int(sys.argv[1])          # sample size (sample contains ss cases,
                                   # (each case corresponds to a unique patient assessment)

    cutoff = float(sys.argv[2])    # the cutoff level (see Section 4, paragraph on Dataset)

  except (ValueError, IndexError):

    (ss, cutoff) = (100, 14)

  main(ss, cutoff)

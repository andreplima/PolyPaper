import os
import sys

from customDefs import tsprint, deserialise
from customDefs import plotOfferGrid

def main():

  print()
  tsprint('Job started')

  # recovers serialised data
  tsprint('-- deserialising data from previous simulation')
  (data, train, test) = deserialise('dataset')
  plotGridParams = deserialise('plotGridParams')
  (case_ids_ao, case_s, treatment_s, details, tomains, ulimits, sample_order, history) = plotGridParams

  # plots a demand vs offer diagram with instances ordered as they appear in the dendrogram
  tsprint('-- plotting a patient vs intervention diagram, in affinity order')

  case_ids  = [case_id for case_id in case_ids_ao if case_id in train.index]
  filename  = 'patient_treatment_train'
  plotTitle = 'Expected benefit of intervention for each patient, in affinity order.'
  plotOfferGrid(case_ids, case_s, treatment_s, details, tomains, ulimits, sample_order, plotTitle, filename)

  case_ids  = [case_id for case_id in case_ids_ao if case_id in test.index]
  filename  = 'patient_treatment_test'
  plotTitle = 'Expected benefit of intervention for each patient, in affinity order.'
  plotOfferGrid(case_ids, case_s, treatment_s, details, tomains, ulimits, sample_order, plotTitle, filename)

  print()
  tsprint('Job completed.')

if __name__ == "__main__":

  main()

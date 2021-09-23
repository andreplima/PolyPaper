import os
import sys

from customDefs import tsprint, deserialise
from customDefs import plotOfferGrid

def main():

  print()
  tsprint('Job started')

  # recovers serialised data
  tsprint('-- deserialising data from previous simulation')
  plotGridParams = deserialise('plotGridParams')
  (case_ids_ao, case_s, treatment_s, details, tomains, ulimits, sample_order, history) = plotGridParams

  # plots a demand vs offer diagram with instances ordered as they appear in the dendrogram
  tsprint('-- plotting a patient vs intervention diagram, in affinity order')
  filename = 'patient_treatment_grid'
  plotTitle = 'Expected benefit of intervention for each patient, in affinity order.'
  plotOfferGrid(case_ids_ao, case_s, treatment_s, details, tomains, ulimits, sample_order, plotTitle, filename)

  print()
  tsprint('Job completed.')

if __name__ == "__main__":

  main()

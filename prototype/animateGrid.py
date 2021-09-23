import os
import sys

from customDefs import tsprint, deserialise
from customDefs import animateOfferGrid

def main():

  print()
  tsprint('Job started')

  # recovers serialised data
  tsprint('-- deserialising data from previous simulation')
  (data, train, test) = deserialise('dataset')
  plotGridParams = deserialise('plotGridParams')
  (case_ids_ao, case_s, treatment_s, details, tomains, ulimits, sample_order, history) = plotGridParams

  # creates an animation that retraces the process of representation learning
  tsprint('-- creating an animation to retrace the learning process')
  filename    = 'retrace_learning'
  plotTitle   = "Differential evolution applied to offer optimisation "
  animateOfferGrid(history, train, tomains, ulimits, plotTitle, filename)

  print()
  tsprint('Job completed.')

if __name__ == "__main__":

  main()

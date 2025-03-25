#libraries
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
import pandas as pd
from scipy.special import gammaln

#List of functions
def get_min_max_poisson_sample(xmean, samplesize):
  """
  Generates a sample of a Poisson distribution. and spits the minimum value for this sample
  """
  sample = np.random.poisson(lam=xmean, size=samplesize)
  return np.min(sample), np.max(sample)

def get_expected_min_max_poisson_sample(xmean, samplesize, iterations):
  """
  Generates the expected minimum and standard error of iterations samples of Poisson distribution of size samplesize.
  """
  array_minx = []
  array_maxx = []
  for i in range(iterations):
    resultmin, resultmax = get_min_max_poisson_sample(xmean, samplesize)
    array_minx.append(resultmin)
    array_maxx.append(resultmax)

  mean_array_minx = np.mean(array_minx)
  std_dev_array_minx = np.std(array_minx)
  std_error_array_minx = std_dev_array_minx / np.sqrt(iterations)

  mean_array_maxx = np.mean(array_maxx)
  std_dev_array_maxx = np.std(array_maxx)
  std_error_array_maxx = std_dev_array_maxx / np.sqrt(iterations)

  return mean_array_minx, std_error_array_minx, mean_array_maxx, std_error_array_maxx

def get_expected_overA(xmean, xmin, xmax, E_selectioncoeff, E_epsilon):
  result = 0
  if E_epsilon != 0:
    for k in range(int(xmin), int(xmax+1)):
      floatk = float(k)
      probability_k = np.exp(floatk*np.log(xmean)-xmean - gammaln(k + 1))
      result += ((1.0+(E_epsilon/E_selectioncoeff))**floatk)*probability_k
    return result
  else:
    for k in range(int(xmin), int(xmax+1)):
      floatk = float(k)
      probability_k = np.exp(floatk*np.log(xmean)-xmean- gammaln(k + 1))
      result += floatk*probability_k
    return result

def get_growht_rate(segregating_mutations, meansegregating, minsegregating, maxsegregating, E_selectioncoeff, E_epsilon):
  expected_term = get_expected_overA(meansegregating, minsegregating, maxsegregating, E_selectioncoeff, E_epsilon)
  if E_epsilon != 0:
    genotype_term = (1.0+(E_epsilon/E_selectioncoeff))**(segregating_mutations)
    rbest = -((E_selectioncoeff**2)/E_epsilon)*(expected_term-genotype_term)/(expected_term)
  else:
    rbest = -E_selectioncoeff*(expected_term-segregating_mutations)
  return rbest

def get_selective_deaths(meansegregating, minsegregating, maxsegregating, E_selectioncoeff, E_epsilon):
  fitness_best = np.exp(get_growht_rate(minsegregating, meansegregating, minsegregating, maxsegregating, E_selectioncoeff, E_epsilon))
  fraction_selective_deaths = 1.0-(1.0/fitness_best)
  return fraction_selective_deaths

def get_fitness(growth_rate):
  fitness = np.exp(growth_rate)
  return fitness

vget_expected_min_max_poisson_sample = np.vectorize(get_expected_min_max_poisson_sample)
vget_selective_deaths = np.vectorize(get_selective_deaths)
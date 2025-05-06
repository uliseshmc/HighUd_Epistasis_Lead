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

vget_expected_min_max_poisson_sample = np.vectorize(get_expected_min_max_poisson_sample)

def get_expected_overA(xmean, xmin, xmax, R_epistasis):
  result = 0
  if R_epistasis != 1:
    for k in range(int(xmin), int(xmax+1)):
      floatk = float(k)
      probability_k = np.exp(floatk*np.log(xmean)-xmean - gammaln(k + 1))
      result += (R_epistasis**floatk)*probability_k
    return result
  else:
    for k in range(int(xmin), int(xmax+1)):
      floatk = float(k)
      probability_k = np.exp(floatk*np.log(xmean)- xmean- gammaln(k + 1))
      result += floatk*probability_k
    return result
  
vget_expected_overA = np.vectorize(get_expected_overA)

def get_growht_rate(segregating_mutations, E_selectioncoeff, R_epistasis, expected_term):
  if R_epistasis != 1:
    genotype_term = R_epistasis**(segregating_mutations)
    rbest = -(E_selectioncoeff/expected_term)*(expected_term-genotype_term)/(R_epistasis-1)
  else:
    rbest = -E_selectioncoeff*(expected_term-segregating_mutations)
  return rbest

vget_growht_rate = np.vectorize(get_growht_rate)

def get_selective_deaths(meansegregating, minsegregating, maxsegregating, E_selectioncoeff, R_epistasis):
  expected_term = get_expected_overA(meansegregating, minsegregating, maxsegregating, R_epistasis)
  growthrate_best = get_growht_rate(minsegregating, E_selectioncoeff, R_epistasis, expected_term)
  fitness_best = get_fitness(growthrate_best)
  fraction_selective_deaths = 1.0-(1.0/fitness_best)
  return fraction_selective_deaths

vget_selective_deaths = np.vectorize(get_selective_deaths)

def get_fitness(growth_rate):
  fitness = np.exp(growth_rate)
  return fitness

def get_best_fitness(meansegregating, minsegregating, maxsegregating, E_selectioncoeff, R_epistasis):
  expected_term = get_expected_overA(meansegregating, minsegregating, maxsegregating, R_epistasis)
  growthrate_best = get_growht_rate(minsegregating, E_selectioncoeff, R_epistasis, expected_term)
  fitness_best = np.exp(growthrate_best)
  return fitness_best

vget_best_fitness = np.vectorize(get_best_fitness)


def get_fitness_variance(meansegregating, minsegregating, maxsegregating, E_selectioncoeff, R_epistasis):
  expected_term = get_expected_overA(meansegregating, minsegregating, maxsegregating, R_epistasis)
  expectationW = 0
  espectationW2 = 0
  for k in range(int(minsegregating), int(maxsegregating+1)):
    floatk = float(k)
    probability_k = np.exp(floatk*np.log(meansegregating) - meansegregating - gammaln(k + 1))
    growthrate = get_growht_rate(k, E_selectioncoeff, R_epistasis, expected_term) 
    fitness = np.exp(growthrate)
    expectationW += fitness*probability_k
    espectationW2 += (fitness**2)*probability_k
  variance = espectationW2 - expectationW**2
  return variance
import pymc as pm
import polars as pl
import numpy as np
import arviz as az

np.random.seed(42)

#set parameter values
beta0=3
beta1=7
sigma=3
n=100

#simulate data
x=np.random.uniform(0,7,size=n)
y=beta0 + beta1*x + np.random.normal(0, size=n)*sigma #random normal noise with variance of sigma

#create a model object
basic = pm.Model()

#specify model
with basic:
    #prior (could have 3 priors; one for every parameter we have)
    beta = pm.Normal('beta',mu=0,sigma=10,shape=2) #'beta' is named index; mu, sigma, etc are values of the prior; shape=2 because there are 2 betas
    #beta0 = pm.Normal('beta0',mu=0,sigma=10) # in case you want them written separately
    #beta1 = pm.Normal('beta1',mu=0,sigma=10)
    sigma = pm.HalfNormal('sigma',sigma=1) #strictly positive (truncated at 0)

    #Likelihood
    mu = beta[0] + beta[1]*x
    y_obs = pm.Normal('y_obs',mu=mu, sigma=sigma, observed=y) #"observed" data (observations); p(X|theta)
    
#create an InferenceData object
with basic:
    #draw 1000 posterior samples
    idata = pm.sample() #sophisticated sampler to save some sanity (SSSSS) #posterior samples

#have we recovered parameters?
az.summary(idata,round_to=2)

#visualize marginal posterios
az.plot_trace(idata,combined=True) #gives both


### Foxes data

'''
DAG
area -> avgfood -> groupsize
            |          |
            -> weight <-
outcome: weight
variable of interest: avgfood

all paths:
avgfood - weight
avgfood - groupsize - weight

no backdoor paths

close paths
pipe (open until closed): avgfood -> groupsize -> weight
include avgfood and groupsize
'''

#import (standardized) fox data
foxes = pl.read_csv('data/foxes.csv')

#separate predictors and outcome
X = foxes.select(pl.col(['avgfood','groupsize'])).to_numpy()
y = foxes.select(pl.col('weight')).to_numpy().flatten()

with pm.Model() as foxes_model:
    #Data
    X_data = pm.Data('X_data',X)
    y_data = pm.Data('y_data',y)

    #Priors
    alpha = pm.Normal('alpha', mu=0, sigma=0.2)
    beta = pm.Normal('beta', mu=0, sigma=0.5, shape=2)
    sigma = pm.Exponential('sigma',lam=1)

    #Likelihood
    mu=alpha + X_data @ beta
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed = y_data)

#Sample
with foxes_model:
    draws = pm.sample()

#visualize marginal posteriors
az.plot_forest(draws,var_names=['beta'], combined=True, hdi_prob=0.95)

#Sample from prior predictive distribution
with foxes_model:
    prior_draws = pm.sample_prior_predictive()

#Conduct prior predictive check
az.plot_dist(prior_draws.prior_predictive['y_obs'], label='prior predictive')
az.plot_dist(y, color='C1',label='observed')

#sample from posterior predictive distribution
with foxes_model:
    posterior_draws = pm.sample_posterior_predictive(draws)

#Conduct posterior predictive check
az.plot_dist(posterior_draws.posterior_predictive['y_obs'], label='posterior predictive')
az.plot_dist(y, color='C1',label='observed')

import pymc as pm
import polars as pl
import numpy as np
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt

############################################################
# Format data
############################################################

# Read in dataset
df = pd.read_excel(r'.\data\Project\healthdata.xlsx')

# Select features
X = df[["exercise_score", "age", "ses", "pec_severity"]].copy()
X["gender"] = df["gender"].map({"Female": 0, "Male": 1})

# Define outcome
y = df["mh_score"].values

# Convert features to numpy array
X_np = X.values

############################################################
# Specify model
############################################################
ex_model = pm.Model()

with ex_model:
    # Data
    X_data = pm.Data("X_data", X_np)
    y_data = pm.Data("y_data", y)

    # Priors
    alpha = pm.Normal("alpha", mu=0, sigma=1) # mental health score when all predictors = 0 is around 0, give/take 1
    beta = pm.Normal("beta", mu=0, sigma=1, shape=X_np.shape[1]) # exercise/age/etc could affect MH score, centered at 0, likely small
    sigma = pm.HalfNormal("sigma", sigma=1) # says standard deviation is positive, loosely assumes noise around 1

    # Likelihood
    mu = alpha + pm.math.dot(X_data, beta)
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_data)

with ex_model:
    # Sample from posterior
    draws = pm.sample(target_accept=0.95, random_seed=42)

############################################################
# Visualize and interpret
############################################################

# Summarize posterior
summary_df = az.summary(draws, var_names=["beta"], hdi_prob=0.95)
summary_df

# Visualize marginal posteriors
az.plot_forest(draws, var_names=["beta"], combined=True, hdi_prob=0.95, 
               figsize=(8, 5), 
               ridgeplot_overlap=0)
plt.axvline(0, color='gray', linestyle='--')
plt.title("Posterior Estimates with 95% Credible Intervals")
plt.yticks([0,1,2,3,4], ["Gender (Male)", "PEC Severity", "SES", "Age", "Exercise"])
plt.tight_layout()
plt.show()

'''
beta[0]	Effect of exercise_score
beta[1]	Effect of age
beta[2]	Effect of ses
beta[3]	Effect of pec_severity
beta[4]	Effect of gender (Male = 1)
'''

# Save calculated effects
summary_df["variable"] = [
    "exercise_score", 
    "age", 
    "ses", 
    "pec_severity", 
    "gender (Male)"
]

summary_df = summary_df[["variable"] + [col for col in summary_df.columns if col != "variable"]]
summary_df = summary_df[["variable", "mean", "hdi_2.5%", "hdi_97.5%"]]

summary_df.to_csv(".\data\Project\causal_model_summary.csv", index=False)

'''
	    variable    	mean	hdi_2.5%	hdi_97.5%
beta[0]	exercise_score	-0.357	-0.698  	-0.019
beta[1]	age  	 	 	0.015	-0.029	 	0.054
beta[2]	ses	 	 	 	0.038	-0.333	 	0.421
beta[3]	pec_severity	0.341	-0.092	 	0.755
beta[4]	gender (Male)	-0.494	-0.982	 	-0.044
'''

############################################################
# Prior/Posterior Predictive Checks
############################################################

# Prior predictive check
with ex_model:
    prior = pm.sample_prior_predictive()
plt.figure(figsize=(10, 6))
az.plot_dist(prior.prior_predictive["y_obs"], label="Prior Predictive")
az.plot_dist(y, color='C1', label="Observed")
plt.title("Prior Predictive Check")
plt.legend()
plt.show()

# Posterior predictive check
with ex_model:
    posterior = pm.sample_posterior_predictive(draws)
plt.figure(figsize=(10, 6))
az.plot_dist(posterior.posterior_predictive["y_obs"], label="Posterior Predictive")
az.plot_dist(y, color='C1', label="Observed")
plt.title("Posterior Predictive Check")
plt.legend()
plt.show()

############################################################
'''
```{python}
#| echo: false
import pandas as pd

estimates = pd.read_csv("../data/causal_model_summary.csv")
print(estimates)
```
'''
############################################################
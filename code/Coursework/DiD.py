import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import seaborn as sns

# Outcomes
def outcome(t, control_int, treat_int_delta, trend, delta, group, treated):
    return control_int + (treat_int_delta*group) + (t*trend) + (delta*treated*group)

#Y = beta0 + beta_delta*group + trend*time + delta*treated*group

def is_treated(t, intervention_time, group):
    return (t > intervention_time)*group

# true parameters
control_int = 1
treat_int_delta = 0.25
trend = 1
deltap = 0.5
intervention_time = 0.5

df = pd.DataFrame(
    {
        "group": [0,0,1,1]*10,
        "t": [0.0,1.0,0.0,1.0]*10,
        "unit": np.concatenate([[i]*2 for i in range(20)])
    }
)

df['treated'] = is_treated(df['t'], intervention_time, df['group'])

df['y'] = outcome(df['t'], control_int, treat_int_delta, trend, deltap, df['group'], df['treated'])

#introduce noise
df['y'] += np.random.normal(0,0.1, df.shape[0])
df

###########################################################################
#calculate point estimate of diff in diff (FREQUENTIST)
diff_control = (
    df.loc[(df["t"] == 1) & (df["group"] == 0)]["y"].mean()
    - df.loc[(df["t"] == 0) & (df["group"] == 0)]["y"].mean()
)
print(f"Pre/post difference in control group = {diff_control:.2f}")

diff_treat = (
    df.loc[(df["t"] == 1) & (df["group"] == 1)]["y"].mean()
    - df.loc[(df["t"] == 0) & (df["group"] == 1)]["y"].mean()
)

print(f"Pre/post difference in treatment group = {diff_treat:.2f}")

diff_in_diff = diff_treat - diff_control
print(f"Difference in differences = {diff_in_diff:.2f}")

######################################################################

# BAYESIAN
with pm.Model() as model:
    # data
    t = pm.Data("t", df["t"].values, dims="obs_idx")
    treated = pm.Data("treated", df["treated"].values, dims="obs_idx")
    group = pm.Data("group", df["group"].values, dims="obs_idx")
    # priors
    _control_intercept = pm.Normal("control_intercept", 0, 5)
    _treat_intercept_delta = pm.Normal("treat_intercept_delta", 0, 1)
    _trend = pm.Normal("trend", 0, 5)
    _deltap = pm.Normal("deltap", 0, 1)
    sigma = pm.HalfNormal("sigma", 1)
    # expectation
    mu = pm.Deterministic(
        "mu",
        outcome(t, _control_intercept, _treat_intercept_delta, _trend, _deltap, group, treated),
        dims="obs_idx",
    )
    # likelihood
    pm.Normal("obs", mu, sigma, observed=df["y"].values, dims="obs_idx")

with model:
    idata = pm.sample()

az.plot_trace(idata, var_names='~mu');


'''
Milestone

- define a model that will be tailored to your dataset
- 

'''
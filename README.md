# The Effect of Exercise on Mental Health


## Description

Fundamental Question: Does regular exercise reduce symptoms of depression and anxiety in individuals diagnosed with mental health conditions?

Hypothesis: Engaging in regular physical exercise leads to lower levels of depression and anxiety.

Data:
- Primary variables: depression/anxiety scores, exercise frequency, exercise duration, exercise intensity
- Covariates: age, gender, baseline mental health, socioeconomic status, physical health conditions, medication use, therapy involvement
- Potential sources: NHANES, BRFSS, open database studies from OpenICPSR, Kaggle, MIMIC-IV, etc

Methods to Explore:
- Directed acyclic graph
- Control for confounding variables (age, pre-existing conditions, socioeconomic status, etc)
- Propensity score matching
- Difference-in-differences (compare mental health outcomes before/after starting exercise for participants vs non-partipants)--dependent on data available
- Logistic/linear regression
- Bayesian methods to model uncertainty and handle hierarchical structures such as varying effects by demographics

Challenges:
- Data availability may present limitations
- Not all confounding variables may be identified and/or adjusted for, and a high number of observed confounders may be difficult to control
- Self-reported data isn't always reliable (recall, social desirability)
- Potential data imbalance in age, income, etc.
- Different types of exercise might have different effects on mental health

## Project Organization

- `/code` Scripts with prefixes (e.g., `01_import-data.py`,
  `02_clean-data.py`) and functions in `/code/src`.
- `/data` Simulated and real data, the latter not pushed.
- `/figures` PNG images and plots.
- `/output` Output from model runs, not pushed.
- `/presentations` Presentation slides.
- `/private` A catch-all folder for miscellaneous files, not pushed.
- `/writing` Reports, posts, and case studies.
- `/.venv` Hidden project library, not pushed.
- `.gitignore` Hidden Git instructions file.
- `.python-version` Hidden Python version for the reproducible
  environment.
- `requirements.txt` Information on the reproducible environment.

## Reproducible Environment

After cloning this repository, go to the project’s terminal in Positron
and run `python -m venv .venv` to create the `/.venv` project library,
followed by `pip install -r requirements.txt` to install the specified
library versions.

Whenever you install new libraries or decide to update the versions of
libraries you use, run `pip freeze > requirements.txt` to update
`requirements.txt`.

For more details on using GitHub, Quarto, etc. see [ASC
Training](https://github.com/marcdotson/asc-training).

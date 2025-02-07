import numpy as np
import polars as pl
import seaborn as sns
from sklearn.linear_model import LinearRegression

np.random.seed(42)

# Set the parameter values.

intercept = 3
slope_exercise = -2
slope_age = 1.2
slope_gender = 2
slope_conditions = [0.4,0.6,0.8,1]
slope_SES = [-0.5,-1.7,-2.4,-3.3]
n = 100000

sim_data = (
# Simulate predictors using appropriate np.random distributions.
    pl.DataFrame({
        'exercise': np.random.uniform(0, 240, size = n),
        'age': np.random.normal(35,10,size=n),
        'gender':np.random.choice([0,1], size=n).astype(int),
        'conditions':np.random.choice([1,2,3,4,5], size=n).astype(int),
        'SES':np.random.choice([1,2,3,4,5],size=n).astype(int)
})
)

# One-hot encode conditions and SES
sim_data = sim_data.with_columns([
    (pl.col("conditions") == i).cast(pl.Int8).alias(f"conditions_{i}") for i in range(2, 6)
] + [
    (pl.col("SES") == i).cast(pl.Int8).alias(f"SES_{i}") for i in range(2, 6)
])

# Use predictors and parameter values to simulate the outome.
sim_data = sim_data.with_columns([
(intercept + 
slope_exercise * pl.col('exercise') + 
slope_age * pl.col('age') + 
slope_gender * (pl.col('gender') == 1) + 
slope_conditions[0] * (pl.col("conditions") == 2).cast(pl.Int8).alias("conditions_2") +
slope_conditions[1] * (pl.col("conditions") == 3).cast(pl.Int8).alias("conditions_3") +
slope_conditions[2] * (pl.col("conditions") == 4).cast(pl.Int8).alias("conditions_4") +
slope_conditions[3] * (pl.col("conditions") == 5).cast(pl.Int8).alias("conditions_5") +
slope_SES[0] * (pl.col("SES") == 2).cast(pl.Int8).alias("SES_2") +
slope_SES[1] * (pl.col("SES") == 3).cast(pl.Int8).alias("SES_3") +
slope_SES[2] * (pl.col("SES") == 4).cast(pl.Int8).alias("SES_4") +
slope_SES[3] * (pl.col("SES") == 5).cast(pl.Int8).alias("SES_5") +
np.random.normal(0, 3, size = n)).alias('y')
])


sim_data

sns.scatterplot(data=sim_data, x='exercise', y='y')
sns.lmplot(data=sim_data, x='exercise', y='y', height=6, aspect=1, scatter_kws={'s': 10}, line_kws={'color': 'red'})

# Specify the X matrix and y vector.
X = sim_data[['exercise','age','gender','conditions_2','conditions_3','conditions_4','conditions_5','SES_2','SES_3','SES_4','SES_5']]
y = sim_data['y']

# Create a linear regression model.
model = LinearRegression(fit_intercept=True)

# Train the model.
model.fit(X, y)

# Print the coefficients
print(f'Intercept: {model.intercept_}')
print(f'Exercise Slope: {model.coef_[0]}')
print(f'Age Slope: {model.coef_[1]}')
print(f'Gender Slope: {model.coef_[2]}')
for i in range(4):
    print(f'Conditions_{i+2} Slope: {model.coef_[3+i]}')
for i in range(4):
    print(f'SES_{i+2} Slope: {model.coef_[7+i]}')

# Have you recovered the parameters?
'''
n=100
Intercept: 4.680828701419301  # 3; no
Exercise Slope: -2.0161845318234737 # -2; yes
Age Slope: 1.2182294044673265 # 1.2; yes
Gender Slope: 2.7737778165181144 # 2; no
Conditions_2 Slope: 0.2531555685053275 # 0.4; kind of
Conditions_3 Slope: 0.5297217975275962 # 0.6; kind of
Conditions_4 Slope: 0.4282026369570847 # 0.8; no
Conditions_5 Slope: 0.4002248281743458 # 1; no
SES_2 Slope: -2.0669156317467334 # -0.5; no
SES_3 Slope: -2.4464480460046243 # -1.7; no
SES_4 Slope: -3.2691465318741373 # -2.4; no
SES_5 Slope: -3.88511076181787 # -3.3; kind of
'''
'''
n=10000
Intercept: 3.0686296196199407 # 3; yes
Exercise Slope: -1.9998770973416544 # -2; yes
Age Slope: 1.200112529570226 # 1.2; yes
Gender Slope: 1.9962247692174278 # 2; yes
Conditions_2 Slope: 0.3793088476144025 # 0.4; yes
Conditions_3 Slope: 0.5523964193008831 # 0.6; yes
Conditions_4 Slope: 0.7802943022788963 # 0.8; yes
Conditions_5 Slope: 0.9709399513242355 # 1; yes
SES_2 Slope: -0.5657497217461727 # -0.6=5; kind of
SES_3 Slope: -1.7992484906812714 # -1.7; kind of
SES_4 Slope: -2.439256315844938 # -2.4; yes
SES_5 Slope: -3.3693616193116385 # -3.3; kind of
'''
#####################################################
# Import packages and dependencies
#####################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#####################################################
# Read in and observe dataset
#####################################################

df = pd.read_excel(r'.\data\Project\healthdata.xlsx')
df.head()
df.shape

df.info()
df.describe()
df.isnull().sum()
df.nunique()

#####################################################
# Distributions
#####################################################

# individual
plt.figure(figsize=(10,6))
sns.histplot(df["age"], kde=True)
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(df["exercise_score"], kde=True)
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(df["mh_score"], kde=True)
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(df["pec_severity"], bins=3, discrete=True)
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(df["ses"], bins=3, discrete=True)
plt.show()

plt.figure(figsize=(10,6))
sns.countplot(data=df, x="gender")
plt.show()

# combined
fig, axes = plt.subplots(3, 2, figsize=(14, 12))  # 3 rows, 2 columns
fig.suptitle("Variable Distributions", fontsize=16)
axes = axes.flatten()

sns.histplot(df["age"], kde=True, ax=axes[0])
axes[0].set_title("Age Distribution")

sns.histplot(df["exercise_score"], kde=True, ax=axes[1])
axes[1].set_title("Exercise Score Distribution")

sns.histplot(df["mh_score"], kde=True, ax=axes[2])
axes[2].set_title("Mental Health Score Distribution")

sns.histplot(df["pec_severity"], bins=3, discrete=True, ax=axes[3])
axes[3].set_title("Pre-existing Condition Severity")

sns.histplot(df["ses"], bins=3, discrete=True, ax=axes[4])
axes[4].set_title("SES Distribution")

sns.countplot(data=df, x="gender", ax=axes[5])
axes[5].set_title("Gender Count")

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the suptitle
plt.show()

#####################################################
# Correlation Matrix
#####################################################
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

corr = df.corr(numeric_only=True)
plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=True, cmap='coolwarm',center=0)
plt.show()

#####################################################
# Viewing MH vs Exercise
#####################################################

plt.figure(figsize=(10,6))
sns.lmplot(data=df, x="exercise_score", y="mh_score", hue="gender", aspect=1.5)
plt.show()

plt.figure(figsize=(10,6))
sns.lmplot(data=df, x="exercise_score", y="mh_score", col="ses", hue="gender")
plt.show()

plt.figure(figsize=(10,6))
sns.lmplot(data=df, x="exercise_score", y="mh_score", col="pec_severity")
plt.show()
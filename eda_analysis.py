# ==============================
# 1. Import Libraries
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# 2. Load Dataset
# ==============================
df = pd.read_csv("DATASET/titanic.csv")
df.head()

# ==============================
# 3. Basic Data Exploration
# ==============================
df.info()
df.shape
df.describe()
df.isnull().sum()

# ==============================
# 4. Handle Missing Values
# ==============================
df['Age'] = df['Age'].fillna(df['Age'].median())

# ==============================
# 5. Questions Before Analysis
# ==============================
# 1. Which gender survived more?
# 2. Did passenger class affect survival?
# 3. Are there missing values?
# 4. Which age group survived more?
# 5. Which factors affected survival most?

# ==============================
# 6. Data Visualization
# ==============================

# Survival Count
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

# Gender vs Survival
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Gender vs Survival")
plt.show()

# Passenger Class vs Survival
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Passenger Class vs Survival")
plt.show()

# Age Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=20)
plt.title("Age Distribution")
plt.show()

# ==============================
# 7. Correlation Heatmap
# ==============================
numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ==============================
# 8. Outlier Analysis
# ==============================
sns.boxplot(x=df['Fare'])
plt.title("Fare Outliers")
plt.show()

# ==============================
# 9. Final Conclusions
# ==============================
# 1. Females had higher survival rates
# 2. First-class passengers survived more
# 3. Age had missing values (handled using median)
# 4. Fare contains outliers
# 5. Passenger class strongly affected survival
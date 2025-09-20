# main.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris


# Task 1: Load and Explore the Dataset

try:
    iris = load_iris(as_frame=True)
    df = iris.frame  # convert to pandas DataFrame
    print("✅ Dataset loaded successfully!\n")

    # Display first few rows
    print("First 5 rows of dataset:")
    print(df.head(), "\n")

    # Info about dataset
    print("Dataset Info:")
    print(df.info(), "\n")

    # Check missing values
    print("Missing values per column:")
    print(df.isnull().sum(), "\n")

    # Clean data (if missing values existed, here we’d fill/drop them)
    df = df.dropna()

except FileNotFoundError:
    print("❌ File not found. Please check the dataset path.")
except Exception as e:
    print("❌ An error occurred while loading the dataset:", str(e))



# Task 2: Basic Data Analysis


# Summary statistics
print("Summary Statistics:")
print(df.describe(), "\n")

# Group by species and compute mean of each feature
grouped = df.groupby("target").mean()
print("Mean values grouped by species (target):")
print(grouped, "\n")

# Observations
print("Observation: Sepal and petal sizes differ significantly among species.\n")



# Task 3: Data Visualization


sns.set(style="whitegrid")  # nice style for seaborn plots

# 1. Line chart (pretend index = time for simplicity)
plt.figure(figsize=(6, 4))
plt.plot(df.index, df["sepal length (cm)"], label="Sepal Length")
plt.title("Line Chart: Sepal Length Trend")
plt.xlabel("Index (as Time)")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.show()

# 2. Bar chart (average petal length per species)
plt.figure(figsize=(6, 4))
sns.barplot(x="target", y="petal length (cm)", data=df, ci=None)
plt.title("Bar Chart: Avg Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()

# 3. Histogram (distribution of sepal width)
plt.figure(figsize=(6, 4))
plt.hist(df["sepal width (cm)"], bins=15, color="skyblue", edgecolor="black")
plt.title("Histogram: Sepal Width Distribution")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter plot (sepal length vs petal length)
plt.figure(figsize=(6, 4))
sns.scatterplot(x="sepal length (cm)", y="petal length (cm)",
                hue="target", data=df, palette="deep")
plt.title("Scatter Plot: Sepal vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()


# What I Observed from the Data

# 1. Sepal length and petal length vary a lot, while sepal width stays kind of steady.
# 2. Iris-setosa has the smallest petals, Iris-virginica the biggest.
# 3. The petal length histogram shows Iris-setosa is easy to spot, the other two species overlap more.
# 4. Petal length and width go up together, and this helps separate the three species.


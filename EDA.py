# Databricks notebook source
# MAGIC %pip install scikit-learn ucimlrepo

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Libraries

# COMMAND ----------

from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Automobile Dataset

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC **Data Dictionary**
# MAGIC 1. symboling – Insurance risk rating (-3 = safest, +3 = riskiest).
# MAGIC 2. normalized-losses – Normalized average loss per insured vehicle.
# MAGIC 3. make – Manufacturer (e.g., Audi, BMW, Toyota).
# MAGIC 4. fuel-type – Fuel type: gas, diesel.
# MAGIC 5. aspiration – Turbocharged or standard.
# MAGIC 6. num-of-doors – Number of doors (two, four).
# MAGIC 7. body-style – Car body type (e.g., sedan, hatchback, wagon, convertible).
# MAGIC 8. drive-wheels – Drive system (fwd, rwd, 4wd).
# MAGIC 9. engine-location – Position of engine (front, rear).
# MAGIC 10. wheel-base – Distance between front and rear axles (inches).
# MAGIC 11. length – Car length (inches).
# MAGIC 12. width – Car width (inches).
# MAGIC 13. height – Car height (inches).
# MAGIC 14. curb-weight – Weight of the car without passengers or cargo (lbs).
# MAGIC 15. engine-type – Type of engine (e.g., dohc, ohcv, rotor).
# MAGIC 16. num-of-cylinders – Number of engine cylinders (two, three, four, five, six, eight, 
# MAGIC twelve).
# MAGIC 17. engine-size – Engine displacement size (cc).
# MAGIC 18. fuel-system – Type of fuel system (e.g., mpfi, 2bbl, idi).
# MAGIC 19. bore – Diameter of the cylinder bore (inches).
# MAGIC 20. stroke – Length of piston stroke (inches).
# MAGIC 21. compression-ratio – Ratio of cylinder volume at bottom vs. top of stroke.
# MAGIC 22. horsepower – Engine horsepower.
# MAGIC 23. peak-rpm – Revolutions per minute at maximum horsepower.
# MAGIC 24. city-mpg – Miles per gallon in city driving.
# MAGIC 25. highway-mpg – Miles per gallon on highway driving.
# MAGIC 26. price – Manufacturer’s suggested retail price (USD).
# MAGIC

# COMMAND ----------

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
automobile = fetch_ucirepo(id=10) 

X = automobile.data.features 
y = automobile.data.targets 

auto_df = pd.concat([y, X], axis=1)
auto_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploring the Automobile Dataset

# COMMAND ----------

# MAGIC %md
# MAGIC - The dataset consists of 26 columns and 205 records.
# MAGIC - Some columns (price, peak-rpm, horsepower, stroke, bore, num-of-doors, and normalized-losses) have null values.
# MAGIC - num-of-doors is of data type float. We can convert this to int since there are no number of doors in decimal places later on.

# COMMAND ----------

auto_df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC - There are no records having a -3 symboling.
# MAGIC - Most cars lean more above neutral insurance risk.
# MAGIC - Data shows a wide variety of cars (from simple to high-end or luxury cars).

# COMMAND ----------

auto_df.describe().T

# COMMAND ----------

# MAGIC %md
# MAGIC - Check for duplicates

# COMMAND ----------

duplicated = auto_df.duplicated()
print("Number of Duplicates:", duplicated.sum())

# COMMAND ----------

# MAGIC %md
# MAGIC - As mentioned earlier, there are columns (price, peak-rpm, horsepower, stroke, bore, num-of-doors, and normalized-losses) that have null values.
# MAGIC - Columns with null values show that their distribution is not normally distributed. Since data is skewed, we use median to fill in the missing values.

# COMMAND ----------

auto_df.isna().sum()[auto_df.isna().sum() > 0]

# COMMAND ----------

auto_df[auto_df["normalized-losses"].isna()].display()

# COMMAND ----------

null_cols = ["price", "peak-rpm", "horsepower", "stroke", "bore", "num-of-doors", "normalized-losses"]

fig, axes = plt.subplots(3, 3, figsize=(10,6))
fig.suptitle("Distribution of Columns with Null Values")

for idx, col in enumerate(null_cols):
    ax = axes.flatten()[idx]
    sns.histplot(data=auto_df, x=col, kde=True, ax=ax)
    ax.set_xlabel(col)

[fig.delaxes(ax) for ax in axes.flatten() if not ax.has_data()]

fig.tight_layout()

# COMMAND ----------

auto_df = auto_df.fillna(auto_df[null_cols].median())

# COMMAND ----------

# convert num-of-doors to int
auto_df = auto_df.astype({'num-of-doors': 'int64'})
auto_df["num-of-doors"].dtypes

# COMMAND ----------

auto_df.describe().T

# COMMAND ----------

# MAGIC %md
# MAGIC - Our data is imbalance since there are only 3 records of -2 symboling and most records are of 0 or 1 symboling.
# MAGIC - Since -1 and -2 are considered low risk, we can replace those -2 symboling records to -1.

# COMMAND ----------

auto_df["symboling"].value_counts()

# COMMAND ----------

auto_df["symboling"] = auto_df["symboling"].replace(-2, -1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Visualization

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distribution of categorical columns with symboling

# COMMAND ----------

# MAGIC %md
# MAGIC - Most cars cluster around symboling 0 and 1 which means they are on average risk.
# MAGIC - High-risk(2-3) symboling appear more in some cars that are luxury or sportier.
# MAGIC - Low-risk(-1) symboling is common in Volvo

# COMMAND ----------

plt.figure(figsize=(30,10))
sns.countplot(data=auto_df, x="make", hue="symboling", palette='viridis')
plt.title("Distribution of make with symboling")

# COMMAND ----------

# MAGIC %md
# MAGIC - Most cars use mpfi and are mainly in symboling 0.
# MAGIC - Most cars have ohc engines that is mainly around 0-1. dohc and rotor appear more on higher symboling which are possibly sports-oriented engine.
# MAGIC - Most engine location of cars are at front. Few rear cars have symboling 3.
# MAGIC - Most cars are fwd with mainly 1 symboling. rwd shows more spread with much more higher symboling compared to fwd and 4wd.
# MAGIC - Sedans are the most common and is mainly on symboling 0. Convertibles and Hatchback have more higher risks.
# MAGIC - std aspiration mostly falls under symboling 0-1.
# MAGIC - Most cars use gas and covers a wide range of symboling values.

# COMMAND ----------

categorical_cols = auto_df.drop("make", axis=1).select_dtypes(include=["object"]).columns

fig, axes = plt.subplots(3, 3, figsize=(20,15))
fig.suptitle("Distribution of Categorical Columns with symboling")

for idx, col in enumerate(categorical_cols):
    ax = axes.flatten()[idx]
    sns.countplot(data=auto_df, x=col, hue="symboling", ax=ax, palette="viridis")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")

[fig.delaxes(ax) for ax in axes.flatten() if not ax.has_data()]
plt.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distribution of Price vs. categorical columns with symboling

# COMMAND ----------

# MAGIC %md
# MAGIC - Mercedez-Benz, Jaguar, BMW, and Porsche shows the highest prices.
# MAGIC - Most models (luxury and sporty versions) with higher prices tend to have higher symboling values, meaning they're riskier.
# MAGIC - Chevrolet, Honda, Isuzu have lower prices and ranges from lower to moderate symboling.

# COMMAND ----------

plt.figure(figsize=(30,10))
sns.barplot(data=auto_df, x="make", y="price", errorbar=None, hue="symboling", palette="viridis")
plt.title("Distribution of price vs. make with symboling")

# COMMAND ----------

# MAGIC %md
# MAGIC - mpfi and spdi cars are more expensive that have more higher symboling, suggesting high-performance engines. 
# MAGIC - ohcf is linked with a higher price and higher risk. All records with symboling -2 are ohc engine.
# MAGIC - Most cars have front engines, which show a wide price range. Rear-engine cars are very expensive and have higher symboling.
# MAGIC - rwd cars are most expensive and tend to have higher risk.
# MAGIC - hardtop and convertible cars are leaning towards higher price and higher risk.
# MAGIC - turbo cars show higher prices but standard aspiration shows higher risk with symboling 3.
# MAGIC - Cars that use gas has wide price and risk range whereas those that use diesel appear to be less risky.
# MAGIC

# COMMAND ----------

fig, axes = plt.subplots(3, 3, figsize=(20,15))
fig.suptitle("Distribution of Price vs. Categorical Columns with symboling")

for idx, col in enumerate(categorical_cols):
    ax = axes.flatten()[idx]
    sns.barplot(data=auto_df, x=col, y="price", errorbar=None, hue="symboling", palette="viridis", ax=ax)
    ax.set_xlabel(col)
    ax.set_ylabel("Price")

[fig.delaxes(ax) for ax in axes.flatten() if not ax.has_data()]
plt.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distribution of numerical columns vs. symboling

# COMMAND ----------

# MAGIC %md
# MAGIC - Prices are generally higher for lower symboling (-1). Which means, safer cars tend to be more expensive.
# MAGIC - Riskier cars tend to be more fuel-efficient.
# MAGIC - Risker cars have stronger engines and more horsepower.
# MAGIC - Heavier cars are safer than lighter, smaller cars.
# MAGIC - There are some outliers but are not invalid data. We will be keeping this outliers since we only have few records.

# COMMAND ----------

numerical_cols = auto_df.drop({"symboling", "normalized-losses"}, axis=1).select_dtypes(exclude=["object"]).columns

fig, axes = plt.subplots(4, 4, figsize=(20, 15))
fig.suptitle("Distribution of Numerical Columns vs. Symboling")

for idx, col in enumerate(numerical_cols):
  ax = axes.flatten()[idx]
  sns.boxplot(data=auto_df, x="symboling", y=col, ax=ax)
  ax.set_xlabel("symboling")
  ax.set_ylabel(col)

[fig.delaxes(ax) for ax in axes.flatten() if not ax.has_data()]
plt.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC - Cars with higher symboling tend to cause higher insurance losses.

# COMMAND ----------

sns.boxplot(data=auto_df, x="symboling", y="normalized-losses")
plt.title("Symboling vs. Normalized Losses")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Correlation of data

# COMMAND ----------

# MAGIC %md
# MAGIC The following are some insights of the heatmap:
# MAGIC - highway-mpg and city-mpg are nearly a perfect correlation with 0.97.
# MAGIC - The bigger the engine size, the higher its price and horsepower.
# MAGIC - The heavier the car, the longer, wider, and expensive it is.

# COMMAND ----------

numerical_corr = auto_df.drop("symboling", axis=1).select_dtypes(exclude=["object"])
corr = numerical_corr.corr()

plt.figure(figsize=(13,10))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save cleaned data to csv

# COMMAND ----------

auto_df.to_csv("data_modeling/automobile_cleaned.csv")

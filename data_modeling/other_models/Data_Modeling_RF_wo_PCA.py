# Databricks notebook source
# MAGIC %pip install scikit-learn imblearn

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Libraries

# COMMAND ----------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# COMMAND ----------

csv_path = "/Workspace/Users/princess.pabulayan@inchcape.com/automobile_dataset/data_modeling/automobile_cleaned.csv"
auto_df = pd.read_csv(csv_path)
auto_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Preprocessing

# COMMAND ----------

auto_df = auto_df.drop("Unnamed: 0", axis=1)

# COMMAND ----------

# Split features and targets
X = auto_df.drop("symboling", axis=1)
y = auto_df["symboling"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Separate numerical and categorical features
numerical_feat = X.select_dtypes(exclude=["object"]).columns
categorical_feat = X.select_dtypes(include=["object"]).columns

# COMMAND ----------

# MAGIC %md
# MAGIC ### OHE and Standardization on Training Data

# COMMAND ----------

ohe = OneHotEncoder(drop="first", sparse_output=False)

ohe_encoded_train = ohe.fit_transform(X_train[categorical_feat])
ohe_df_train = pd.DataFrame(
  ohe_encoded_train,
  columns=ohe.get_feature_names_out(categorical_feat),
  index=X_train.index
)
X_train_encoded = pd.concat([X_train.drop(categorical_feat, axis=1), ohe_df_train], axis=1)

# COMMAND ----------

scaler = StandardScaler()

scaled_train = scaler.fit_transform(X_train_encoded[numerical_feat])
scaled_df_train = pd.DataFrame(
    scaled_train,
    columns=scaler.get_feature_names_out(numerical_feat),
    index=X_train_encoded.index
)
X_train_scaled = pd.concat([scaled_df_train, X_train_encoded.drop(scaled_df_train.columns, axis=1)], axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### OHE and Standardization on Testing Data

# COMMAND ----------

ohe.fit(X_train[categorical_feat])
ohe_encoded_test = ohe.transform(X_test[categorical_feat])
ohe_df_test = pd.DataFrame(
  ohe_encoded_test,
  columns=ohe.get_feature_names_out(categorical_feat),
  index=X_test.index
)
X_test_encoded = pd.concat([X_test.drop(categorical_feat, axis=1), ohe_df_test], axis=1)

# COMMAND ----------

scaler.fit(X_train_encoded[numerical_feat])
scaled_test = scaler.transform(X_test_encoded[numerical_feat])
scaled_df_test = pd.DataFrame(
    scaled_test, 
    columns=scaler.get_feature_names_out(numerical_feat),
    index=X_test_encoded.index
)
X_test_scaled = pd.concat([scaled_df_test, X_test_encoded.drop(scaled_df_test.columns, axis=1)], axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Oversampling Using SMOTE

# COMMAND ----------

# MAGIC %md
# MAGIC - To address the class imbalance, we use SMOTE to create synthetic data.

# COMMAND ----------

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# COMMAND ----------

y_train_resampled.value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Modelling Using Random Forest

# COMMAND ----------

clf = RandomForestClassifier(n_estimators=84, random_state=42)
clf.fit(X_train_resampled, y_train_resampled)
y_pred = clf.predict(X_test_scaled)

# COMMAND ----------

print(classification_report(y_test, y_pred))

# COMMAND ----------

conf_matrix = confusion_matrix(y_test, y_pred)

sns.heatmap(conf_matrix, annot=True, cmap="YlGnBu",
            xticklabels=auto_df["symboling"].unique(),
            yticklabels=auto_df["symboling"].unique())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

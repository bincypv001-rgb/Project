#!/usr/bin/env python
# coding: utf-8

# #### EDA for Solar Power Generation Dataset

# In[5]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


# Load dataset
file_path = "solarpowergeneration (1).csv"
df = pd.read_csv(file_path)


# In[7]:


df


# In[8]:


#  Basic Information
print("----- BASIC INFO -----")
print(df.info())
print("\n----- SHAPE -----")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n----- COLUMN NAMES -----")
print(df.columns.tolist())


# In[9]:


# Rename columns for easy access
df.columns = [col.strip().lower().replace("-", "_").replace("(", "").replace(")", "") for col in df.columns]


# In[10]:


# Descriptive Statistics
print("\n----- SUMMARY STATISTICS -----")
print(df.describe().T)


# In[11]:


# 3️⃣ Missing Values & Duplicates
print("\n----- MISSING VALUES -----")
print(df.isnull().sum())

print("\n----- DUPLICATE ROWS -----")
print(df.duplicated().sum())


# In[12]:


# Univariate Analysis

num_features = df.select_dtypes(include=['int64', 'float64']).columns

plt.figure(figsize=(15, 12))
for i, col in enumerate(num_features, 1):
    plt.subplot(4, 3, i)
    sns.histplot(df[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Distribution of {col}', fontsize=10)
plt.tight_layout()
plt.show()


# In[13]:


#  Outlier Detection using Boxplots
plt.figure(figsize=(15, 12))
for i, col in enumerate(num_features, 1):
    plt.subplot(4, 3, i)
    sns.boxplot(x=df[col], color='lightgreen')
    plt.title(f'Boxplot of {col}', fontsize=10)
plt.tight_layout()
plt.show()


# In[14]:


# Pairplot to See Relationships
sns.pairplot(df.sample(300), diag_kind='kde')
plt.suptitle("Pairplot of Features (Sample)", y=1.02)
plt.show()


# In[15]:


# Correlation Analysis
plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap")
plt.show()


# In[16]:


# Strong correlations with target
target_corr = corr["power_generated"].sort_values(ascending=False)
print("\n----- CORRELATION WITH TARGET (power_generated) -----")
print(target_corr)


# In[17]:


#  Bivariate Analysis with Target
target = 'power_generated'

plt.figure(figsize=(15, 12))
for i, col in enumerate(num_features.drop(target), 1):
    plt.subplot(4, 3, i)
    sns.scatterplot(data=df, x=col, y=target, color='teal', alpha=0.6)
    plt.title(f'{col} vs {target}')
plt.tight_layout()
plt.show()


# In[18]:


# Insights Summary
print("----- INSIGHTS -----")
print("""
'distance_to_solar_noon' likely shows a sinusoidal pattern affecting energy generation.
'temperature' and 'humidity' may have moderate influence on power output.
'sky_cover' (cloudiness) negatively affects energy production.
Wind-related variables (wind_speed, average_wind_speed_period) show mild correlation.
No major missing data; outliers may exist in 'power_generated' due to environmental spikes.
Data suitable for regression modeling after normalization or feature scaling.
""")


# ### MODEL BUILDING

# In[26]:


# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns


# #### Handle missing values

# In[27]:


imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)


# #### TRAIN–TEST SPLIT + SCALING

# In[28]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# #### DEFINE ALL MODELS

# In[29]:


models = {
    "Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(alpha=0.01),
    "SVR": SVR(kernel="rbf"),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
}


# #### MODEL EVALUATION FUNCTION

# In[30]:


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    
    return mae, mse, rmse, r2


# #### TRAIN ALL MODELS + STORE RESULTS

# In[31]:


results = {}

for name, model in models.items():
    print(f" Training {name}...")
    
    # Scaled models
    if name in ["Linear Regression", "Lasso Regression", "SVR"]:
        model.fit(X_train_scaled, y_train)
        metrics = evaluate_model(model, X_test_scaled, y_test)
    else:
        # Tree models don't need scaling
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

    results[name] = metrics


# In[32]:


#### COMPARE ALL MODELS


# In[33]:


results_df = pd.DataFrame(
    results,
    index=["MAE", "MSE", "RMSE", "R²"]
).T

results_df


# #### CROSS-VALIDATION

# In[35]:


cv_scores = {}

for name, model in models.items():
    if name in ["Linear Regression", "Lasso Regression", "SVR"]:
        score = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="r2")
    else:
        score = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    
    cv_scores[name] = score

cv_df = pd.DataFrame(cv_scores)
cv_df


# #### RESIDUAL PLOTS

# In[37]:


for name, model in models.items():
    plt.figure(figsize=(6,4))

    if name in ["Linear Regression", "Lasso Regression", "SVR"]:
        preds = model.predict(X_test_scaled)
    else:
        preds = model.predict(X_test)

    residuals = y_test - preds

    sns.scatterplot(x=preds, y=residuals)
    plt.axhline(0, color='red')
    plt.title(f"Residual Plot - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.show()


# #### FEATURE IMPORTANCE (TREE MODELS)

# In[38]:


tree_models = ["Random Forest", "Gradient Boosting", "XGBoost"]

for name in tree_models:
    model = models[name]
    importance = model.feature_importances_

    plt.figure(figsize=(8,4))
    sns.barplot(x=importance, y=X.columns)
    plt.title(f"Feature Importance - {name}")
    plt.show()


# #### BEST MODEL IDENTIFICATION

# In[39]:


best_model_name = results_df["R²"].idxmax()
best_model_score = results_df["R²"].max()

print("=================================================")
print(f" BEST MODEL: {best_model_name}")
print(f" R² Score: {best_model_score}")
print("=================================================")


# #### Show Accuracy (R² Score) of Each Model

# In[40]:


print("\nModel Accuracy (R² Score):")
print("====================================")

for name, model in models.items():
    
    # Scaled models
    if name in ["Linear Regression", "Lasso Regression", "SVR"]:
        r2 = r2_score(y_test, model.predict(X_test_scaled))
    else:
        r2 = r2_score(y_test, model.predict(X_test))
    
    print(f"{name}: {r2:.4f}")


# In[ ]:





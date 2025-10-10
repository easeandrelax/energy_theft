# Data handling
import pandas as pd
import numpy as np
import os
import datetime
import warnings

# Visualization and Modeling
import matplotlib
# CRITICAL FIX for headless environment: Use the 'Agg' backend to save plots to files
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE 
import optuna
import joblib

warnings.filterwarnings("ignore")

# --- CONFIGURATION & FILE PATHS ---
PROJECT_ROOT = '/home/swathi/energy_theft_project'
RAW_INPUT_FILE = os.path.join(PROJECT_ROOT, 'data', 'consumption_data.csv')
CLEANED_OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'data', 'cleaned_features.csv')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'plots')

CUSTOMER_ID_COL_NAME = 'CONS_NO'
FLAG_COLUMN = 'FLAG'
VALUE_COL_NAME = 'Consumption' 

# Ensure plots directory exists
os.makedirs(PLOTS_DIR, exist_ok=True)

print(f"--- STARTING DATA PIPELINE ---")

# --- 1. DATA LOADING AND TRANSFORMATION (FIXED FOR REAL FLAG) ---

# 1. Load the CSV, assuming no proper header
df = pd.read_csv(RAW_INPUT_FILE, header=None, low_memory=False)

# 2. Promote first row to headers and clean
# NOTE: We assume the FLAG column is now one of the columns in the first row.
new_header = df.iloc[0].tolist()
new_header[0] = CUSTOMER_ID_COL_NAME 
df.columns = new_header
df = df[1:].reset_index(drop=True)
df.columns = df.columns.astype(str).str.strip()
df = df.loc[:, ~df.columns.duplicated()]

# 3. Reshape dataset (Melt)
# CRITICAL CHANGE: Keep both CONS_NO and FLAG as ID variables
df_long = df.melt(
    id_vars=[CUSTOMER_ID_COL_NAME, FLAG_COLUMN], 
    var_name="Date",
    value_name=VALUE_COL_NAME
)

# 4. Convert and clean consumption values
df_long[VALUE_COL_NAME] = pd.to_numeric(df_long[VALUE_COL_NAME], errors='coerce')
df_long.dropna(subset=[VALUE_COL_NAME], inplace=True)

# 5. Convert date strings to datetime
df_long["Date"] = pd.to_datetime(df_long["Date"], errors="coerce", dayfirst=True)
df_long.dropna(subset=["Date"], inplace=True)

# 6. Convert FLAG to numeric (if it wasn't already)
df_long[FLAG_COLUMN] = pd.to_numeric(df_long[FLAG_COLUMN], errors='coerce').astype('Int64')
df_long.dropna(subset=[FLAG_COLUMN], inplace=True)

# --- 2. DATA CLEANING AND LABELING (REMOVED RANDOM ASSIGNMENT) ---

# 7. Imputation and Outlier Treatment
df_long[VALUE_COL_NAME] = df_long.groupby(CUSTOMER_ID_COL_NAME)[VALUE_COL_NAME].transform(
    lambda x: x.interpolate(method='linear', limit_direction="both")
)
valid_customers = df_long.groupby(CUSTOMER_ID_COL_NAME)[VALUE_COL_NAME].sum()
valid_customers = valid_customers[valid_customers > 0].index
df_long = df_long[df_long[CUSTOMER_ID_COL_NAME].isin(valid_customers)]

q_low, q_high = df_long[VALUE_COL_NAME].quantile([0.01, 0.99])
df_long[VALUE_COL_NAME] = df_long[VALUE_COL_NAME].clip(lower=q_low, upper=q_high)

# --- 3. FEATURE ENGINEERING (Z-SCORE & Simplified Time-Series) ---

# 1. Feature Creation on df_long
df_long['Consumption_Diff'] = df_long.groupby(CUSTOMER_ID_COL_NAME)[VALUE_COL_NAME].diff()
df_long['Consumption_Lag1'] = df_long.groupby(CUSTOMER_ID_COL_NAME)[VALUE_COL_NAME].shift(1)
df_long['Month'] = df_long['Date'].dt.month


# 2. Aggregation (Creating the Features DataFrame)
features = df_long.groupby(CUSTOMER_ID_COL_NAME).agg(
    # Basic Aggregates
    cons_mean=(VALUE_COL_NAME, 'mean'),
    cons_total=(VALUE_COL_NAME, 'sum'),
    
    # Advanced: Volatility of Daily Changes
    diff_std=('Consumption_Diff', 'std'),
    
    # Advanced: Stability (Correlation between today and yesterday)
    lag1_corr=('Consumption_Lag1', lambda x: x.corr(df_long.loc[x.index, VALUE_COL_NAME])),
    
    # Advanced: Seasonal Variability 
    month_std=('Consumption', lambda x: x.groupby(df_long.loc[x.index, 'Month']).std().mean()),
    
    # Target Flag (Max works because 0/1 are constant per customer)
    FLAG=(FLAG_COLUMN, 'max')
).reset_index()


# --- NEW: Z-SCORE OUTLIER FEATURE ---
features['cons_total_zscore'] = (
    features['cons_total'] - features['cons_total'].mean()
) / features['cons_total'].std()
# --- END NEW Z-SCORE FEATURE ---


# Final cleaning and saving
features = features.fillna(0)

features.columns = ['CONS_NO', 'cons_mean', 'cons_total', 
                    'diff_std', 'lag1_corr', 'month_std', 'FLAG', 'cons_total_zscore']

print("Successfully created advanced Z-Score and time-series features.")
features.to_csv(CLEANED_OUTPUT_FILE, index=False)


# --- 4. MODELING AND RESAMPLING ---

X = features.drop([CUSTOMER_ID_COL_NAME, FLAG_COLUMN], axis=1)
y = features[FLAG_COLUMN]

# Check the true imbalance ratio
true_theft_count = y.sum()
true_normal_count = len(y) - true_theft_count
print(f"\nTRUE Target Distribution (Real Imbalance):\nNormal (0): {true_normal_count}\nTheft (1): {true_theft_count}")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# SMOTE Resampling for Training Data
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)
print(f"SMOTE Resampled Training Shape: {X_res.shape}")
print(f"SMOTE Resampled Target Distribution:\n{y_res.value_counts()}")

# --- Initial Model Training (Using Resampled Data) ---
model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)
model.fit(X_res, y_res) 

y_prob = model.predict_proba(X_test)[:, 1]

print("\n--- INITIAL MODEL RESULTS (Trained with SMOTE) ---")
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))


# --- 5. OPTUNA TUNING (Using Resampled Data for Training) ---

def objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "random_state": 42
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_res, y_res, eval_set=[(X_test, y_test)], verbose=False) 
    preds = model.predict_proba(X_test)[:,1]
    return roc_auc_score(y_test, preds)

print("\n--- OPTUNA TUNING (30 Trials) ---")
try:
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, show_progress_bar=False) 
    
    print("Best Params (from Optuna study):\n", study.best_params)
    
    best_params = study.best_params
    best_params["random_state"] = 42 
    
except Exception as e:
    print(f"Optuna error. Falling back to hardcoded parameters.")
    best_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 3,
        "learning_rate": 0.1623391231965192,
        "n_estimators": 236,
        "subsample": 0.99904295023646,
        "colsample_bytree": 0.7592636429453455,
        "min_child_weight": 4,
        "random_state": 42
    }


# Final Model Training and Evaluation
final_model = xgb.XGBClassifier(**best_params)
final_model.fit(X_res, y_res)

y_pred_final = final_model.predict(X_test)
y_prob_final = final_model.predict_proba(X_test)[:,1]

print("\n--- FINAL MODEL RESULTS (Tuned with SMOTE) ---")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_final))
print("\nClassification Report:\n", classification_report(y_test, y_pred_final))
print("\nROC-AUC Score:", roc_auc_score(y_test, y_prob_final))


# 1. Define the filename for the saved model
model_filename = 'energy_theft_model.joblib' 

# 2. Use joblib.dump() to save the trained model object to the file
joblib.dump(model, model_filename)

print(f"✅ Trained model successfully saved to {model_filename}")
# --- 6. VISUALIZATION (Plots are saved to the PLOTS_DIR) ---

print(f"\n--- VISUALIZATION (Saving Plots to files in the '{PLOTS_DIR}' directory) ---")

# Plot 1: Class Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x="FLAG", data=features)
plt.title("Class Distribution (Normal vs Theft)")
plt.savefig(os.path.join(PLOTS_DIR, '01_class_distribution_final.png'))
plt.close()

# Plot 2: Distribution of Daily Consumption Variability (diff_std)
plt.figure(figsize=(8, 5))
sns.kdeplot(data=features, x="diff_std", hue="FLAG", fill=True)
plt.title("Distribution of Daily Consumption Variability (diff_std)")
plt.savefig(os.path.join(PLOTS_DIR, '02_diff_std_distribution.png'))
plt.close()

# Plot 3: Distribution of Total Consumption Z-Score (cons_total_zscore)
plt.figure(figsize=(8, 5))
sns.kdeplot(data=features, x="cons_total_zscore", hue="FLAG", fill=True)
plt.title("Distribution of Total Consumption Z-Score")
plt.savefig(os.path.join(PLOTS_DIR, '03_zscore_distribution.png'))
plt.close()

# Plot 4: Feature Correlation Heatmap
corr = features.drop([CUSTOMER_ID_COL_NAME, FLAG_COLUMN], axis=1).corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap (Advanced Features)")
plt.savefig(os.path.join(PLOTS_DIR, '04_feature_correlation_heatmap.png'))
plt.close()


# --- RE-ADDED MISSING PLOTS ---

# Plot 5: Consumption Trends for Sample Customers (Saves three files: 05_customer_1_trend.png, etc.)
sample_customers = df_long[CUSTOMER_ID_COL_NAME].unique()[:3]
for i, cust in enumerate(sample_customers):
    cust_data = df_long[df_long[CUSTOMER_ID_COL_NAME] == cust]

    if not cust_data.empty:
        plt.figure(figsize=(12,4))
        plt.plot(cust_data["Date"], cust_data[VALUE_COL_NAME])
        plt.title(f"Consumption Trend - Customer {cust}")
        plt.xlabel("Date")
        plt.ylabel("Consumption (kWh)")
        plt.savefig(os.path.join(PLOTS_DIR, f'05_customer_{i+1}_trend.png'))
        plt.close()

# Plot 6: Total Daily Energy Consumption
daily_total = df_long.groupby("Date")[VALUE_COL_NAME].sum()
plt.figure(figsize=(12,5))
plt.plot(daily_total.index, daily_total.values)
plt.title("Total Daily Energy Consumption (Aggregate)")
plt.xlabel("Date")
plt.ylabel("Total Consumption (kWh)")
plt.savefig(os.path.join(PLOTS_DIR, '06_total_daily_consumption.png'))
plt.close()

print(f"✅ All 6 plots saved to the '{PLOTS_DIR}' directory.")
print("--- PIPELINE COMPLETE ---")
'''
# --- 6. VISUALIZATION (Saving to Files) ---
print(f"\n--- VISUALIZATION (Saving Plots to files in the '{PLOTS_DIR}' directory) ---")

# Plot 1: Class Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x="FLAG", data=features)
plt.title("Class Distribution (Normal vs Theft) - REAL LABELS")
plt.savefig(os.path.join(PLOTS_DIR, '01_class_distribution_REAL.png'))
plt.close()

# ... (Rest of the plotting logic remains the same for analysis) ...

# Plot 4: Distribution of Z-Score
plt.figure(figsize=(8, 5))
sns.kdeplot(data=features, x="cons_total_zscore", hue="FLAG", fill=True)
plt.title("Distribution of Total Consumption Z-Score")
plt.savefig(os.path.join(PLOTS_DIR, '04_zscore_distribution.png'))
plt.close()

# Plot 5: Distribution of Consumption Variability
plt.figure(figsize=(8, 5))
sns.kdeplot(data=features, x="diff_std", hue="FLAG", fill=True)
plt.title("Distribution of Daily Consumption Variability (diff_std)")
plt.savefig(os.path.join(PLOTS_DIR, '05_diff_std_distribution.png'))
plt.close()

# Plot 6: Feature Correlation Heatmap
corr = features.drop([CUSTOMER_ID_COL_NAME, FLAG_COLUMN], axis=1).corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.savefig(os.path.join(PLOTS_DIR, '06_feature_correlation_heatmap.png'))
plt.close()

print(f"✅ All 6 plots saved to the '{PLOTS_DIR}' directory.")
print("--- PIPELINE COMPLETE ---")
'''

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt


# ==============================
# LOAD IPO DATA
# ==============================

df = pd.read_excel("ipo_data.xlsx")
df.columns = df.columns.str.strip()
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]


# ==============================
# FETCH NIFTY 50 DATA
# ==============================

print("Fetching Nifty 50 data...")
nifty = yf.download("^NSEI", period="10y", interval="1d", auto_adjust=True)
nifty = nifty[['Close']].copy()
nifty.columns = ['Nifty_Close']
nifty.index = pd.to_datetime(nifty.index)


# ==============================
# ENGINEER NIFTY FEATURES
# ==============================

nifty['Nifty_Ret_5d']  = nifty['Nifty_Close'].pct_change(5)  * 100
nifty['Nifty_Ret_20d'] = nifty['Nifty_Close'].pct_change(20) * 100
nifty['Nifty_Ret_50d'] = nifty['Nifty_Close'].pct_change(50) * 100

daily_ret = nifty['Nifty_Close'].pct_change()
nifty['Nifty_Vol_20d'] = daily_ret.rolling(20).std() * 100

nifty['Nifty_MA50']  = nifty['Nifty_Close'].rolling(50).mean()
nifty['Nifty_MA200'] = nifty['Nifty_Close'].rolling(200).mean()
nifty['Above_MA50']  = (nifty['Nifty_Close'] > nifty['Nifty_MA50']).astype(int)
nifty['Above_MA200'] = (nifty['Nifty_Close'] > nifty['Nifty_MA200']).astype(int)

delta = nifty['Nifty_Close'].diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
rs    = gain / loss
nifty['Nifty_RSI'] = 100 - (100 / (1 + rs))

nifty.dropna(inplace=True)

nifty_features = [
    'Nifty_Ret_5d', 'Nifty_Ret_20d', 'Nifty_Ret_50d',
    'Nifty_Vol_20d', 'Above_MA50', 'Above_MA200', 'Nifty_RSI'
]

print(f"Nifty data loaded: {len(nifty)} trading days")


# ==============================
# MERGE IPO + NIFTY ON DATE
# ==============================

date_col = None
for col in df.columns:
    if 'date' in col.lower() or 'listing' in col.lower():
        date_col = col
        break

if date_col is None:
    raise ValueError("No date column found.")

print(f"Using date column: '{date_col}'")

df[date_col] = pd.to_datetime(df[date_col], errors='coerce').astype('datetime64[ns]')
df = df.dropna(subset=[date_col])

nifty_reset = nifty[nifty_features].reset_index()
nifty_reset.columns = ['Date'] + nifty_features
nifty_reset['Date'] = pd.to_datetime(nifty_reset['Date']).astype('datetime64[ns]')

df = pd.merge_asof(
    df.sort_values(date_col),
    nifty_reset.sort_values('Date'),
    left_on=date_col,
    right_on='Date',
    direction='backward'
)

print(f"IPOs after merge: {len(df)}")
df = df.dropna(subset=nifty_features)


# ==============================
# FEATURES + TARGET
# ==============================

ipo_features = ['Issue_Size(crores)', 'QIB', 'HNI', 'RII','Total', 'Offer Price']
all_features = ipo_features + nifty_features

X = df[all_features].apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.mean())

df['Listing Gain'] = pd.to_numeric(df['Listing Gain'], errors='coerce')
df = df.dropna(subset=['Listing Gain'])
y = (df['Listing Gain'] > 0).astype(int)
X = X.loc[y.index]

print(f"\nClass balance    : {y.value_counts().to_dict()}")


# ==============================
# TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")


# ==============================
# RANDOM FOREST MODEL
# No scaling needed — tree-based models are scale-invariant
# ==============================

model = RandomForestClassifier(
    n_estimators=200,       # Number of trees
    max_depth=6,            # Max depth per tree — prevents overfitting
    min_samples_split=10,   # Min samples to split a node
    min_samples_leaf=5,     # Min samples at leaf — smooths predictions
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)


# ==============================
# EVALUATION
# ==============================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n===== MODEL PERFORMANCE =====")
print(f"Accuracy  : {round(accuracy_score(y_test, y_pred) * 100, 2)}%")
print(f"ROC AUC   : {round(roc_auc_score(y_test, y_prob), 4)}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Gain", "Gain"]))


# ==============================
# FEATURE IMPORTANCE
# Unique to tree-based models — shows which features matter most
# ==============================

print("\n===== FEATURE IMPORTANCE =====")
feat_imp = pd.Series(model.feature_importances_, index=all_features).sort_values(ascending=False)
for feat, val in feat_imp.items():
    print(f"  {feat}: {round(val, 3)}")


# ==============================
# PLOT — ROC Curve + Feature Importance
# ==============================

from sklearn.metrics import roc_curve

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc_score   = roc_auc_score(y_test, y_prob)

axes[0].plot(fpr, tpr, color='forestgreen', lw=2, label=f"AUC = {round(auc_score, 4)}")
axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
axes[0].set_title("ROC Curve - Random Forest", fontsize=13)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend(loc="lower right")

# --- Feature Importance Bar Chart ---
feat_imp_sorted = feat_imp.sort_values()
axes[1].barh(feat_imp_sorted.index, feat_imp_sorted.values, color='forestgreen', alpha=0.8)
axes[1].set_title("Feature Importance - Random Forest", fontsize=13)
axes[1].set_xlabel("Importance Score")
axes[1].set_ylabel("Feature")

plt.tight_layout()
plt.show()


# ==============================
# PREDICTIONS VS ACTUAL TABLE
# ==============================

results_df = pd.DataFrame({
    "Actual":      y_test.values,
    "Predicted":   y_pred,
    "Probability": np.round(y_prob, 4),
    "Correct":     np.where(y_test.values == y_pred, "✓", "✗")
})

print("\n===== PREDICTIONS VS ACTUAL (first 20) =====")
print(results_df.head(20).to_string(index=False))


# ==============================
# SAMPLE PREDICTION
# ==============================

latest_nifty = nifty[nifty_features].iloc[-1]

sample = pd.DataFrame([{
    'Issue_Size(crores)': 1200,
    'QIB':                10,
    'HNI':                5,
    'RII':                3,
    'Total':              5,
    'Offer Price':        450,
    **latest_nifty.to_dict()
}])

prediction = model.predict(sample)[0]
prob       = model.predict_proba(sample)[0][1]

print("\n===== SAMPLE PREDICTION (with live Nifty context) =====")
print(f"  Market RSI:          {round(float(latest_nifty['Nifty_RSI']), 1)}")
print(f"  Market 20d Return:   {round(float(latest_nifty['Nifty_Ret_20d']), 2)}%")
print(f"  Market 50d Return:   {round(float(latest_nifty['Nifty_Ret_50d']), 2)}%")
print(f"  Volatility (20d):    {round(float(latest_nifty['Nifty_Vol_20d']), 2)}%")
print(f"  Above 50-day MA:     {'Yes' if latest_nifty['Above_MA50']  else 'No'}")
print(f"  Above 200-day MA:    {'Yes' if latest_nifty['Above_MA200'] else 'No'}")
print(f"\n  Prediction:          {'Gain ✓' if prediction == 1 else 'No Gain ✗'}")
print(f"  Probability of Gain: {round(prob * 100, 2)}%")
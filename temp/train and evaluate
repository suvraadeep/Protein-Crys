import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mean_absolute_error, r2_score, f1_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

def make_splits(X, df, n_clusters, pca_dims):
    df = df.copy()
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42).fit(X[:, :pca_dims])
    df["cluster"] = kmeans.labels_
    
    train_idx, tmp_idx = train_test_split(np.arange(len(df)), test_size=0.4, random_state=42, stratify=df["cluster"])
    val_idx, test_idx = train_test_split(tmp_idx, test_size=0.5, random_state=42, stratify=df.iloc[tmp_idx]["cluster"])
    
    return train_idx, val_idx, test_idx

def train_ensemble_regressor(X_tr, y_tr, X_va, y_va):
    m1 = xgb.XGBRegressor(n_estimators=800, learning_rate=0.02, max_depth=6, device="cuda").fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    m2 = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.02, max_depth=6, verbose=-1).fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
    m3 = cb.CatBoostRegressor(iterations=800, learning_rate=0.02, depth=6, task_type="GPU", verbose=0).fit(X_tr, y_tr, eval_set=(X_va, y_va))
    
    return lambda X: (m1.predict(X) + m2.predict(X) + m3.predict(X)) / 3

def train_ensemble_classifier(X_tr, y_tr, X_va, y_va, sw_tr):
    m1 = xgb.XGBClassifier(n_estimators=800, learning_rate=0.02, num_class=3, device="cuda").fit(X_tr, y_tr, sample_weight=sw_tr, eval_set=[(X_va, y_va)], verbose=False)
    m2 = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.02, num_class=3, verbose=-1).fit(X_tr, y_tr, sample_weight=sw_tr, eval_set=[(X_va, y_va)])
    m3 = cb.CatBoostClassifier(iterations=800, learning_rate=0.02, task_type="GPU", verbose=0).fit(X_tr, y_tr, sample_weight=sw_tr, eval_set=(X_va, y_va))
    
    return lambda X: ((m1.predict_proba(X) + m2.predict_proba(X) + m3.predict_proba(X)) / 3).argmax(axis=1)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def bin_temp(k):
    c = k - 273.15
    return 0 if c <= 10 else (1 if c <= 20 else 2)

def save_plots(y_true_r, y_pred_r, y_true_c, y_pred_c, class_names, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#0f1117")
    
    axes[0].scatter(y_true_r, y_pred_r, alpha=0.3, s=10, color="#4ecdc4")
    axes[0].set_title("Regression Accuracy", color="white")
    
    axes[1].hist(y_pred_r - y_true_r, bins=40, color="#ff6b6b")
    axes[1].set_title("Residuals", color="white")
    
    cm = confusion_matrix(y_true_c, y_pred_c)
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrBr", xticklabels=class_names, yticklabels=class_names, ax=axes[2])
    
    plt.savefig(save_path, dpi=150, facecolor="#0f1117")
    plt.close()

from src.data_loader import fetch_data
from src.features import extract_esm, build_feature_matrix
from src.trainer import make_splits, train_ensemble_regressor, train_ensemble_classifier
from src.utils import bin_temp, save_plots
from sklearn.utils.class_weight import compute_sample_weight
import os

CONFIG = {
    "SAVE_DIR": "outputs/", "TARGET_N": 10000, "PCA_DIMS": 200, "N_CLUSTERS": 12,
    "BATCH_SIZE": 8, "TEMP_MIN": 250, "TEMP_MAX": 310,
    "CLASS_NAMES": ["Cold", "Room", "Warm"],
    "AMINO_ACIDS": list("ACDEFGHIKLMNPQRSTVWY")
}

os.makedirs(CONFIG["SAVE_DIR"], exist_ok=True)

df = fetch_data(CONFIG["TARGET_N"], CONFIG["SAVE_DIR"], CONFIG["TEMP_MIN"], CONFIG["TEMP_MAX"])
X_esm = extract_esm(df["sequence"].tolist(), CONFIG["SAVE_DIR"], CONFIG["BATCH_SIZE"])
X = build_feature_matrix(X_esm, df["sequence"].tolist(), CONFIG["AMINO_ACIDS"], CONFIG["PCA_DIMS"], CONFIG["SAVE_DIR"])

tr, va, te = make_splits(X, df, CONFIG["N_CLUSTERS"], CONFIG["PCA_DIMS"])
df["temp_class"] = df["temp_k"].apply(bin_temp)

y_tr_r, y_va_r, y_te_r = df.iloc[tr]["temp_k"].values, df.iloc[va]["temp_k"].values, df.iloc[te]["temp_k"].values
y_tr_c, y_va_c, y_te_c = df.iloc[tr]["temp_class"].values, df.iloc[va]["temp_class"].values, df.iloc[te]["temp_class"].values

reg_predict = train_ensemble_regressor(X[tr], y_tr_r, X[va], y_va_r)
clf_predict = train_ensemble_classifier(X[tr], y_tr_c, X[va], y_va_c, compute_sample_weight("balanced", y_tr_c))

save_plots(y_te_r, reg_predict(X[te]), y_te_c, clf_predict(X[te]), CONFIG["CLASS_NAMES"], f"{CONFIG['SAVE_DIR']}/results.png")

import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

def bin_temp(k):
    c = k - 273.15
    return 0 if c <= 10 else (1 if c <= 20 else 2)


def prepare_data(X, df):
    df = df.copy()
    df["temp_class"] = df["temp_k"].apply(bin_temp)

    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.2, random_state=42)

    return df, tr, te


def train_models(X_tr, y_tr_r, y_tr_c):
    sw = compute_sample_weight("balanced", y_tr_c)

    xgb_reg = xgb.XGBRegressor(n_estimators=500, learning_rate=0.03)
    xgb_reg.fit(X_tr, y_tr_r)

    lgb_reg = lgb.LGBMRegressor(n_estimators=500)
    lgb_reg.fit(X_tr, y_tr_r)

    cb_reg = cb.CatBoostRegressor(iterations=500, verbose=0)
    cb_reg.fit(X_tr, y_tr_r)

    xgb_clf = xgb.XGBClassifier(n_estimators=500)
    xgb_clf.fit(X_tr, y_tr_c, sample_weight=sw)

    lgb_clf = lgb.LGBMClassifier(n_estimators=500)
    lgb_clf.fit(X_tr, y_tr_c, sample_weight=sw)

    cb_clf = cb.CatBoostClassifier(iterations=500, verbose=0)
    cb_clf.fit(X_tr, y_tr_c, sample_weight=sw)

    return (xgb_reg, lgb_reg, cb_reg,
            xgb_clf, lgb_clf, cb_clf)


def evaluate(models, X_te, y_te_r, y_te_c):
    xgb_reg, lgb_reg, cb_reg, xgb_clf, lgb_clf, cb_clf = models

    reg_pred = (
        xgb_reg.predict(X_te) +
        lgb_reg.predict(X_te) +
        cb_reg.predict(X_te)
    ) / 3

    clf_pred = np.argmax(
        xgb_clf.predict_proba(X_te) +
        lgb_clf.predict_proba(X_te) +
        cb_clf.predict_proba(X_te),
        axis=1
    )

    print("MAE:", mean_absolute_error(y_te_r, reg_pred))
    print("R2 :", r2_score(y_te_r, reg_pred))
    print("F1 :", f1_score(y_te_c, clf_pred, average="weighted")) 

from data_pipeline import fetch_data
from feature_engineering import extract_esm, build_phys_matrix, build_feature_matrix
from train_and_evaluate import prepare_data, train_models, evaluate

SAVE_DIR = "outputs"
TARGET_N = 10000

def main():
    df = fetch_data(TARGET_N, SAVE_DIR)

    seqs = df["sequence"].tolist()

    X_esm = extract_esm(seqs, SAVE_DIR)
    X_phys = build_phys_matrix(seqs, SAVE_DIR)
    X = build_feature_matrix(X_esm, X_phys, SAVE_DIR)

    df, tr, te = prepare_data(X, df)

    X_tr, X_te = X[tr], X[te]
    y_tr_r = df.iloc[tr]["temp_k"].values
    y_te_r = df.iloc[te]["temp_k"].values
    y_tr_c = df.iloc[tr]["temp_class"].values
    y_te_c = df.iloc[te]["temp_class"].values

    models = train_models(X_tr, y_tr_r, y_tr_c)
    evaluate(models, X_te, y_te_r, y_te_c)

if __name__ == "__main__":
    main()

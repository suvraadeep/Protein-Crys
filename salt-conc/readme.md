Terminal :
Hand-Crafted    → MAE: 0.4041 M | R²: 0.4453
  ESM-2         → MAE: 0.4474 M | R²: 0.3305

Dataset: 4236 samples, 14 salt types

  MODEL A — Salt Type Classifier Results:
    Hand-Crafted → Best model: XGBoost      | Test Acc: 28.2% | CV Acc: 38.9%
    ESM-2        → Best model: XGBoost      | Test Acc: 34.1% | CV Acc: 39.6%
    Combined     → Best model: XGBoost      | Test Acc: 34.2% | CV Acc: 39.2%

  Best overall classifier: ESM-2 features (XGBoost)

  MODEL B — Concentration Regressor:
    MAE:  0.4156 M
    RMSE: 0.6355 M
    R²:   0.4480

  MODEL A — Deep Learning Classifier:
    Feature set: DL Combined
    Test Accuracy: 31.4%
    Best validation accuracy: 30.8%

  FEATURE COMPARISON:
    Hand-Crafted    → MAE: 0.4041 M | R²: 0.4453
    ESM-2           → MAE: 0.4474 M | R²: 0.3305

from joblib import load

pipeline = load("results/metrics/tpot_exported_pipeline.joblib")

def predict(X):
    return pipeline.predict(X)

def predict_proba(X):
    return pipeline.predict_proba(X)

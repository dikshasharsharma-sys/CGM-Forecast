import joblib, pandas as pd, json, traceback, sys
from pathlib import Path

cfg = json.loads(Path('reports/feature_config.json').read_text())
feature_columns = cfg['feature_columns']
print('Features:', len(feature_columns))
inp = {f: 110.0 for f in feature_columns}
df = pd.DataFrame([inp])
print('DataFrame shape:', df.shape)

try:
    m = joblib.load('models/AZT1D_to_AZT1D_30m_HistGBM.joblib')
    print('Model type:', type(m).__name__)
    if hasattr(m, 'steps'):
        print('Pipeline steps:', [s[0] for s in m.steps])
    if hasattr(m, 'feature_names_in_'):
        fn = list(m.feature_names_in_)
        print('Model expects features:', len(fn), fn[:5])
        missing = [f for f in fn if f not in df.columns]
        extra = [f for f in df.columns if f not in fn]
        print('Missing from df:', missing)
        print('Extra in df:', extra)
    pred = m.predict(df)[0]
    print('Pred:', pred)
except Exception:
    traceback.print_exc()

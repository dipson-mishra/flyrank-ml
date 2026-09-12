from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

repo = Path(__file__).resolve().parents[1]
print('repo root:', repo)
p = repo / 'data' / 'raw' / 'content_refresh_anonymized.csv'
if not p.exists():
    raise SystemExit('starter CSV missing at ' + str(p))

df = pd.read_csv(p)
print('rows,cols:', df.shape)
df['is_declining_label'] = df['trend_direction'].fillna('').str.lower() == 'down'
print('label counts:\n', df['is_declining_label'].value_counts(dropna=False))

# Honest grouped split
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_ix, test_ix = next(gss.split(df, groups=df['client_id']))
print('train rows, test rows:', len(train_ix), len(test_ix))
print('train label rate:', df.loc[train_ix, 'is_declining_label'].mean())
print('test label rate:', df.loc[test_ix, 'is_declining_label'].mean())

# Leakage probes
leaky = [c for c in df.columns if 'trend' in c.lower()]
print('columns containing trend:', leaky)
num = df.select_dtypes(include=['number']).copy()
if 'is_declining_label' in num.columns:
    num = num.drop(columns=['is_declining_label'])
corrs = num.corrwith(df['is_declining_label']).abs().sort_values(ascending=False)
print('Top numeric correlations with label (abs):')
print(corrs.head(10))

bucket = df.groupby('age_tier')['is_declining_label'].agg(['mean','count']).rename(columns={'mean':'rate'})
print('age_tier rates (showing n):')
print(bucket.sort_values('count', ascending=False))

# Export a small probe
out = repo / 'work' / 'outputs'
out.mkdir(parents=True, exist_ok=True)
probe = df[['content_id','client_id','is_declining_label']].sample(min(200, len(df)))
probe.to_csv(out / 'w06_probe.csv', index=False)
print('W06 probe exported to', out / 'w06_probe.csv')

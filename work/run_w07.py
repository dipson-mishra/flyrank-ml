from pathlib import Path
import pandas as pd
import numpy as np

repo = Path(__file__).resolve().parents[1]
p = repo / 'data' / 'raw' / 'content_refresh_anonymized.csv'
if not p.exists():
    raise SystemExit('starter CSV missing at ' + str(p))

df = pd.read_csv(p)
df['is_declining_label'] = df['trend_direction'].fillna('').str.lower() == 'down'
# Rule: stale AND visible
df['stale'] = (df['days_since_last_update'] >= 180).astype(int)
df['visible'] = (df['impressions_90d'] >= 500).astype(int)
df['baseline_score'] = df['stale'] * df['visible'] * df['impressions_90d']
# reason codes
def reason(row):
    codes = []
    if row['stale'] and row['visible']:
        codes.append('stale_visible')
    elif row['stale']:
        codes.append('stale_only')
    elif row['visible']:
        codes.append('visible_only')
    return '|'.join(codes) if codes else 'none'

df['reason_code'] = df.apply(reason, axis=1)

# precision@K
from numpy import argsort

def precision_at_k(scores, labels, k):
    order = argsort(-np.asarray(scores))
    return np.asarray(labels)[order[:k]].mean()

K = 50
base_rate = df['is_declining_label'].mean()
p_at_k = precision_at_k(df['baseline_score'], df['is_declining_label'], K)
print(f'base rate: {base_rate:.3f}, precision@{K}: {p_at_k:.3f}')

out = repo / 'work' / 'outputs'
out.mkdir(parents=True, exist_ok=True)
topk = df.sort_values('baseline_score', ascending=False).head(200).copy()
topk.to_csv(out / 'baseline_refresh_queue.csv', index=False)
print('Exported top picks to', out / 'baseline_refresh_queue.csv')

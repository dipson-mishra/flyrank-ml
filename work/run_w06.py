from pathlib import Path
import os
import duckdb
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

repo = Path(__file__).resolve().parents[1]


def load_hf_token():
    candidates = [
        repo / '.env',
        repo.parent / '.env',
        repo.parent.parent / '.env',
    ]
    for path in candidates:
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip().startswith('HF_TOKEN='):
                    return line.split('=', 1)[1].strip().strip('"').strip("'")
    return os.environ.get('HF_TOKEN')


token = load_hf_token()
if not token:
    raise SystemExit('HF_TOKEN was not found in .env or the environment.')

con = duckdb.connect()
con.execute('CREATE OR REPLACE SECRET hf (TYPE HUGGINGFACE, TOKEN ?)', [token])

FACT = "hf://datasets/FlyRank/internship-warehouse/fact_content_daily_performance/month=2026-03/*.parquet"
CONTENT = "hf://datasets/FlyRank/internship-warehouse/dim_content.parquet"

query = """
WITH daily AS (
  SELECT content_hash_id, client_hash_id,
         SUM(COALESCE(gsc_impressions, 0)) AS impressions,
         SUM(COALESCE(gsc_clicks, 0)) AS clicks,
         SUM(COALESCE(gsc_sum_position, 0)) AS sum_position,
         SUM(COALESCE(ga4_sessions, 0)) AS sessions,
         SUM(COALESCE(ga4_engaged_sessions, 0)) AS engaged_sessions,
         SUM(COALESCE(scroll_events, 0)) AS scroll_events,
         SUM(COALESCE(ga4_pageviews, 0)) AS pageviews,
         SUM(CASE WHEN report_date >= '2026-03-16' THEN COALESCE(gsc_clicks, 0) ELSE 0 END) AS clicks_h2,
         SUM(CASE WHEN report_date < '2026-03-16' THEN COALESCE(gsc_clicks, 0) ELSE 0 END) AS clicks_h1
  FROM read_parquet(?)
  WHERE gsc_data_available IS TRUE
  GROUP BY content_hash_id, client_hash_id
)
SELECT d.*, c.content_type, c.main_intent, c.word_count,
       d.clicks * 100.0 / NULLIF(d.impressions, 0) AS ctr,
       d.sum_position * 1.0 / NULLIF(d.impressions, 0) AS avg_position,
       CASE WHEN d.clicks_h2 < d.clicks_h1 THEN 1 ELSE 0 END AS is_declining_label
FROM daily d
LEFT JOIN read_parquet(?) c USING (content_hash_id)
WHERE d.impressions >= 500
"""

df = con.execute(query, [FACT, CONTENT]).df()
print('rows,cols:', df.shape)
print('label counts:\n', df['is_declining_label'].value_counts(dropna=False))

# Honest grouped split
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_ix, test_ix = next(gss.split(df, groups=df['client_hash_id']))
print('train rows, test rows:', len(train_ix), len(test_ix))
print('train label rate:', df.loc[train_ix, 'is_declining_label'].mean())
print('test label rate:', df.loc[test_ix, 'is_declining_label'].mean())

# Leakage probes
num = df.select_dtypes(include=['number']).copy()
if 'is_declining_label' in num.columns:
    num = num.drop(columns=['is_declining_label'])
num = num.select_dtypes(include=['number'])
correlations = num.corrwith(df['is_declining_label']).abs().sort_values(ascending=False)
print('Top numeric correlations with label (abs):')
print(correlations.head(10))

bucket = df.groupby('content_type')['is_declining_label'].agg(['mean', 'count']).rename(columns={'mean': 'rate'})
print('content_type rates (showing n):')
print(bucket.sort_values('count', ascending=False).head(10))

# Export a small probe
out = repo / 'work' / 'outputs'
out.mkdir(parents=True, exist_ok=True)
probe = df[['content_hash_id', 'client_hash_id', 'is_declining_label']].sample(min(200, len(df)))
probe.to_csv(out / 'w06_probe.csv', index=False)
print('W06 probe exported to', out / 'w06_probe.csv')

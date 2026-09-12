from pathlib import Path
import os
import duckdb
import pandas as pd
import numpy as np

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

# Transparent rule-based baseline: low-visibility pages with poor position are prioritized
# Reason codes surface the simple logic a human can trust.
df['position_sinking'] = (df['avg_position'] >= 10).astype(int)
df['visible'] = (df['impressions'] >= 500).astype(int)
df['baseline_score'] = df['position_sinking'] * df['visible'] * df['impressions']


def reason(row):
    codes = []
    if row['position_sinking'] and row['visible']:
        codes.append('position_sinking_visible')
    elif row['position_sinking']:
        codes.append('position_sinking_only')
    elif row['visible']:
        codes.append('visible_only')
    return '|'.join(codes) if codes else 'none'


df['reason_code'] = df.apply(reason, axis=1)


def precision_at_k(scores, labels, k):
    order = np.argsort(-np.asarray(scores))
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

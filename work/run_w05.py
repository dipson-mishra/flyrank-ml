from pathlib import Path
import os
import duckdb
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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


def precision_at_k(y_true, y_prob, k=50):
    df_eval = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    top_k = df_eval.sort_values('y_prob', ascending=False).head(k)
    return top_k['y_true'].mean()


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
       CASE WHEN d.clicks_h2 < d.clicks_h1 THEN 1 ELSE 0 END AS is_declining
FROM daily d
LEFT JOIN read_parquet(?) c USING (content_hash_id)
WHERE d.impressions >= 500
"""

df = con.execute(query, [FACT, CONTENT]).df()
df['engagement_rate'] = (df['engaged_sessions'] * 100.0 / df['sessions'].replace(0, np.nan)).fillna(0)
df['scroll_rate'] = (df['scroll_events'] * 100.0 / df['pageviews'].replace(0, np.nan)).fillna(0)
df['word_count'] = df['word_count'].fillna(df['word_count'].median())

df['position_tier'] = pd.cut(df['avg_position'], bins=[0, 3, 10, 20, 50, float('inf')], labels=['top_3', 'page_1', 'striking', 'page_3_5', 'deep'])
ref = df.groupby('position_tier', observed=True)['ctr'].median().rename('expected_ctr')
df = df.join(ref, on='position_tier')
df['opportunity_score'] = (df['expected_ctr'] - df['ctr']).clip(lower=0) * (1 + df['impressions']).pow(0.5)

features = ['impressions', 'avg_position', 'word_count', 'engagement_rate', 'scroll_rate']
X = df[features]
y = df['is_declining']
groups = df['client_hash_id']

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
base_scores_test = df.iloc[test_idx]['opportunity_score']

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
model_probs = rf.predict_proba(X_test)[:, 1]

metrics = {
    'Metric': ['Base Rate', 'Precision@50 (Baseline)', 'Precision@50 (Model)'],
    'Score': [
        y_test.mean(),
        precision_at_k(y_test, base_scores_test, 50),
        precision_at_k(y_test, model_probs, 50),
    ],
}
print(pd.DataFrame(metrics).round(3))

import os
import duckdb
from pathlib import Path

def load_token():
    for path in [Path.cwd() / '.env', Path.cwd().parent / '.env', Path.cwd().parent.parent / '.env']:
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip().startswith('HF_TOKEN='):
                    return line.split('=', 1)[1].strip().strip(chr(34)).strip(chr(39))
    return os.environ.get('HF_TOKEN')

HF_TOKEN = load_token()
if not HF_TOKEN:
    print("HF_TOKEN not found")
    exit(1)

conn = duckdb.connect(database=":memory:")
conn.execute("INSTALL httpfs; LOAD httpfs;")
conn.execute("CREATE SECRET (TYPE HUGGINGFACE, TOKEN ?)", [HF_TOKEN])

try:
    print("Checking dim_content columns:")
    print(conn.execute("DESCRIBE 'hf://datasets/FlyRank/internship-warehouse/dim_content/**/*.parquet'").df())
except Exception as e:
    print(f"Error checking dim_content: {e}")

try:
    print("\nChecking fact_content_daily_performance columns:")
    print(conn.execute("DESCRIBE 'hf://datasets/FlyRank/internship-warehouse/fact_content_daily_performance/**/*.parquet'").df())
except Exception as e:
    print(f"Error checking fact_content_daily_performance: {e}")

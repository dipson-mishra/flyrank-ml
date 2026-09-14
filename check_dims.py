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

# Try different possible paths for dim_content
paths = [
    'hf://datasets/FlyRank/internship-warehouse/dim_content.parquet',
    'hf://datasets/FlyRank/internship-warehouse/dim_content/*.parquet',
    'hf://datasets/FlyRank/internship-warehouse/dim_content/**/*.parquet',
]

for path in paths:
    try:
        print(f"Trying {path}...")
        print(conn.execute(f"DESCRIBE '{path}'").df())
        print("Success!")
        break
    except Exception as e:
        print(f"Failed: {e}")

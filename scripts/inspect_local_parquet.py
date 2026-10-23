import os
import sys
from pathlib import Path
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.paths import resolve_parquet_root

DATA_ROOT = resolve_parquet_root()

files = sorted(DATA_ROOT.glob("*.parquet"))

if not files:
    raise FileNotFoundError(f"No parquet files found in {DATA_ROOT}")

print("=" * 70)
print("UTTA-Med: Local Camelyon17 Parquet Inspection")
print("=" * 70)

print(f"Data root: {DATA_ROOT}")
print(f"Number of parquet files: {len(files)}")

for path in files:
    print(f"  {path.name}")

print("\nInspecting first shard:")
first_file = files[0]

parquet_file = pq.ParquetFile(first_file)

print(f"File: {first_file.name}")
print(f"Rows: {parquet_file.metadata.num_rows}")

print("\nSchema:")
print(parquet_file.schema_arrow)

print("\nFirst record:")

table = pq.read_table(
    first_file,
    columns=[
        "label",
        "center",
        "image_id",
        "patient",
        "node",
        "x_coord",
        "y_coord",
        "slide",
    ],
    use_threads=True,
)

row = table.slice(0, 1).to_pydict()

for key, value in row.items():
    print(f"{key}: {value}")

print("=" * 70)
print("Inspection complete.")
print("=" * 70)

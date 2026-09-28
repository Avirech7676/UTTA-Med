import os
from pathlib import Path
import pyarrow.parquet as pq

default_path = Path(r"C:\Users\avina\OneDrive\Desktop\Camelyon17-Data\data")
env_root = os.environ.get("CAMELYON17_ROOT")
if env_root:
    DATA_ROOT = Path(env_root) / "data" if (Path(env_root) / "data").exists() else Path(env_root)
elif default_path.exists():
    DATA_ROOT = default_path
else:
    raise RuntimeError(
        "Camelyon17 data root not found. Please set CAMELYON17_ROOT environment variable."
    )

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

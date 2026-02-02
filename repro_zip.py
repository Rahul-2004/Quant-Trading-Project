
from pathlib import Path
import zipfile
import polars as pl

file_path = Path("data/BTCUSDT-trades-2024-10-29.zip")
print(f"Testing file: {file_path}")
print(f"Exists: {file_path.exists()}")
print(f"Absolute: {file_path.absolute()}")

try:
    with zipfile.ZipFile(file_path) as z:
        print(f"Zip opened. Files: {z.namelist()}")
        csv_files = [f for f in z.namelist() if f.endswith('.csv')]
        if csv_files:
            print(f"Reading {csv_files[0]}...")
            with z.open(csv_files[0]) as f:
                trades = pl.read_csv(f.read(), try_parse_dates=True)
                print(f"Read {len(trades)} rows.")
                print(trades.head())
        else:
            print("No CSV found in zip.")
except Exception as e:
    print(f"Error: {e}")

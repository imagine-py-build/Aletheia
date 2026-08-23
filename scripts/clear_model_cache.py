"""Safely clear Aletheia's local model cache after confirmation."""
from pathlib import Path
import os, shutil

root = Path(os.getenv("MODEL_DIR", ".local/model-cache"))
if not root.exists():
    print("Nothing to clear.")
    raise SystemExit(0)

files = [p for p in root.rglob("*") if p.is_file()]
total = sum(p.stat().st_size for p in files)
print(f"Cache: {root.resolve()}")
print(f"Size: {total / (1024**3):.2f} GB")
answer = input("Delete ALL downloaded model files from this cache? [y/N]: ").strip().lower()
if answer != "y":
    print("Cancelled.")
    raise SystemExit(0)

for child in root.iterdir():
    if child.is_dir():
        shutil.rmtree(child)
    else:
        child.unlink()
print("Model cache cleared. Models will be downloaded again only when an analysis needs them.")

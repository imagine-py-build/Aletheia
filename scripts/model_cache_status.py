"""Show the size and contents of Aletheia's persistent model cache."""
from pathlib import Path
import os

root = Path(os.getenv("MODEL_DIR", ".local/model-cache"))
if not root.exists():
    print(f"Model cache does not exist yet: {root}")
    raise SystemExit(0)

files = [p for p in root.rglob("*") if p.is_file()]
total = sum(p.stat().st_size for p in files)
print(f"Model cache: {root.resolve()}")
print(f"Files: {len(files)}")
print(f"Size: {total / (1024**3):.2f} GB")
for p in sorted(files, key=lambda x: x.stat().st_size, reverse=True)[:20]:
    print(f"{p.stat().st_size / (1024**2):8.1f} MB  {p.relative_to(root)}")

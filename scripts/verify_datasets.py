from pathlib import Path
for task in ['image','audio','video','nlp']:
 p=Path('datasets')/task; print(task, 'OK' if p.exists() else 'MISSING')

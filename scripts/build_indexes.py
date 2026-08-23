from pathlib import Path
import hashlib,json
for task in ['image','audio','video']:
 items=[]
 for p in (Path('datasets')/task).rglob('*'):
  if p.is_file(): items.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 Path('datasets').mkdir(exist_ok=True); (Path('datasets')/f'{task}_index.json').write_text(json.dumps(items,indent=2))
 print(task,len(items))

import argparse,shutil
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--task',required=True,choices=['image','audio','video','nlp']);p.add_argument('--source');p.add_argument('--target');a=p.parse_args(); target=Path(a.target or f'datasets/{a.task}');target.mkdir(parents=True,exist_ok=True)
 if a.source:
  src=Path(a.source)
  for x in src.iterdir(): shutil.copytree(x,target/x.name,dirs_exist_ok=True) if x.is_dir() else shutil.copy2(x,target/x.name)
 print(f'Prepared {target}. Expected formats are documented in README.md.')
if __name__=='__main__':main()

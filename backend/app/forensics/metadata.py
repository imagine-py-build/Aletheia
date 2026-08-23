import json, subprocess
from pathlib import Path
from PIL import Image

def extract_metadata(path):
    result={}
    try:
        p=subprocess.run(['exiftool','-j',str(path)],capture_output=True,text=True,timeout=20)
        if p.returncode==0: result.update(json.loads(p.stdout)[0])
    except (FileNotFoundError, subprocess.SubprocessError): pass
    try:
        with Image.open(path) as im:
            result.setdefault('format',im.format); result.setdefault('width',im.width); result.setdefault('height',im.height)
            result.setdefault('mode',im.mode)
    except Exception: pass
    return result

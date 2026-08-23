import hashlib, mimetypes, os, uuid
from pathlib import Path
from datetime import datetime, timezone
from backend.app.core.config import settings

def hash_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def save_upload(content:bytes, filename:str, incident_id:str):
    safe=Path(filename).name
    folder=Path(settings.storage_dir)/incident_id
    folder.mkdir(parents=True, exist_ok=True)
    target=folder/f'{uuid.uuid4()}_{safe}'
    target.write_bytes(content)
    return str(target), hash_file(target), mimetypes.guess_type(safe)[0] or 'application/octet-stream'

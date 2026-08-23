import hashlib, mimetypes, os, uuid
from pathlib import Path
from datetime import datetime, timezone
from backend.app.core.config import settings

def hash_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def _detect_mime(content: bytes, filename: str) -> str:
    safe = Path(filename).name
    guessed = mimetypes.guess_type(safe)[0]

    # Detect common image formats from their actual file signatures first.
    if content.startswith(b"\\xff\\xd8\\xff"):
        return "image/jpeg"
    if content.startswith(b"\\x89PNG\\r\\n\\x1a\\n"):
        return "image/png"
    if content.startswith(b"RIFF") and len(content) >= 12 and content[8:12] == b"WEBP":
        return "image/webp"
    if content.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if content.startswith(b"BM"):
        return "image/bmp"

    # Extension fallback for supported image uploads.
    image_mimes = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".avif": "image/avif",
    }
    ext = Path(safe).suffix.lower()
    if ext in image_mimes:
        return image_mimes[ext]

    return guessed or "application/octet-stream"


def save_upload(content:bytes, filename:str, incident_id:str):
    safe=Path(filename).name
    folder=Path(settings.storage_dir)/incident_id
    folder.mkdir(parents=True, exist_ok=True)
    target=folder/f'{uuid.uuid4()}_{safe}'
    target.write_bytes(content)
    return str(target), hash_file(target), _detect_mime(content, safe)

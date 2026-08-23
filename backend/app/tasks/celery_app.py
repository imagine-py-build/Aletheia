from celery import Celery
from backend.app.core.config import settings
celery=Celery('aletheia',broker=settings.redis_url,backend=settings.redis_url)

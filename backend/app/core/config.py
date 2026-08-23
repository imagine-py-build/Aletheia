from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    app_env: str = 'development'
    database_url: str = 'sqlite:///./aletheia.db'
    redis_url: str = 'redis://localhost:6379/0'
    mlflow_uri: str = 'http://localhost:5000'
    storage_dir: str = './storage'
    model_dir: str = './models'
    dataset_dir: str = './datasets'
    secret_key: str = 'change-me'
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

settings = Settings()
Path(settings.storage_dir).mkdir(parents=True, exist_ok=True)

# Operations checklist

1. Preserve the original upload and hash it before analysis.
2. Restrict access to authorized investigators.
3. Never overwrite original evidence.
4. Treat model scores as investigative indicators.
5. Require human verification before a finding becomes verified.
6. Record every access and processing action.
7. Keep dataset/model versions with evaluation metrics.
8. Keep test sets untouched during training.

## Persistent model cache

Model weights are deliberately outside Git. Configure `ALETHEIA_MODEL_CACHE` in `.env` to a stable host directory such as `C:/AletheiaData/models`. The compose file mounts that directory to `/app/models` and sets `HF_HOME`/`TRANSFORMERS_CACHE` there. Restarting or rebuilding containers reuses the existing cache.
